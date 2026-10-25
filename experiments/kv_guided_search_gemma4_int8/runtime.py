"""Gemma 4 text-only adapter; model weights remain quantized during inference."""
import copy
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from mlx_lm.models.cache import QuantizedKVCache
from mlx_lm.utils import load_model
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'model'
MODEL_ID = 'mlx-community/gemma-4-e2b-it-8bit'
REVISION = '03dcf209f3f549b4075e7191e77cf69b3d48e1b2'
LAYERS = (10, 15)  # Independent full-attention KV producers, numbered from one.


def tensor(x):
    return torch.from_numpy(np.array(x.astype(mx.float32), copy=True))


class SharedQuantizedCache:
    """Tell MLX SDPA to consume packed shared KV without appending it again.

    Gemma's shared attention receives the producer's packed tuples. The stock
    implementation otherwise passes cache=None, selecting floating-point SDPA.
    This view has no buffers and does not generate/store an independent KV.
    """
    bits = 8
    group_size = 64

    def update_and_fetch(self, keys, values):
        assert isinstance(keys, tuple) and isinstance(values, tuple)
        return keys, values


class Runtime:
    def __init__(self):
        self.model, self.config = load_model(MODEL, lazy=False)
        self.lm = self.model.language_model
        self.base = self.lm.model
        self.full_layers = self.base.layers
        self.producer_count = self.lm.args.num_hidden_layers - self.lm.args.num_kv_shared_layers
        assert self.producer_count == 15 and len(self.full_layers) == 35
        assert self.base.previous_kvs[34] == 14  # Final layer consumes layer 15 KV.
        self.base_dims = self.lm.args.hidden_size
        self.weight_bits = self.base.embed_tokens.bits
        assert self.weight_bits == self.config['quantization']['bits'] == 8
        for _, module in self.model.named_modules():
            if hasattr(module, 'bits'):
                assert module.bits == 8

    def make_cache(self, mode, full=True):
        if mode == 'native':
            return self.model.make_cache()
        assert mode == 'int8'
        # Short contexts only. A nonrotating quantized cache is mathematically
        # sufficient below the 512-token local window; masks retain window_size.
        cache = [QuantizedKVCache(group_size=64, bits=8) for _ in range(self.producer_count)]
        if full:
            cache += [SharedQuantizedCache() for _ in range(35 - self.producer_count)]
        return cache

    def prepare(self, prefix_ids, mode):
        assert len(prefix_ids) + 1 < self.lm.args.sliding_window
        self.base.layers = self.full_layers
        cache = self.make_cache(mode)
        logits = self.model(mx.array([prefix_ids]), cache=cache)
        mx.eval(logits)
        return cache, logits[0, -1]

    def embeddings(self, ids):
        values = self.base.embed_tokens(mx.array(ids))
        mx.eval(values)
        return tensor(values)

    def metadata(self):
        params = tree_flatten(self.model.parameters())
        reconstructed_parameters = sum(
            p.size * (32 // self.weight_bits if p.dtype == mx.uint32 else 1)
            for name, p in params if not name.endswith(('.scales', '.biases')))
        return {'model_id': MODEL_ID, 'model_revision': REVISION,
                'weight_quantization': self.config['quantization'],
                'model_text_parameters': reconstructed_parameters,
                'embedding_table_shape': [self.lm.args.vocab_size, self.base_dims],
                'per_layer_embedding_table_shape': [self.lm.args.vocab_size_per_layer_input,
                    self.lm.args.num_hidden_layers * self.lm.args.hidden_size_per_layer_input],
                'independent_kv_producers': self.producer_count,
                'layer_to_kv_producer_1based': [i + 1 for i in self.base.previous_kvs],
                'observed_layers_1based': list(LAYERS),
                'kv_width_by_layer': {str(i): 2 * self.full_layers[i-1].self_attn.head_dim
                                      * self.full_layers[i-1].self_attn.n_kv_heads for i in LAYERS},
                'text_only': True, 'main_embedding_only_surrogate': True,
                'device': 'Apple MLX GPU', 'sliding_window': self.lm.args.sliding_window}


def fork_cache(cache, batch_size, layer):
    result = []
    for c in cache[:layer]:
        new = copy.copy(c)
        def expand(x):
            return mx.broadcast_to(x, (batch_size,) + x.shape[1:])
        if isinstance(c, QuantizedKVCache):
            new.keys = tuple(expand(x) for x in c.keys)
            new.values = tuple(expand(x) for x in c.values)
        else:
            new.keys, new.values = expand(c.keys), expand(c.values)
        result.append(new)
    return result


def last_kv(c):
    if isinstance(c, QuantizedKVCache):
        def last(triple):
            return mx.dequantize(*(x[:, :, c.offset-1:c.offset, :] for x in triple),
                                 group_size=c.group_size, bits=c.bits)
        k, v = last(c.keys), last(c.values)
    else:
        k, v = c.keys[:, :, c.offset-1:c.offset, :], c.values[:, :, c.offset-1:c.offset, :]
    n = k.shape[0]
    return tensor(mx.concatenate([k.reshape(n, -1), v.reshape(n, -1)], axis=-1))


def cache_description(c):
    if isinstance(c, QuantizedKVCache):
        return {'format': '8-bit affine packed uint32 + scales + biases',
                'bits': c.bits, 'group_size': c.group_size,
                'packed_dtype': str(c.keys[0].dtype), 'scale_dtype': str(c.keys[1].dtype)}
    return {'format': 'native floating KV', 'dtype': str(c.keys.dtype)}


class Oracle:
    def __init__(self, runtime, prefix_cache, ids, layer, batch_size=64):
        self.runtime, self.prefix_cache, self.ids, self.layer = runtime, prefix_cache, ids, layer
        self.count, self.batch_size = 0, batch_size

    def query(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long)
        self.count += len(indices)
        self.runtime.base.layers = self.runtime.full_layers[:self.layer]
        parts = []
        for chunk in indices.split(self.batch_size):
            cache = fork_cache(self.prefix_cache, len(chunk), self.layer)
            inputs = mx.array(self.ids[chunk].tolist())[:, None]
            hidden = self.runtime.base(inputs, cache=cache)
            mx.eval(hidden)
            parts.append(last_kv(cache[self.layer - 1]))
        return torch.cat(parts)
