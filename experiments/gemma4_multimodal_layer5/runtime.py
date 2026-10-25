"""Gemma 4 W8 multimodal forward map; no modification of installed libraries."""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_vlm import load

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent / "kv_guided_search_gemma4_int8/model"


def array(x):
    return np.array(x.astype(mx.float32))


def nmse(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return float(np.mean((x-y)**2) / max(np.mean(y*y), 1e-12))


def load_runtime():
    model, processor = load(str(MODEL))
    model.eval()
    processor.image_processor.max_soft_tokens = 70
    assert model.config.text_config.layer_types[4] == "full_attention"
    assert model.config.text_config.hidden_size == 1536
    return model, processor


class Layer5Map:
    """Receives public prompt structure, NEVER the original media features."""
    def __init__(self, model, input_ids, media_positions, token_types=None):
        self.model = model
        self.base = model.language_model.model
        self.ids = mx.array(input_ids)
        self.positions = np.array(media_positions, dtype=int)
        assert np.array_equal(self.positions, np.arange(self.positions[0],
                                                        self.positions[-1]+1))
        self.start, self.end = int(self.positions[0]), int(self.positions[-1]+1)
        self.n = self.end-self.start
        self.types = mx.array(token_types) if token_types is not None else None
        public_emb = self.base.embed_tokens(self.ids) * self.base.embed_scale
        # The unknown media slots are not read from any target feature array.
        self.before = public_emb[:, :self.start, :]
        self.after = public_emb[:, self.end:, :]
        ple_ids = mx.concatenate([self.ids[:, :self.start],
                                  mx.zeros((1,self.n), dtype=mx.int32),
                                  self.ids[:, self.end:]], axis=1)
        self.ple = self.base.get_per_layer_inputs(ple_ids)
        self.masks = self.base._make_masks(public_emb, [None]*len(self.base.layers),
                                           self.types)
        self.dtype = public_emb.dtype
        mx.eval(self.before, self.after, self.ple)

    def insert(self, z):
        return mx.concatenate([self.before, z.astype(self.dtype), self.after], axis=1)

    def __call__(self, z):
        h = self.insert(z)
        pli = self.base.project_per_layer_inputs(h, self.ple)
        for i in range(4):
            h, _, _ = self.base.layers[i](h, self.masks[i], None,
                                           per_layer_input=pli[:, :, i, :])
        layer = self.base.layers[4]
        a = layer.self_attn
        h = layer.input_layernorm(h)
        k = a.k_norm(a.k_proj(h).reshape(1, h.shape[1], a.n_kv_heads, a.head_dim))
        k = a.rope(k.transpose(0,2,1,3), offset=0)
        v = a.v_norm(a.v_proj(h).reshape(1, h.shape[1], a.n_kv_heads, a.head_dim))
        v = v.transpose(0,2,1,3)
        return mx.concatenate([k[:, :, self.start:self.end, :].reshape(1,self.n,-1),
                               v[:, :, self.start:self.end, :].reshape(1,self.n,-1)],
                              axis=-1).astype(mx.float32)

    def answer(self, z, max_tokens=64):
        cache = self.model.make_cache()
        out = self.model.language_model(
            inputs_embeds=self.insert(z), per_layer_inputs=self.ple, cache=cache,
            mm_token_type_ids=self.types, logits_to_keep=1)
        ids = []
        for _ in range(max_tokens):
            token = int(mx.argmax(out.logits[0, -1]).item())
            if token in (1, 106, 50): break
            ids.append(token)
            out = self.model.language_model(mx.array([[token]]), cache=cache,
                                            logits_to_keep=1)
        return ids
