"""Gemma 4 media inversion ported from gemma4_multimodal_layer5.

MLX is imported only when a Gemma problem is constructed. This module uses
model parameters, known/recovered structure, and observed KV only; it never
opens original media or evaluation labels. The verified graph is layer 5.
"""
import time
import numpy as np


def validate_structure(input_ids, positions, token_types, soft_id, type_id, vocab_size):
    ids, pos, types = np.asarray(input_ids), np.asarray(positions), np.asarray(token_types)
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype.kind not in 'iu':
        raise ValueError('Gemma input_ids must be a single integer sequence')
    if (pos.ndim != 1 or not len(pos) or pos.dtype.kind not in 'iu' or pos[0] < 0
            or not np.array_equal(pos, np.arange(pos[0], pos[-1]+1))):
        raise ValueError('Media positions must be contiguous and nonnegative')
    # The old simple pilot included a known question after media. Cropping it
    # here ensures a future secret prompt cannot be read by the optimizer.
    if ids.shape[1] != int(pos[-1])+1:
        raise ValueError('Input structure must end at the final media slot; exclude all future text')
    if np.any(ids < 0) or np.any(ids >= vocab_size):
        raise ValueError('Only known/recovered prefix IDs and media placeholders are allowed')
    if types.shape != ids.shape or types.dtype.kind not in 'iu' or not np.isin(types, [0, 1, 3]).all():
        raise ValueError('Explicit text/image/audio token_types are required; video is not supported')
    if not np.all(ids[0, pos] == soft_id) or not np.all(types[0, pos] == type_id):
        raise ValueError('Media placeholder or token type disagrees with modality')
    return ids.copy(), pos.astype(np.int64), types.copy()


class Gemma4Problem:
    """Optimize pre-adapter U, project Z=adapter(U), compare layer-5 K+V."""
    def __init__(self, model, input_ids, media_positions, targets, token_types, *,
                 modality, adapter, adapter_input_dim, steps=3000, lr=.08, seed=17,
                 layer_index=4, prefix_media=()):
        import mlx.core as mx
        self.mx = mx
        if layer_index != 4 or model.config.text_config.layer_types[4] != 'full_attention':
            raise ValueError('This pilot-derived Gemma map supports the layer-5 global producer only')
        if modality not in ('image', 'audio') or not callable(adapter):
            raise ValueError('A supported modality adapter is required')
        if type(steps) is not int or steps < 0 or not np.isfinite(lr) or lr <= 0 or type(seed) is not int or seed < 0:
            raise ValueError('Invalid optimization settings')
        soft_id = model.config.image_token_id if modality == 'image' else model.config.audio_token_id
        type_id = 1 if modality == 'image' else 3
        ids, pos, types = validate_structure(input_ids, media_positions, token_types, soft_id,
                                             type_id, model.config.text_config.vocab_size)
        self.model, self.base, self.adapter = model, model.language_model.model, adapter
        self.positions, self.start, self.end = pos, int(pos[0]), int(pos[-1])+1
        self.n, self.steps, self.lr, self.seed = len(pos), steps, float(lr), seed
        self.input_dim = adapter_input_dim
        self.feature_dim = model.config.text_config.hidden_size
        self.modality, self.ids, self.types = modality, mx.array(ids), mx.array(types)
        self.layer_index = layer_index
        self.attention = self.base.layers[layer_index].self_attn
        if getattr(self.attention, 'is_kv_shared_layer', False):
            raise ValueError('Cannot use a KV consumer layer as the producer map')
        width = 2*self.attention.n_kv_heads*self.attention.head_dim
        target = np.asarray(targets, dtype=np.float32)
        if target.shape == (1, self.n, width):
            target = target[0]
        if target.shape != (self.n, width) or not np.isfinite(target).all():
            raise ValueError('Observed target must contain per-position original-value K+V')
        self.target = mx.array(target[None])
        self.energy = mx.mean(self.target.square())
        if float(self.energy.item()) <= 0:
            raise ValueError('Normalized KV objective requires nonzero target energy')

        public = self.base.embed_tokens(self.ids)*self.base.embed_scale
        # Prior media, if any, must be supplied as already recovered replacement
        # features. Never silently use their placeholder token embeddings.
        covered = np.zeros(self.start, dtype=bool)
        for previous in prefix_media:
            if set(previous) != {'start', 'features'} or type(previous['start']) is not int:
                raise ValueError('Prior recovered media requires start and features')
            start = previous['start']; z = np.asarray(previous['features'])
            if z.ndim != 3 or z.shape[0] != 1 or z.shape[2] != self.feature_dim or not np.isfinite(z).all():
                raise ValueError('Invalid prior recovered media shape')
            end = start + z.shape[1]
            if start < 0 or end > self.start or end <= start or covered[start:end].any() or not (types[0, start:end] != 0).all():
                raise ValueError('Prior media overlaps or includes text/unknown future positions')
            covered[start:end] = True
            public = mx.concatenate([public[:, :start], mx.array(z).astype(public.dtype), public[:, end:]], 1)
        if not np.array_equal(covered, types[0, :self.start] != 0):
            raise ValueError('All preceding media must have a recovered representation')
        self.before = public[:, :self.start]
        ple_ids = ids.copy(); ple_ids[types != 0] = 0
        self.ple = self.base.get_per_layer_inputs(mx.array(ple_ids))
        self.masks = self.base._make_masks(public, [None]*len(self.base.layers), self.types)
        self.dtype = public.dtype
        mx.eval(self.before, self.ple)
        self.initial = np.random.default_rng(seed).standard_normal((1, self.n, adapter_input_dim)).astype(np.float32)
        self.normalize_latent = True
        self._vg = mx.value_and_grad(self.loss)
        self.best_u = None
        self.fit_metadata = dict(algorithm='Gemma4 adapter-constrained layer5 inversion',
            modality=modality, adapter='embed_vision' if modality=='image' else 'embed_audio',
            adapter_input_dim=adapter_input_dim, feature_dim=self.feature_dim,
            layer_index=layer_index, steps=steps, lr=self.lr, seed=seed,
            optimizer='Adam beta1=.9 beta2=.999 eps=1e-8; cosine floor=.15; per-position RMS',
            objective='mean((predicted_KV-observed_KV)^2)/mean(observed_KV^2)',
            proposal_backend='MLX', proposal_kv_dtype='BF16 converted to float32 for loss',
            selection='minimum observed KV NMSE; no original media/features/transcript',
            input_ends_at_media=True, original_media_accessed=False)

    def insert(self, z):
        return self.mx.concatenate([self.before, z.astype(self.dtype)], axis=1)

    def forward(self, z):
        mx = self.mx
        h = self.insert(z)
        pli = self.base.project_per_layer_inputs(h, self.ple)
        for i in range(self.layer_index):
            h, _, _ = self.base.layers[i](h, self.masks[i], None, per_layer_input=pli[:, :, i, :])
        a = self.attention
        h = self.base.layers[self.layer_index].input_layernorm(h)
        k = a.k_norm(a.k_proj(h).reshape(1, h.shape[1], a.n_kv_heads, a.head_dim))
        k = a.rope(k.transpose(0, 2, 1, 3), offset=0)
        v = a.v_norm(a.v_proj(h).reshape(1, h.shape[1], a.n_kv_heads, a.head_dim)).transpose(0, 2, 1, 3)
        # Explicit per-position layout also handles more than one KV head.
        k = k[:, :, self.start:self.end].transpose(0, 2, 1, 3).reshape(1, self.n, -1)
        v = v[:, :, self.start:self.end].transpose(0, 2, 1, 3).reshape(1, self.n, -1)
        return mx.concatenate([k, v], axis=-1).astype(mx.float32)

    def loss(self, u):
        z = self.adapter(u.astype(self.mx.bfloat16))
        return self.mx.mean((self.forward(z)-self.target).square())/self.energy

    def loss_and_grad(self, u):
        value, grad = self._vg(self.mx.array(u))
        self.mx.eval(value, grad)
        return float(value.item()), np.array(grad.astype(self.mx.float32))

    def project(self, u):
        z = self.adapter(self.mx.array(u).astype(self.mx.bfloat16))
        self.mx.eval(z)
        return np.array(z.astype(self.mx.float32))

    def fit_native(self):
        """Preserve the pilot's MLX update arithmetic instead of NumPy updates."""
        mx = self.mx
        u = mx.array(self.initial); m = mx.zeros_like(u); v = mx.zeros_like(u)
        best = float('inf'); best_u = u; trace = []; started = time.perf_counter()
        for i in range(self.steps+1):
            loss, grad = self._vg(u)
            mx.eval(loss, grad)
            value = float(loss.item())
            if not np.isfinite(value):
                raise ValueError('Nonfinite media loss')
            if value < best:
                best, best_u = value, u
            if i % 25 == 0 or i == self.steps:
                trace.append(dict(step=i, kv_nmse=value, best_kv_nmse=best,
                                  seconds=time.perf_counter()-started))
            if i % 500 == 0 or i == self.steps:
                print(f'{self.modality} seed={self.seed} step={i} best_KV_NMSE={best:.8g}', flush=True)
            if i == self.steps:
                break
            g = grad.astype(mx.float32)
            m = .9*m + .1*g; v = .999*v + .001*g.square()
            lr = self.lr*(.15+.85*.5*(1+np.cos(np.pi*i/self.steps)))
            u = u-lr*(m/(1-.9**(i+1)))/(mx.sqrt(v/(1-.999**(i+1)))+1e-8)
            u = u/mx.sqrt(mx.mean(u.square(), axis=-1, keepdims=True)+1e-8)
            mx.eval(u, m, v)
        self.best_u = np.array(best_u.astype(mx.float32))
        return self.project(self.best_u), best, trace

    def replay(self, features):
        """Fresh full-model replay to media end; returns usable cache and logits.

        This is an MLX verification path, not an NPU execution claim.
        """
        mx = self.mx
        z = mx.array(features).astype(self.dtype)
        cache = self.model.make_cache()
        output = self.model.language_model(inputs_embeds=self.insert(z), per_layer_inputs=self.ple,
            cache=cache, mm_token_type_ids=self.types, logits_to_keep=1)
        c = cache[self.layer_index]
        parts = [x[:, :, self.start:self.end].transpose(0, 2, 1, 3).reshape(self.n, -1)
                 for x in (c.keys, c.values)]
        codes = mx.concatenate(parts, -1).astype(mx.float32)
        mx.eval(codes, output.logits)
        return np.array(codes), cache, output.logits
