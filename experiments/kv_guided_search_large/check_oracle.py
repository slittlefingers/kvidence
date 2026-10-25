#!/usr/bin/env python3
"""Check cached, truncated-layer KV queries against full teacher-forced forwards."""
import json
import torch
from search import ROOT, MODEL, LAYERS, Oracle, AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)
torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
lm = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
    torch_dtype=torch.float32, attn_implementation='eager').eval().to('mps')
lm.requires_grad_(False)
base, full_layers = lm.model, lm.model.layers
prefix = tokenizer.encode('The password is', return_tensors='pt', add_special_tokens=False).to('mps')
ids = torch.tensor([tokenizer.encode(s, add_special_tokens=False)[0]
                    for s in [' apple', ' window', ' secret', ' blue']], device='mps')
evidence = {}
with torch.inference_mode():
    prepared = base(input_ids=prefix, use_cache=True)
    cache = prepared.past_key_values.to_legacy_cache()
    references = {layer: [] for layer in LAYERS}
    for token in ids:
        full = base(input_ids=torch.cat([prefix, token.reshape(1, 1)], dim=1), use_cache=True)
        for layer in LAYERS:
            k, v = full.past_key_values[layer - 1]
            references[layer].append(torch.cat([k[:, :, -1, :].flatten(),
                                                v[:, :, -1, :].flatten()]).cpu())
    for layer in LAYERS:
        base.layers = torch.nn.ModuleList(list(full_layers)[:layer])
        oracle = Oracle(base, cache, ids, layer)
        batched = oracle.query(range(len(ids)))
        individual = torch.stack([oracle.query([i])[0] for i in range(len(ids))])
        reference = torch.stack(references[layer])
        energy = reference.square().mean().clamp_min(1e-12)
        mse_full = ((batched - reference).square().mean() / energy).item()
        mse_individual = ((batched - individual).square().mean() / energy).item()
        assert mse_full < 1e-8 and mse_individual < 1e-8
        assert all(k.shape[-2] == prefix.shape[-1] for k, v in cache)
        evidence[str(layer)] = {'cached_vs_full_normalized_mse': mse_full,
                                'batched_vs_individual_normalized_mse': mse_individual,
                                'prefix_cache_unchanged': True}
(ROOT / 'oracle_validation.json').write_text(json.dumps(evidence, indent=2) + '\n')
print(json.dumps(evidence, indent=2))
