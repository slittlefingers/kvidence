#!/usr/bin/env python3
"""Audit true cache types, cached/full agreement, shared KV and GPU ranking."""
import json

import mlx.core as mx
import torch
from transformers import AutoTokenizer
from mlx_lm.models.cache import QuantizedKVCache

from runtime import ROOT, MODEL, Runtime, Oracle, LAYERS, last_kv, tensor, cache_description
from search import proxy_ranking, reference_proxy_ranking

torch.set_num_threads(4)
torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
runtime = Runtime()
prefix = [tokenizer.bos_token_id] + tokenizer.encode('The password is',add_special_tokens=False)
ids = torch.tensor([tokenizer.encode(s,add_special_tokens=False)[0]
                    for s in [' apple',' window',' secret',' blue']])
result = {'model':runtime.metadata(), 'modes':{}}
with torch.inference_mode():
    for mode in ['native','int8']:
        cache, logits = runtime.prepare(prefix,mode)
        if mode=='int8':
            assert all(isinstance(c,QuantizedKVCache) and c.bits==8 for c in cache[:15])
            assert all(c.keys[0].dtype==mx.uint32 for c in cache[:15])
        original = [last_kv(c).clone() for c in cache[:15]]
        offsets = [c.offset for c in cache[:15]]
        references = {layer:[] for layer in LAYERS}
        for tid in ids.tolist():
            full, _ = runtime.prepare(prefix+[tid],mode)
            for layer in LAYERS:references[layer].append(last_kv(full[layer-1])[0])
        evidence = {'storage':cache_description(cache[14]), 'layers':{}}
        for layer in LAYERS:
            oracle = Oracle(runtime,cache,ids,layer)
            batched = oracle.query(range(len(ids)))
            individual = torch.stack([oracle.query([i])[0] for i in range(len(ids))])
            full = torch.stack(references[layer])
            energy = full.square().mean().clamp_min(1e-12)
            cached_error = ((batched-full).square().mean()/energy).item()
            batch_error = ((batched-individual).square().mean()/energy).item()
            # BF16 arithmetic can vary slightly with shape. Require a tight
            # relative agreement, and independently check exact identification.
            assert cached_error < .005 and batch_error < .005, (mode,layer,cached_error,batch_error)
            distances = (batched[:,None,:]-individual[None,:,:]).square().mean(-1)
            assert distances.argmin(-1).tolist()==list(range(len(ids)))
            evidence['layers'][str(layer)] = {'cached_vs_full_normalized_mse':cached_error,
                'batched_vs_individual_normalized_mse':batch_error,
                'correct_nearest_token_for_all_four_probes':True}
        assert [c.offset for c in cache[:15]]==offsets
        assert all(torch.equal(last_kv(c),old) for c,old in zip(cache[:15],original))
        evidence['prefix_cache_values_and_offsets_unchanged']=True
        result['modes'][mode]=evidence
    # Both methods solve the same ridge objective; only score arithmetic differs.
    e=torch.randn(1000,128);observed=torch.randperm(1000)[:200]
    kv=torch.randn(200,64);target=torch.randn(64)
    cpu=reference_proxy_ranking(e,observed,kv,target)
    gpu=proxy_ranking(e,observed,kv,target)
    assert cpu[:100].tolist()==gpu[:100].tolist()
    result['ridge_cpu_gpu_top_100_identical']=True
(ROOT/'oracle_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
