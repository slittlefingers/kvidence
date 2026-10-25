#!/usr/bin/env python3
"""Audit quantized traces, exact recovery and the paired KV-mode design."""
import argparse
import json
import math
from pathlib import Path

from runtime import ROOT
from summary import summarize

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--results',type=Path,default=ROOT/'results')
args=p.parse_args()
runs={}
for path in sorted(args.results.glob('seed_*')):
    m=json.loads((path/'metrics.json').read_text())
    candidates=json.loads((path/'candidates.json').read_text())
    trials=json.loads((path/'trials.json').read_text())
    assert m['weight_quantization']['bits']==8
    assert m['embedding_table_shape']==[262144,1536]
    assert m['layers']==[10,15] and m['layer_to_kv_producer_1based'][34]==15
    assert len(candidates)==len({c['token_id'] for c in candidates})==m['candidate_count']
    assert len(trials)==m['trials_per_scenario_per_layer']*4
    for r in trials:
        assert len(r['initial_indices'])==len(set(r['initial_indices']))==200
        original=None
        for trace in r['methods'].values():
            ids,distances=trace['indices'],trace['normalized_mse']
            assert len(ids)==len(set(ids))==len(distances)==600
            assert ids[:200]==r['initial_indices']
            assert all(0<=i<len(candidates) for i in ids)
            assert all(math.isfinite(x) and x>=0 for x in distances)
            if original is None:original=distances[:200]
            assert original==distances[:200]
    # Count final nearest-KV answers independently of target membership.
    audited=summarize(trials)
    for layer,scenarios in audited.items():
        for scenario,methods in scenarios.items():
            for method,budgets in methods.items():
                for budget,value in budgets.items():
                    saved=m['summary'][layer][scenario][method][budget]
                    assert saved['trials']==value['trials']
                    assert saved['hits']==value['hits']
                    assert saved['exact_recoveries']==value['exact_recoveries']
    if m['kv_mode']=='native':
        assert m['kv_storage']['dtype']=='mlx.core.bfloat16'
    else:
        assert m['kv_storage']['bits']==8 and m['kv_storage']['packed_dtype']=='mlx.core.uint32'
    runs[(m['seed'],m['kv_mode'])]=(candidates,trials)
    print(path.name,':',len(trials),'cases, counts and storage verified.')
for seed in sorted({k[0] for k in runs}):
    native,quantized=runs[(seed,'native')],runs[(seed,'int8')]
    assert native[0]==quantized[0]
    for a,b in zip(native[1],quantized[1]):
        for key in ['scenario','trial','layer','prefix_token_ids','target_index','target_token_id','initial_indices']:
            assert a[key]==b[key],(seed,key)
    print(seed,': native/KV8 targets, prefixes, pools and initial candidate indices paired exactly.')
assert len(runs)==4, 'Expected two seeds and two KV modes'
