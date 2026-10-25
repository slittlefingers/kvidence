#!/usr/bin/env python3
"""Small actual-model CPU timing check; no hidden-table replay for queries."""
import json
import random
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

import experiment as e

METHODS=('original','target_weighted_ridge','linear_rbf','target_design_16_4')


class TimedOracle:
    def __init__(self,underlying):
        self.underlying=underlying
        self.indices=[]
        self.batches=[]
        self.returned=[]
        self.started=time.perf_counter()

    def query(self,indices):
        values=indices.tolist()
        assert not set(values)&set(self.indices)
        started=time.perf_counter()
        measured=self.underlying.query(indices)
        ended=time.perf_counter()
        self.indices.extend(values)
        self.batches.append({'indices':values,'query_seconds':ended-started,
                             'end_to_end_seconds':ended-self.started})
        self.returned.append(measured.clone())
        return measured


def main():
    torch.set_num_threads(4)
    root=e.ROOT/'results'
    cases=json.loads((root/'trials.json').read_text())
    selected=[c for c in cases if c['split']=='confirmation' and c['seed']==e.SEEDS[0] and c['trial']<4]
    tokenizer=AutoTokenizer.from_pretrained(e.old.MODEL,local_files_only=True)
    lm=AutoModelForCausalLM.from_pretrained(e.old.MODEL,local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='eager').eval()
    lm.requires_grad_(False)
    base=lm.model
    full_layers=base.layers
    with np.load(e.PRIOR/'data'/str(e.SEEDS[0])/'features.npz') as raw:
        ids=torch.from_numpy(raw['ids'].copy())
        embedding=torch.from_numpy(raw['embedding'].copy())
        pca=torch.from_numpy(raw['pca'].copy())
        phi=torch.from_numpy(raw['phi'].copy())
    records=[]
    with torch.inference_mode():
        for case in selected:
            layer=case['layer']
            base.layers=full_layers[:layer]
            prefix_ids=tokenizer.encode(case['prefix'],return_tensors='pt',add_special_tokens=False)
            prepared=base(input_ids=prefix_ids,use_cache=True)
            cache=prepared.past_key_values.to_legacy_cache()
            with np.load(e.PRIOR/'data'/str(case['seed'])/f'prefix_{case["prefix_index"]}.npz') as raw:
                table=torch.from_numpy(raw[f'kv_{layer}'].copy())
            order=list(METHODS)
            random.Random(20261018+case['trial']*101+layer).shuffle(order)
            record={'key':case['key'],'layer':layer,'prefix_index':case['prefix_index'],
                    'target_index':case['target_index'],'method_order':order,'methods':{}}
            # Warm the cached-forward path before timed searches. This overhead,
            # model loading, and prefix-cache preparation are explicitly excluded.
            e.old.Oracle(base,cache,ids,layer).query(torch.tensor(case['initial_indices'][:20]))
            for method in order:
                wrapped=TimedOracle(e.old.Oracle(base,cache,ids,layer))
                trace=e.search(method,wrapped,embedding,pca,phi,torch.tensor(case['initial_indices']),
                               table[case['target_index']].clone(),case['search_seed'])
                elapsed=time.perf_counter()-wrapped.started
                # Audit observations only after search has returned.
                returned=torch.cat(wrapped.returned)
                reference=table[wrapped.indices]
                maximum_error=float((returned-reference).abs().max())
                assert torch.allclose(returned,reference,atol=3e-4,rtol=3e-4)
                first=next((batch for batch in wrapped.batches if case['target_index'] in batch['indices']),None)
                queries=0
                first_budget=None
                for batch in wrapped.batches:
                    queries+=len(batch['indices'])
                    if case['target_index'] in batch['indices'] and first_budget is None:first_budget=queries
                record['methods'][method]={'indices':trace['indices'],'batches':wrapped.batches,
                    'first_hit_batch':first_budget,
                    'first_hit_seconds':None if first is None else first['end_to_end_seconds'],
                    'search_seconds':trace['learner_seconds']['600'],
                    'query_seconds':sum(b['query_seconds'] for b in wrapped.batches),
                    'total_seconds_600':elapsed,'max_kv_difference_from_replay':maximum_error,
                    'same_indices_as_replay':trace['indices']==case['methods'][method]['trace']['indices']}
            records.append(record)
            e.save(root/'online.partial.json',records)
            print(case['key']+': '+', '.join(f'{m} hit={v["first_hit_batch"]} total={v["total_seconds_600"]:.3f}s'
                for m,v in record['methods'].items()),flush=True)
    base.layers=full_layers
    e.save(root/'online.json',{'config':{'model':config_model(),'device':'cpu','threads':4,
        'dtype':'float32','cases':len(records),'methods':METHODS,
        'selection':'first four confirmation trials in seed 20261019, both layers',
        'repeats_per_case_method':1,'method_order':'deterministically shuffled per case',
        'target_observation':'prior FP32 KV row','query_path':'actual model cached forward',
        'excluded_costs':['model loading','prefix cache preparation','warmup','feature loading'],
        'online_script_sha256':e.digest(e.ROOT/'online_check.py')},'cases':records})
    (root/'online.partial.json').unlink(missing_ok=True)


def config_model():
    return 'Qwen/Qwen2.5-0.5B'


if __name__=='__main__':main()
