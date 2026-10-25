#!/usr/bin/env python3
"""Paired W8/native-KV versus W8/KV8 search on real Gemma 4 caches."""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from transformers import AutoTokenizer

from runtime import ROOT, MODEL, LAYERS, Runtime, Oracle, cache_description, tensor
from summary import summarize

# Reuse the exact query loop, controls, candidate filter and scoring from E4.
spec = importlib.util.spec_from_file_location('qwen_search', ROOT.parent/'kv_guided_search_large/search.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
common.LAYERS = LAYERS
BUDGETS, METHODS, PREFIXES = common.BUDGETS, common.METHODS, common.PREFIXES


def proxy_ranking(embedding, observed, kv, target, ridge=.01):
    """Same ridge problem; CPU solve and float32 GPU candidate scoring."""
    x = embedding[observed]
    mean, scale = x.mean(0), x.std(0).clamp_min(1e-4)
    x = (x-mean)/scale
    ym, ys = kv.mean(0), kv.std(0).clamp_min(1e-4)
    y = (kv-ym)/ys
    xc, yc = x-x.mean(0), y-y.mean(0)
    dual = torch.linalg.solve(xc@xc.T+torch.eye(len(x))*(ridge*len(x)), yc)
    def gpu(t): return mx.array(t.numpy())
    all_x = (gpu(embedding)-gpu(mean))/gpu(scale)-gpu(x.mean(0))
    # Associativity changes arithmetic order, not the ridge objective.
    predicted = ((all_x@gpu(xc.T.contiguous()))@gpu(dual)+gpu(y.mean(0)))*gpu(ys)+gpu(ym)
    distances = mx.mean(mx.square(predicted-gpu(target)), axis=-1)
    distances[mx.array(observed.numpy())] = float('inf')
    ranked = mx.argsort(distances)
    return torch.from_numpy(np.array(ranked, copy=True)).long()


reference_proxy_ranking = common.proxy_ranking
common.proxy_ranking = proxy_ranking


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', type=int, default=10000)
    p.add_argument('--trials-per-scenario', type=int, default=16)
    p.add_argument('--seeds', type=int, nargs='+', default=[20261017, 20261018])
    p.add_argument('--kv-modes', choices=['native', 'int8'], nargs='+', default=['native', 'int8'])
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--output', type=Path, default=ROOT/'results')
    args = p.parse_args()
    assert args.candidates >= max(BUDGETS)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    runtime = Runtime()
    metadata = runtime.metadata()
    (ROOT/'model_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with torch.inference_mode():
        for seed in args.seeds:
            ids, eligible_count = common.candidate_set(tokenizer,args.candidates,seed)
            embeddings = runtime.embeddings(ids.tolist())
            candidates = [{'index':i,'token_id':tid,
                'text':tokenizer.decode([tid],clean_up_tokenization_spaces=False)}
                for i,tid in enumerate(ids.tolist())]
            # Verify all candidate boundaries, not merely the selected targets.
            for prefix in PREFIXES:
                before = tokenizer.encode(prefix,add_special_tokens=False)
                assert all(tokenizer.encode(prefix+c['text'],add_special_tokens=False)
                           == before+[c['token_id']] for c in candidates)
            for mode in args.kv_modes:
                output = args.output/f'seed_{seed}_{mode}'
                output.mkdir(parents=True,exist_ok=True)
                (output/'candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
                all_trials = []
                started = time.perf_counter()
                storage = None
                for scenario in ('uniform_target','model_probability_target'):
                    for trial in range(args.trials_per_scenario):
                        prefix = PREFIXES[trial%len(PREFIXES)]
                        prefix_ids = [tokenizer.bos_token_id]+tokenizer.encode(prefix,add_special_tokens=False)
                        native_cache,native_logits = runtime.prepare(prefix_ids,'native')
                        reference_probability = tensor(mx.softmax(native_logits[mx.array(ids.tolist())])).flatten()
                        if mode=='native':
                            cache,probability = native_cache,reference_probability
                        else:
                            cache,logits = runtime.prepare(prefix_ids,mode)
                            probability = tensor(mx.softmax(logits[mx.array(ids.tolist())])).flatten()
                        if storage is None:
                            storage = cache_description(cache[LAYERS[-1]-1])
                        gen = torch.Generator().manual_seed(seed+trial+(10000 if scenario=='model_probability_target' else 0))
                        if scenario=='uniform_target':
                            target_index=torch.randint(len(ids),(),generator=gen).item()
                        else:
                            # Native reference ensures the SAME weighted targets in
                            # both KV modes; each baseline uses its own mode logits.
                            target_index=torch.multinomial(reference_probability,1,generator=gen).item()
                        initial=torch.randperm(len(ids),generator=torch.Generator().manual_seed(seed+100000+trial))[:200]
                        for layer in LAYERS:
                            target=Oracle(runtime,cache,ids,layer,args.batch_size).query([target_index])[0]
                            shared=Oracle(runtime,cache,ids,layer,args.batch_size)
                            begin=time.perf_counter();observed=shared.query(initial);initial_seconds=time.perf_counter()-begin
                            assert shared.count==200
                            entry={'scenario':scenario,'trial':trial,'layer':layer,'prefix':prefix,
                                'prefix_token_ids':prefix_ids,'kv_mode':mode,'seed':seed,
                                'target_index':target_index,'target_token_id':ids[target_index].item(),
                                'target_text':candidates[target_index]['text'],
                                'probability_rank':int((probability>probability[target_index]).sum())+1,
                                'initial_indices':initial.tolist(),'initial_forward_seconds':initial_seconds,
                                'methods':{}}
                            for method in METHODS:
                                oracle=Oracle(runtime,cache,ids,layer,args.batch_size)
                                entry['methods'][method]=common.run_search(method,oracle,embeddings,
                                    initial,observed,target,probability,seed+200000+trial)
                            all_trials.append(entry)
                            (output/'trials.partial.json').write_text(json.dumps(all_trials,indent=2)+'\n')
                        print(f'seed={seed} kv={mode} {scenario}: {trial+1}/{args.trials_per_scenario} targets; both producers',flush=True)
                report={**metadata,'seed':seed,'kv_mode':mode,'kv_storage':storage,
                    'candidate_count':len(ids),'eligible_english_digit_tokens':eligible_count,
                    'layers':list(LAYERS),'initial_queries':200,'budgets':BUDGETS,
                    'trials_per_scenario_per_layer':args.trials_per_scenario,
                    'known_correct_prefix':True,'prefix_bos':True,'chat_template_used':False,
                    'float32_complete_kv':False,'analysis_vectors_cast_to_float32':True,
                    'probability_target_reference':'same W8/native-KV distribution in both modes',
                    'probability_baseline':'current KV-mode full-depth conditional logits',
                    'target_indices_used_only_for_target_creation_and_scoring':True,
                    'surrogate':'main embedding(1536) -> dequantized real KV(1024), ridge; no offline inversion training',
                    'ridge':.01,'batch_queries':20,'oracle_batch_size':args.batch_size,
                    'shared_initial_queries':True,'no_early_stopping':True,
                    'int8_cache_window_limit':'all experiment contexts <512; nonrotating packed cache with local masks',
                    'total_seconds':time.perf_counter()-started,'summary':summarize(all_trials)}
                (output/'trials.json').write_text(json.dumps(all_trials,indent=2)+'\n')
                (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
                (output/'trials.partial.json').unlink(missing_ok=True)
                for layer,scenarios in report['summary'].items():
                    for scenario,methods in scenarios.items():
                        print(f'Layer {layer} {scenario}: '+', '.join(f'{m} @220={v["220"]["exact_recoveries"]}/{args.trials_per_scenario} @600={v["600"]["exact_recoveries"]}/{args.trials_per_scenario}' for m,v in methods.items()),flush=True)
                print(f'Saved {output}; {report["total_seconds"]:.2f} sec',flush=True)


if __name__=='__main__':main()
