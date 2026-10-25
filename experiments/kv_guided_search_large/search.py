#!/usr/bin/env python3
"""Does a 200-query KV surrogate guide search better than random/probability?"""
import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'model'
REVISION = '060db6499f32faf8b98477b0a26969ef7d8b9987'
LAYERS = (12, 24)
PREFIXES = [
    'Once upon a time, a little girl',
    'The boy opened the door and saw',
    'My favorite color is',
    'The number written on the wall was',
    'The password is',
    'Please remember this secret code:',
    'A small dog walked into the garden and',
    'The girl found a book about',
]
BUDGETS = [200, 220, 240, 300, 400, 600]
METHODS = ['random', 'language_probability', 'kv_surrogate', 'shuffled_embedding_control']


def candidate_set(tokenizer, size, seed):
    eligible = []
    for tid in range(tokenizer.vocab_size):
        text = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if (re.fullmatch(r' [A-Za-z0-9]{1,16}|[0-9]{1,3}', text)
                and tid not in tokenizer.all_special_ids
                and tokenizer.encode(text, add_special_tokens=False) == [tid]):
            eligible.append(tid)
    rng = random.Random(seed)
    # Include elementary digits/letters; the remaining classes come from English/digit BPE pieces.
    # Qwen separates a leading space from a digit. Include standalone digits,
    # rather than accidentally labelling the space token as a digit candidate.
    elementary = {tokenizer.encode(' ' + c, add_special_tokens=False)[0]
                  for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'}
    elementary.update(tokenizer.encode(c, add_special_tokens=False)[0] for c in '0123456789')
    assert elementary <= set(eligible)
    pool = sorted(elementary) + rng.sample(sorted(set(eligible) - elementary), size - len(elementary))
    assert len(pool) == len(set(pool)) == size
    return torch.tensor(pool, dtype=torch.long), len(eligible)


class Oracle:
    """Forward only explicitly requested candidates, at the same known prefix."""
    def __init__(self, base, prefix_cache, ids, layer, batch_size=128):
        self.base, self.prefix_cache, self.ids, self.layer = base, prefix_cache, ids, layer
        self.count = 0
        self.batch_size = batch_size

    def query(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long)
        parts = []
        self.count += len(indices)
        for chunk in indices.split(self.batch_size):
            n = len(chunk)
            expanded = tuple((k.expand(n, -1, -1, -1), v.expand(n, -1, -1, -1))
                             for k, v in self.prefix_cache[:self.layer])
            out = self.base(input_ids=self.ids[chunk, None],
                            past_key_values=DynamicCache.from_legacy_cache(expanded), use_cache=True)
            k, v = out.past_key_values[self.layer - 1]
            parts.append(torch.cat([k[:, :, -1, :].reshape(n, -1),
                                    v[:, :, -1, :].reshape(n, -1)], dim=-1).cpu())
        return torch.cat(parts)


def proxy_ranking(embedding, observed, kv, target, ridge=.01):
    """Fit embedding -> deep KV using queried pairs ONLY; rank unseen points."""
    x = embedding[observed]
    mean, scale = x.mean(0), x.std(0).clamp_min(1e-4)
    x = (x - mean) / scale
    ym, ys = kv.mean(0), kv.std(0).clamp_min(1e-4)
    y = (kv - ym) / ys
    # Equivalent ridge regression with an unpenalized intercept, solved in the
    # smaller sample space because n=200..580 is below d=896. No new training data.
    xc, yc = x - x.mean(0), y - y.mean(0)
    dual = torch.linalg.solve(xc @ xc.T + torch.eye(len(x)) * (ridge * len(x)), yc)
    coefficients = xc.T @ dual
    all_x = (embedding - mean) / scale
    predicted = ((all_x - x.mean(0)) @ coefficients + y.mean(0)) * ys + ym
    distances = (predicted - target).square().mean(-1)
    distances[observed] = float('inf')
    return distances.argsort()


def run_search(method, oracle, embeddings, initial_indices, initial_kv, target, probability, seed):
    rng = torch.Generator().manual_seed(seed)
    queried = initial_indices.clone()
    kv = initial_kv.clone()
    target_energy = target.square().mean().clamp_min(1e-12)
    distances = ((kv - target).square().mean(-1) / target_energy).tolist()
    if method == 'random':
        order = torch.randperm(len(embeddings), generator=rng)
    elif method == 'language_probability':
        order = probability.argsort(descending=True)
    if method == 'shuffled_embedding_control':
        # Fixed wrong token-to-embedding correspondence; no target/label information.
        embeddings = embeddings[torch.randperm(len(embeddings), generator=rng)]
    fit_seconds, query_seconds = 0., 0.
    for budget in BUDGETS[1:]:
        # Twenty-query batches allow fitting/update after newly measured KV.
        while len(queried) < budget:
            start = time.perf_counter()
            if method in ('kv_surrogate', 'shuffled_embedding_control'):
                ranking = proxy_ranking(embeddings, queried, kv, target)
            else:
                mask = torch.ones(len(embeddings), dtype=torch.bool)
                mask[queried] = False
                ranking = order[mask[order]]
            selected = ranking[:min(20, budget - len(queried))]
            assert not torch.isin(selected, queried).any()
            fit_seconds += time.perf_counter() - start
            start = time.perf_counter()
            measured = oracle.query(selected)
            query_seconds += time.perf_counter() - start
            distances.extend(((measured - target).square().mean(-1) / target_energy).tolist())
            queried = torch.cat([queried, selected])
            kv = torch.cat([kv, measured])
    assert len(set(queried.tolist())) == max(BUDGETS)
    assert oracle.count == max(BUDGETS) - BUDGETS[0]
    return {'indices': queried.tolist(), 'normalized_mse': distances,
            'adaptive_fit_and_rank_seconds': fit_seconds,
            'additional_forward_seconds': query_seconds}


def summarize(trials):
    summary = {}
    for layer in LAYERS:
        summary[str(layer)] = {}
        for scenario in ('uniform_target', 'model_probability_target'):
            cases = [r for r in trials if r['layer'] == layer and r['scenario'] == scenario]
            miss = [r for r in cases if r['target_index'] not in r['initial_indices']]
            by_method = {}
            for method in METHODS:
                by_budget = {}
                for budget in BUDGETS:
                    hits = sum(r['target_index'] in r['methods'][method]['indices'][:budget] for r in cases)
                    hit_miss = sum(r['target_index'] in r['methods'][method]['indices'][:budget] for r in miss)
                    def recovered(r):
                        trace = r['methods'][method]
                        best = min(range(budget), key=lambda i: trace['normalized_mse'][i])
                        return trace['indices'][best] == r['target_index']
                    recovered_count = sum(recovered(r) for r in cases)
                    recovered_miss = sum(recovered(r) for r in miss)
                    # Full float KV in this experiment should identify a queried target.
                    assert recovered_count == hits, 'Found target but distance selected a different token'
                    best = [min(r['methods'][method]['normalized_mse'][:budget]) for r in cases]
                    gains = []
                    for r in miss:
                        baseline = min(r['methods'][method]['normalized_mse'][:200])
                        now = min(r['methods'][method]['normalized_mse'][:budget])
                        gains.append((baseline-now)/max(baseline,1e-12))
                    by_budget[str(budget)] = {'trials': len(cases), 'hits': hits,
                        'hit_rate': hits / len(cases), 'exact_recoveries': recovered_count,
                        'exact_recovery_rate': recovered_count / len(cases),
                        'exact_recoveries_among_initial_misses': recovered_miss,
                        'initial_miss_trials': len(miss),
                        'hits_among_initial_misses': hit_miss,
                        'conditional_hit_rate': hit_miss / len(miss) if miss else None,
                        'mean_best_normalized_mse': sum(best)/len(best),
                        'mean_relative_distance_improvement_on_initial_misses': sum(gains)/len(gains) if gains else None}
                by_method[method] = by_budget
            summary[str(layer)][scenario] = by_method
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', type=int, default=10000)
    p.add_argument('--trials-per-scenario', type=int, default=16)
    p.add_argument('--seed', type=int, default=20261017)
    p.add_argument('--output', type=Path, default=ROOT/'results')
    p.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    p.add_argument('--batch-size', type=int, default=128)
    args = p.parse_args()
    assert args.candidates >= max(BUDGETS)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
            torch_dtype=torch.float32, attn_implementation='eager').eval().to(args.device)
    lm.requires_grad_(False)
    base = lm.model
    full_layers = base.layers
    ids, eligible_count = candidate_set(tokenizer, args.candidates, args.seed)
    model_ids = ids.to(args.device)
    embeddings = base.embed_tokens(model_ids).detach().cpu().clone()
    candidates = [{'index':i, 'token_id':tid, 'text':tokenizer.decode([tid],clean_up_tokenization_spaces=False)}
                  for i,tid in enumerate(ids.tolist())]
    (args.output/'candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
    all_trials = []
    started = time.perf_counter()
    with torch.inference_mode():
        for scenario in ('uniform_target','model_probability_target'):
            for trial in range(args.trials_per_scenario):
                prefix = PREFIXES[trial % len(PREFIXES)]
                base.layers = full_layers
                prefix_ids = tokenizer.encode(prefix, return_tensors='pt', add_special_tokens=False).to(args.device)
                prepared = base(input_ids=prefix_ids, use_cache=True)
                # Probability baseline uses original model's full-depth next-token logits.
                probability = lm.lm_head(prepared.last_hidden_state[:, -1, :])[0, model_ids].softmax(-1).cpu()
                gen = torch.Generator().manual_seed(args.seed + trial + (10000 if scenario=='model_probability_target' else 0))
                if scenario == 'uniform_target':
                    target_index = torch.randint(len(ids), (), generator=gen).item()
                else:
                    target_index = torch.multinomial(probability,1,generator=gen).item()
                assert tokenizer.encode(prefix+candidates[target_index]['text'],add_special_tokens=False) == prefix_ids[0].tolist()+[ids[target_index].item()]
                # Independent of target; all methods share these exact first 200 observations.
                initial = torch.randperm(len(ids), generator=torch.Generator().manual_seed(args.seed+100000+trial))[:200]
                for layer in LAYERS:
                    base.layers = torch.nn.ModuleList(list(full_layers)[:layer])
                    cache = prepared.past_key_values.to_legacy_cache()
                    target_oracle = Oracle(base,cache,model_ids,layer,args.batch_size)
                    target = target_oracle.query([target_index])[0]
                    shared_oracle = Oracle(base,cache,model_ids,layer,args.batch_size)
                    qstart=time.perf_counter();observed = shared_oracle.query(initial);initial_seconds=time.perf_counter()-qstart
                    assert shared_oracle.count==200
                    entry={'scenario':scenario,'trial':trial,'layer':layer,'prefix':prefix,
                           'target_index':target_index,'target_token_id':ids[target_index].item(),
                           'target_text':candidates[target_index]['text'],
                           'probability_rank':int((probability>probability[target_index]).sum())+1,
                           'initial_indices':initial.tolist(),'initial_forward_seconds':initial_seconds,'methods':{}}
                    for method in METHODS:
                        oracle = Oracle(base,cache,model_ids,layer,args.batch_size)
                        entry['methods'][method]=run_search(method,oracle,embeddings,initial,observed,
                                                           target,probability,args.seed+200000+trial)
                    all_trials.append(entry)
                    # Preserve completed cases if a longer experiment is interrupted.
                    (args.output/'trials.partial.json').write_text(json.dumps(all_trials,indent=2)+'\n')
                print(f'{scenario}: {trial+1}/{args.trials_per_scenario} targets complete (both layers)',flush=True)
    report={'model_id':'Qwen/Qwen2.5-0.5B','model_revision':REVISION,'seed':args.seed,
            'candidate_count':len(ids),'eligible_english_digit_tokens':eligible_count,
            'layers':list(LAYERS),'initial_queries':200,'budgets':BUDGETS,
            'trials_per_scenario_per_layer':args.trials_per_scenario,
            'known_correct_prefix':True,'float32_complete_kv':True,
            'target_indices_used_only_for_target_creation_and_scoring':True,
            'surrogate':'dual ridge embedding(896) -> observed deep K+V(256); no offline inversion training',
            'device':args.device,'dtype':'float32','oracle_batch_size':args.batch_size,
            'embedding_table_shape':list(base.embed_tokens.weight.shape),
            'model_parameters':sum(p.numel() for p in lm.parameters()),
            'tokenizer_vocab_size':tokenizer.vocab_size,
            'ridge':.01,'batch_queries':20,'shared_initial_queries':True,
            'no_early_stopping':True,'total_seconds':time.perf_counter()-started,
            'summary':summarize(all_trials)}
    (args.output/'trials.json').write_text(json.dumps(all_trials,indent=2)+'\n')
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'trials.partial.json').unlink(missing_ok=True)
    for layer,scenarios in report['summary'].items():
        for scenario,methods in scenarios.items():
            print(f'Layer {layer} {scenario}: '+', '.join(f'{m} @220={v["220"]["hits"]}/{args.trials_per_scenario} @600={v["600"]["hits"]}/{args.trials_per_scenario}' for m,v in methods.items()),flush=True)
    print(f'Saved results to {args.output}; {report["total_seconds"]:.2f} sec',flush=True)


if __name__=='__main__':
    main()
