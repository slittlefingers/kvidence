#!/usr/bin/env python3
"""Audited, fixed-budget LinFACT-G adaptations on a hidden Qwen-0.5B KV table.

No claim of reproducing the paper's PAC bounds: calibration uses finite held-out
residuals, not a proven global misspecification bound. Search never receives the
target index or complete KV table; scoring takes place AFTER search returns.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'kv_guided_search_large'
spec = importlib.util.spec_from_file_location('qwen_baseline', OLD / 'search.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
SEEDS = (20261019, 20261020)
PREFIXES = (old.PREFIXES[0], old.PREFIXES[2], old.PREFIXES[4], old.PREFIXES[5])
LAYERS = (12, 24)
BUDGETS = (200, 220, 240, 300, 400, 600)
EPSILONS = (0.10, 0.25)
METHODS = ('random', 'kv_surrogate_full', 'kv_surrogate_pca',
           'linfact_g_scalar', 'linfact_g_vector')


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def features(embedding, seed, dimension=32):
    """Unsupervised feature-only PCA; no labels or KV in this function."""
    torch.manual_seed(seed)
    centered = embedding - embedding.mean(0)
    _, _, rotation = torch.pca_lowrank(centered, q=dimension + 8, center=False, niter=4)
    z = centered @ rotation[:, :dimension]
    z = z / z.std(0).clamp_min(1e-8)
    phi = torch.cat((torch.ones(len(z), 1), z), dim=1).double()
    return z, phi


def build(args):
    torch.set_num_threads(args.threads)
    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable in this execution environment; use cpu.')
    tokenizer = AutoTokenizer.from_pretrained(old.MODEL, local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(old.MODEL, local_files_only=True,
            torch_dtype=torch.float32, attn_implementation='eager').eval().to(device)
    lm.requires_grad_(False)
    base = lm.model
    with torch.inference_mode():
        for seed in args.seeds:
            out = ROOT / 'data' / str(seed)
            out.mkdir(parents=True, exist_ok=True)
            ids, eligible_count = old.candidate_set(tokenizer, args.candidates, seed)
            texts = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in ids.tolist()]
            for prefix in PREFIXES:
                before = tokenizer.encode(prefix, add_special_tokens=False)
                assert all(tokenizer.encode(prefix + t, add_special_tokens=False) == before + [i]
                           for i, t in zip(ids.tolist(), texts))
            embedding = base.embed_tokens(ids.to(device)).cpu().clone()
            z, phi = features(embedding, seed)
            np.savez(out / 'features.npz', ids=ids.numpy(), embedding=embedding.numpy(),
                     pca=z.numpy(), phi=phi.numpy())
            save_json(out / 'candidates.json', [{'index':i, 'token_id':t, 'text':s}
                      for i, (t, s) in enumerate(zip(ids.tolist(), texts))])
            provenance = {'model':'Qwen/Qwen2.5-0.5B', 'revision':old.REVISION,
                'seed':seed, 'candidate_count':len(ids), 'eligible_count':eligible_count,
                'device':device, 'dtype':'float32', 'layers':LAYERS, 'prefixes':PREFIXES,
                'pca_dimension':32, 'torch':torch.__version__, 'numpy':np.__version__,
                'platform':platform.platform(), 'batch_size':args.batch_size, 'prefix_runs':[]}
            for pi, prefix in enumerate(PREFIXES):
                path = out / f'prefix_{pi}.npz'
                if path.exists() and not args.force:
                    print(f'reuse {path}', flush=True)
                    continue
                input_ids = tokenizer.encode(prefix, return_tensors='pt', add_special_tokens=False).to(device)
                prepared = base(input_ids=input_ids, use_cache=True)
                prefix_cache = prepared.past_key_values.to_legacy_cache()
                tables = {layer:[] for layer in LAYERS}
                started = time.perf_counter()
                for start in range(0, len(ids), args.batch_size):
                    chunk = ids[start:start+args.batch_size].to(device)
                    n = len(chunk)
                    expanded = tuple((k.expand(n, -1, -1, -1), v.expand(n, -1, -1, -1))
                                     for k,v in prefix_cache)
                    result = base(input_ids=chunk[:,None],
                        past_key_values=DynamicCache.from_legacy_cache(expanded), use_cache=True)
                    for layer in LAYERS:
                        k, v = result.past_key_values[layer-1]
                        tables[layer].append(torch.cat((k[:,:,-1,:].reshape(n,-1),
                                                       v[:,:,-1,:].reshape(n,-1)), dim=1).cpu())
                tables = {layer:torch.cat(parts) for layer,parts in tables.items()}
                elapsed = time.perf_counter() - started
                # Independent full-prefix forward checks for both observed layers.
                max_errors = {}
                for ci in (0, len(ids)//2, len(ids)-1):
                    full = base(input_ids=torch.cat((input_ids, ids[ci:ci+1].to(device)[None]), 1), use_cache=True)
                    for layer in LAYERS:
                        k,v = full.past_key_values[layer-1]
                        direct = torch.cat((k[:,:,-1,:].flatten(),v[:,:,-1,:].flatten())).cpu()
                        err = (direct-tables[layer][ci]).abs().max().item()
                        max_errors[str(layer)] = max(err, max_errors.get(str(layer), 0.))
                        assert torch.allclose(direct, tables[layer][ci], rtol=3e-4, atol=3e-4), (layer,ci,err)
                assert all(k.shape[2] == input_ids.shape[1] for k,v in prefix_cache)
                np.savez(path, **{f'kv_{layer}':table.numpy() for layer,table in tables.items()})
                item = {'prefix_index':pi, 'prefix_tokens':input_ids[0].tolist(),
                        'offline_candidate_forwards':len(ids), 'offline_seconds':elapsed,
                        'oracle_validation_max_abs_error':max_errors}
                provenance['prefix_runs'].append(item)
                save_json(out / f'prefix_{pi}_metadata.json', item)
                print(f'seed={seed} prefix={pi} all {len(ids)} KV rows: {elapsed:.2f}s; validation={max_errors}', flush=True)
            save_json(out / 'metadata.json', provenance)


class QueryOracle:
    """Hidden offline data, exposed solely by query(), with no duplicate pulls."""
    def __init__(self, table):
        self.__table = table
        self.indices = []

    def query(self, indices):
        idx = torch.as_tensor(indices, dtype=torch.long)
        assert len(set(idx.tolist())) == len(idx)
        assert not set(idx.tolist()).intersection(self.indices)
        self.indices.extend(idx.tolist())
        return self.__table[idx].clone()


def g_design(phi, limit=100, tolerance=.05):
    """Approximate G design, with max-leverage/rank optimality certificate.

    Vertex-direction D-optimal iterations. Equivalence theorem gives G-optimality
    when the maximum design leverage approaches rank. No model observations used.
    """
    cov = phi.T @ phi / len(phi)
    eigen, basis = torch.linalg.eigh(cov)
    keep = eigen > eigen.max() * 1e-10
    x = phi @ basis[:,keep]
    d = x.shape[1]
    inv = torch.diag(eigen[keep].reciprocal())
    leverage = (x @ inv * x).sum(1)
    weights = torch.full((len(x),), 1/len(x), dtype=torch.float64)
    iterations = 0
    for iterations in range(limit):
        maximum, selected = leverage.max(0)
        if maximum <= d*(1+tolerance):
            break
        step = float((maximum-d)/(d*(maximum-1))) if d > 1 else .5
        step = min(.99, max(0., step))
        u = inv @ x[selected]
        cross = x @ u
        denominator = 1-step+step*maximum
        leverage = (leverage-step*cross.square()/denominator)/(1-step)
        inv = (inv-step*torch.outer(u,u)/denominator)/(1-step)
        weights *= 1-step
        weights[selected] += step
    # Recompute the certificate rather than trusting accumulated rank-one updates.
    final_cov = x.T @ (weights[:,None]*x)
    certificate = float((x @ torch.linalg.inv(final_cov) * x).sum(1).max()/d)
    return weights, {'rank':d, 'max_leverage_over_rank':certificate, 'iterations':iterations+1}


def fit_intervals(phi, observed, measured, target, mode, calibration_count=50, inflation=1.5):
    """Empirical prediction intervals, explicitly NOT certified simultaneous CIs."""
    calibration = observed[:calibration_count]
    train = observed[calibration_count:]
    x, cx = phi[train], phi[calibration]
    normalized = measured.double() / target.abs().max().clamp_min(1e-12)
    ty = target.double() / target.abs().max().clamp_min(1e-12)
    y = (normalized-ty).abs().amax(1,keepdim=True) if mode == 'scalar' else normalized
    # Full-rank OLS; adaptive design on features and initial random samples.
    inverse = torch.linalg.inv(x.T @ x)
    coefficient = inverse @ x.T @ y[calibration_count:]
    prediction = phi @ coefficient
    leverage = (phi @ inverse * phi).sum(1).clamp_min(0)
    error = (y[:calibration_count] - cx @ coefficient).abs().amax(1)
    standardized = error / (1+leverage[calibration]).sqrt()
    floor = float(standardized.max()) * inflation
    radius = floor*(1+leverage).sqrt()
    estimate = prediction[:,0] if mode == 'scalar' else (prediction-ty).abs().amax(1)
    lower, upper = (estimate-radius).clamp_min(0), (estimate+radius).clamp_min(0)
    # Queried distances are deterministic and directly observed, not predictions.
    exact = (normalized-ty).abs().amax(1)
    lower[observed] = exact
    upper[observed] = exact
    return lower, upper, {'empirical_radius_floor':floor,
        'calibration_max_residual':float(error.max()), 'calibration_count':calibration_count,
        'training_count':len(train), 'gram_condition':float(torch.linalg.cond(x.T@x))}


def classify(lower, upper, good, bad, epsilon):
    # Keep confirmed good candidates that could still determine the minimum.
    remaining = ~bad
    best_upper = upper[remaining].min()
    active = remaining & (~good | (lower <= best_upper))
    best_lower, best_upper = lower[active].min(), upper[active].min()
    undecided = ~(good|bad)
    good = good | (undecided & (upper <= best_lower+epsilon))
    bad = bad | (undecided & (lower > best_upper+epsilon))
    assert not (good&bad).any()
    active = ~bad & (~good | (lower <= best_upper))
    return good, bad, active


def search(method, oracle, embedding, pca, phi, initial, target, epsilon, seed, budgets=BUDGETS):
    """No target index or hidden true distances supplied to the learner."""
    n = len(embedding)
    observed = initial.clone()
    measured = oracle.query(observed)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator)
    good = torch.zeros(n,dtype=torch.bool)
    bad = good.clone()
    snapshots = []
    learner_seconds, lookup_seconds = 0., 0.
    fitted_at = -1
    designs = []
    stopped = False
    selected_batches = [initial.tolist()]
    lower = upper = None
    fit_stats = {}
    start_total = time.perf_counter()
    for budget in budgets:
        while True:
            start = time.perf_counter()
            if method.startswith('linfact') and fitted_at != len(observed):
                lower,upper,fit_stats = fit_intervals(phi,observed,measured,target,
                            'scalar' if 'scalar' in method else 'vector')
                good,bad,active = classify(lower,upper,good,bad,epsilon)
                fitted_at = len(observed)
                stopped = bool((good|bad).all())
            learner_seconds += time.perf_counter()-start
            if len(observed) >= budget or stopped:
                break
            start = time.perf_counter()
            unseen = torch.ones(n,dtype=torch.bool)
            unseen[observed] = False
            take = min(20,budget-len(observed))
            if method.startswith('linfact'):
                pool = (unseen&active).nonzero().flatten()
                if not len(pool):
                    raise RuntimeError('Unresolved classification but no unqueried active candidates')
                weights,design = g_design(phi[pool])
                chosen = torch.multinomial(weights,min(take,len(pool)),replacement=False,generator=generator)
                selected = pool[chosen]
                designs.append({'after_queries':len(observed),'active_unqueried':len(pool),**design})
            elif method == 'random':
                selected = order[unseen[order]][:take]
            else:
                use_embedding = embedding if method.endswith('full') else pca
                selected = old.proxy_ranking(use_embedding,observed,measured,target)[:take]
            learner_seconds += time.perf_counter()-start
            start = time.perf_counter()
            extra = oracle.query(selected)
            lookup_seconds += time.perf_counter()-start
            selected_batches.append(selected.tolist())
            observed = torch.cat((observed,selected))
            measured = torch.cat((measured,extra))
        exact = (measured-target).abs().amax(1)/target.abs().max().clamp_min(1e-12)
        nearest = int(observed[exact.argmin()])
        if not method.startswith('linfact'):
            good = torch.zeros(n,dtype=torch.bool)
            bad = good.clone()
            good[observed[exact <= exact.min()+epsilon]] = True
        snapshots.append({'budget':budget,'queries':len(observed),'nearest':nearest,
            'queried':observed.clone(),'good':good.clone(),'bad':bad.clone(),
            'lower':None if lower is None else lower.clone(),
            'upper':None if upper is None else upper.clone(),
            'stopped':stopped,'learner_seconds':learner_seconds,
            'lookup_seconds':lookup_seconds,'fit_stats':dict(fit_stats)})
    assert oracle.indices == observed.tolist()
    return snapshots, {'indices':observed.tolist(),'batches':selected_batches,'designs':designs,
                       'elapsed_seconds':time.perf_counter()-start_total}


def grade(snapshots, truth_kv, target, target_index, epsilon):
    """Only evaluator sees all true distances, after the learner has finished."""
    true_distance = (truth_kv-target).abs().amax(1)/target.abs().max().clamp_min(1e-12)
    truth = true_distance <= true_distance.min()+epsilon
    graded = []
    for snap in snapshots:
        good,bad = snap['good'],snap['bad']
        tp = int((good&truth).sum())
        item = {k:v for k,v in snap.items() if k not in ('queried','good','bad','lower','upper')}
        item.update({'target_queried':bool((snap['queried']==target_index).any()),
            'nearest_recovered':snap['nearest']==target_index,'truth_count':int(truth.sum()),
            'confirmed_count':int(good.sum()),'true_positive_count':tp,
            'precision':tp/int(good.sum()) if good.any() else None,
            'recall':tp/int(truth.sum()),'exact_set':bool(torch.equal(good,truth)),
            'false_eliminations':int((bad&truth).sum()),'target_eliminated':bool(bad[target_index]),
            'possible_count':int((~bad).sum()),
            'possible_recall':float((truth&~bad).sum()/truth.sum()),
            'false_inclusions':int((good&~truth).sum())})
        if snap['lower'] is not None:
            valid = (true_distance.double() >= snap['lower']-1e-7)&(true_distance.double() <= snap['upper']+1e-7)
            unseen = torch.ones(len(truth),dtype=torch.bool)
            unseen[snap['queried']] = False
            item.update({'interval_coverage':float(valid.double().mean()),
                'all_intervals_cover':bool(valid.all()),
                'unqueried_interval_coverage':float(valid[unseen].double().mean()) if unseen.any() else 1.,
                'target_interval_covers':bool(valid[target_index])})
        graded.append(item)
    return graded, true_distance, truth


def run(args):
    torch.set_num_threads(args.threads)
    out = ROOT / args.output
    trace_dir = out/'traces'
    trace_dir.mkdir(parents=True,exist_ok=True)
    all_cases = []
    for seed in args.seeds:
        data = ROOT/'data'/str(seed)
        with np.load(data/'features.npz') as feature_file:
            embedding = torch.from_numpy(feature_file['embedding'].copy())
            pca = torch.from_numpy(feature_file['pca'].copy())
            phi = torch.from_numpy(feature_file['phi'].copy())
        if args.candidates_limit:
            embedding,pca,phi = (v[:args.candidates_limit] for v in (embedding,pca,phi))
        for trial in range(args.trials):
            pi = trial%len(PREFIXES)
            rng = torch.Generator().manual_seed(seed+trial*101)
            target_index = int(torch.randint(len(embedding),(),generator=rng))
            initial = torch.randperm(len(embedding),generator=torch.Generator().manual_seed(seed+100000+trial))[:200]
            with np.load(data/f'prefix_{pi}.npz') as table_file:
                tables = {layer:torch.from_numpy(table_file[f'kv_{layer}'].copy())[:len(embedding)] for layer in args.layers}
            for layer,table in tables.items():
                target = table[target_index].clone()
                for epsilon in args.epsilons:
                    key = f's{seed}_t{trial}_l{layer}_e{epsilon:g}'
                    record = {'key':key,'seed':seed,'trial':trial,'prefix_index':pi,'prefix':PREFIXES[pi],
                        'layer':layer,'epsilon':epsilon,'target_index':target_index,
                        'candidate_count':len(embedding),'initial_indices':initial.tolist(),'methods':{}}
                    for method in args.methods:
                        # Each method owns an independent oracle/query ledger.
                        snaps,trace = search(method,QueryOracle(table),embedding,pca,phi,initial,
                                             target,epsilon,seed+200000+trial)
                        metrics,distance,truth = grade(snaps,table,target,target_index,epsilon)
                        record['methods'][method] = metrics
                        arrays = {'truth':truth.numpy(),'distance':distance.numpy()}
                        for snap in snaps:
                            b = str(snap['budget'])
                            for field in ('queried','good','bad','lower','upper'):
                                if snap[field] is not None:
                                    arrays[f'{field}_{b}'] = snap[field].numpy()
                        np.savez_compressed(trace_dir/f'{key}_{method}.npz',**arrays)
                        save_json(trace_dir/f'{key}_{method}.json',trace)
                    all_cases.append(record)
                    save_json(out/'trials.partial.json',all_cases)
                    summary = ', '.join(f'{m}: hit={v[-1]["nearest_recovered"]}, '
                        f'recall={v[-1]["recall"]:.2f}, false_drop={v[-1]["false_eliminations"]}, '
                        f'n={v[-1]["queries"]}' for m,v in record['methods'].items())
                    print(f'{key}: {summary}',flush=True)
    save_json(out/'trials.json',all_cases)
    (out/'trials.partial.json').unlink(missing_ok=True)
    save_json(out/'config.json',{'seeds':args.seeds,'trials_per_seed':args.trials,'layers':args.layers,
        'epsilons':args.epsilons,'methods':args.methods,'budgets':BUDGETS,'initial_queries':200,
        'known_correct_prefix':True,'target_distribution':'uniform', 'candidate_count':len(embedding),
        'distance':'Linf(KV-target)/maxabs(target)','observation_noise':None,
        'dtype':'float32','weights_quantized':False,'kv_quantized':False,
        'calibration_count':50,'calibration_inflation':1.5,'pca_dimension':32,
        'certified_confidence_bounds':False,'paper_reproduction':False,'benchmark_type':'hidden_table_replay',
        'threads':args.threads,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('build')
    p.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS))
    p.add_argument('--candidates',type=int,default=10000)
    p.add_argument('--device',choices=['cpu','mps'],default='cpu')
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--force',action='store_true')
    p = sub.add_parser('run')
    p.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS))
    p.add_argument('--trials',type=int,default=8)
    p.add_argument('--layers',type=int,nargs='+',default=list(LAYERS))
    p.add_argument('--epsilons',type=float,nargs='+',default=list(EPSILONS))
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--candidates-limit',type=int,default=0)
    p.add_argument('--output',default='results')
    args = parser.parse_args()
    {'build':build,'run':run}[args.command](args)


if __name__=='__main__':
    main()
