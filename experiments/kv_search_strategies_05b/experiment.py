#!/usr/bin/env python3
"""Independent search variants. The original KVidence source is read-only."""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent/'kv_linfact_05b'
spec = importlib.util.spec_from_file_location('prior_benchmark',PRIOR/'experiment.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
old = prior.old
SEEDS = prior.SEEDS
LAYERS = (12,24)
BUDGETS = (200,220,240,300,400,600)
METHODS = ('original','random','linfact_vector_reference','ridge_linf',
           'target_weighted_ridge','linear_rbf','target_design_16_4')
SPLITS = ('development','confirmation')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,obj):
    prior.save_json(path,obj)


def prepare(embedding,observed,measured):
    x = embedding[observed]
    mean,scale = x.mean(0),x.std(0).clamp_min(1e-4)
    x = (x-mean)/scale
    ym,ys = measured.mean(0),measured.std(0).clamp_min(1e-4)
    y = (measured-ym)/ys
    return x,(embedding-mean)/scale,y,ym,ys


def ridge_predict(embedding,observed,measured,weights=None):
    x,all_x,y,ym,ys = prepare(embedding,observed,measured)
    lam = .01*len(x)
    if weights is None:
        xm,yc_mean = x.mean(0),y.mean(0)
        xc,yc = x-xm,y-yc_mean
        matrix = xc@xc.T+torch.eye(len(x))*lam
        coefficients = xc.T@torch.linalg.solve(matrix,yc)
    else:
        weights = weights/weights.mean()
        xm = (weights[:,None]*x).sum(0)/weights.sum()
        yc_mean = (weights[:,None]*y).sum(0)/weights.sum()
        xc,yc = x-xm,y-yc_mean
        root = weights.sqrt()[:,None]
        weighted_x,weighted_y = root*xc,root*yc
        matrix = weighted_x@weighted_x.T+torch.eye(len(x))*lam
        coefficients = weighted_x.T@torch.linalg.solve(matrix,weighted_y)
    centered_all = all_x-xm
    prediction = (centered_all@coefficients+yc_mean)*ys+ym
    return prediction,{'x':xc,'all_x':centered_all,'matrix':matrix,'lambda':lam}


def squared_distances(x,y):
    return (x.square().sum(1)[:,None]+y.square().sum(1)[None,:]-2*x@y.T).clamp_min(0)


def rbf_predict(embedding,observed,measured):
    x,all_x,y,ym,ys = prepare(embedding,observed,measured)
    xm = x.mean(0)
    x,all_x = x-xm,all_x-xm
    d2 = squared_distances(x,x)
    off_diagonal = ~torch.eye(len(x),dtype=torch.bool)
    bandwidth = d2[off_diagonal].median().clamp_min(1e-8)
    dimension = x.shape[1]
    kernel = x@x.T+dimension*torch.exp(-d2/bandwidth)
    kernel_all = all_x@x.T+dimension*torch.exp(-squared_distances(all_x,x)/bandwidth)
    row_mean,column_mean,overall = kernel.mean(1),kernel.mean(0),kernel.mean()
    centered = kernel-row_mean[:,None]-column_mean[None,:]+overall
    centered_all = kernel_all-kernel_all.mean(1)[:,None]-column_mean[None,:]+overall
    ymean = y.mean(0)
    coefficient = torch.linalg.solve(centered+torch.eye(len(x))*(.01*len(x)),y-ymean)
    return (centered_all@coefficient+ymean)*ys+ym


def condition_covariance(covariance,index,lam):
    column = covariance[:,index].clone()
    return covariance-torch.outer(column,column)/(lam+covariance[index,index].clamp_min(0))


def targeted_design(ranking,state,take=20):
    pool = ranking[:min(200,len(ranking))]
    # Use double precision for the covariance difference and sequential updates.
    x = state['x'].double()
    p = state['all_x'][pool].double()
    lam = state['lambda']
    cross = p@x.T
    covariance = p@p.T-cross@torch.linalg.solve(x@x.T+torch.eye(len(x),dtype=x.dtype)*lam,cross.T)
    covariance = (covariance+covariance.T)/2
    greedy = min(16,take)
    selected = list(range(greedy))
    for index in selected:
        covariance = condition_covariance(covariance,index,lam)
    while len(selected)<take:
        variance = covariance.diagonal().clone()
        variance[selected] = -float('inf')
        index = int(variance.argmax())
        selected.append(index)
        covariance = condition_covariance(covariance,index,lam)
    return pool[selected]


def select(method,embedding,observed,measured,target,take):
    if method == 'original':
        return old.proxy_ranking(embedding,observed,measured,target)[:take]
    if method == 'linear_rbf':
        prediction = rbf_predict(embedding,observed,measured)
    else:
        weights = None
        if method == 'target_weighted_ridge':
            distance = (measured-target).square().mean(1)
            q = torch.quantile(distance,.25).clamp_min(1e-12)
            weights = .1+torch.exp(-distance/q)
        prediction,state = ridge_predict(embedding,observed,measured,weights)
    score = ((prediction-target).abs().amax(1) if method == 'ridge_linf'
             else (prediction-target).square().mean(1))
    score[observed] = float('inf')
    ranking = score.argsort()[:len(embedding)-len(observed)]
    if method == 'target_design_16_4':
        return targeted_design(ranking,state,take)
    return ranking[:take]


def search(method,oracle,embedding,pca,phi,initial,target,seed):
    if method == 'linfact_vector_reference':
        snapshots,trace = prior.search('linfact_g_vector',oracle,embedding,pca,phi,
            initial,target,.1,seed,budgets=BUDGETS)
        return {'indices':trace['indices'],'batches':trace['batches'],
                'learner_seconds':{str(s['budget']):s['learner_seconds'] for s in snapshots},
                'lookup_seconds':{str(s['budget']):s['lookup_seconds'] for s in snapshots},
                'prior_stopped':snapshots[-1]['stopped']}
    observed = initial.clone()
    measured = oracle.query(observed)
    order = torch.randperm(len(embedding),generator=torch.Generator().manual_seed(seed))
    batches = [initial.tolist()]
    elapsed,lookup = 0.,0.
    times,lookup_times = {'200':0.},{'200':0.}
    for budget in BUDGETS[1:]:
        while len(observed)<budget:
            started = time.perf_counter()
            take = min(20,budget-len(observed))
            if method == 'random':
                unseen = torch.ones(len(embedding),dtype=torch.bool)
                unseen[observed] = False
                selected = order[unseen[order]][:take]
            else:
                selected = select(method,embedding,observed,measured,target,take)
            elapsed += time.perf_counter()-started
            started = time.perf_counter()
            extra = oracle.query(selected)
            lookup += time.perf_counter()-started
            observed = torch.cat((observed,selected))
            measured = torch.cat((measured,extra))
            batches.append(selected.tolist())
        times[str(budget)],lookup_times[str(budget)] = elapsed,lookup
    assert oracle.indices == observed.tolist()
    return {'indices':observed.tolist(),'batches':batches,
            'learner_seconds':times,'lookup_seconds':lookup_times,'prior_stopped':False}


def grade(trace,table,target_index):
    target = table[target_index]
    nmse = (table-target).square().mean(1)/target.square().mean().clamp_min(1e-12)
    linf = (table-target).abs().amax(1)/target.abs().max().clamp_min(1e-12)
    indices = trace['indices']
    rows=[]
    first=None
    cumulative=0
    for batch in trace['batches']:
        cumulative+=len(batch)
        if first is None and target_index in batch:
            first=cumulative
    for budget in BUDGETS:
        queried = torch.tensor(indices[:budget])
        row={'budget':budget,'queries':len(queried),'hit':target_index in queried.tolist(),
             'nearest_recovered':int(queried[nmse[queried].argmin()])==target_index,
             'best_nmse':float(nmse[queried].min()),'best_linf':float(linf[queried].min()),
             'learner_seconds':trace['learner_seconds'][str(budget)]}
        for epsilon in (.1,.25):
            truth = linf<=epsilon
            found = int(truth[queried].sum())
            row[f'recall_{epsilon:g}'] = found/int(truth.sum())
            row[f'found_{epsilon:g}'] = found
            row[f'truth_count_{epsilon:g}'] = int(truth.sum())
        rows.append(row)
    return {'checkpoints':rows,'first_hit_batch':first,
            'capped_first_hit_queries':first if first is not None else 600}


def cases_for_seed(seed,n,trials,token_ids,forbidden_token_ids):
    # Preserve prior trial/initial definitions in development. Confirmation
    # targets exclude all development token indices across the four prefixes.
    development=[]
    for trial in range(trials):
        target=int(torch.randint(n,(),generator=torch.Generator().manual_seed(seed+trial*101)))
        development.append(target)
    used=set(development)
    cases=[]
    for split in SPLITS:
        offset=0 if split=='development' else 900000
        for trial in range(trials):
            if split=='development':
                target=development[trial]
            else:
                generator=torch.Generator().manual_seed(seed+offset+trial*101)
                target=int(torch.randint(n,(),generator=generator))
                while target in used or int(token_ids[target]) in forbidden_token_ids:
                    target=int(torch.randint(n,(),generator=generator))
                used.add(target)
                forbidden_token_ids.add(int(token_ids[target]))
            initial=torch.randperm(n,generator=torch.Generator().manual_seed(seed+100000+trial+offset))[:200]
            cases.append((split,trial,trial%4,target,initial,seed+200000+trial+offset))
    return cases


def run(args):
    torch.set_num_threads(args.threads)
    out=ROOT/args.output
    out.mkdir(parents=True,exist_ok=True)
    baseline_path=Path(old.__file__)
    original_hash=digest(baseline_path)
    config={'seeds':args.seeds,'trials_per_seed_per_split':args.trials,'layers':args.layers,
            'splits':args.splits,'methods':args.methods,'budgets':BUDGETS,
            'initial_queries':200,'batch_size':20,'candidate_count':10000,
            'model':'Qwen/Qwen2.5-0.5B','revision':old.REVISION,
            'dtype':'float32','known_correct_prefix':True,'noise':None,
            'benchmark_type':'hidden_table_replay','primary_metric':'hit_at_220',
            'threads':args.threads,'source_sha256':digest(Path(__file__)),
            'original_source_path':str(baseline_path),'original_source_sha256':original_hash,
            'prior_experiment_sha256':digest(PRIOR/'experiment.py'),
            'confirmation_exclusion':'token IDs across all configured candidate pools',
            'parameters':{'ridge':.01,'target_weight_quantile':.25,'target_weight_floor':.1,
                          'rbf_amplitude':896,'rbf_bandwidth':'median_train_pair_squared_distance',
                          'target_design_pool':200,'target_design_greedy':16,'target_design_explore':4}}
    save(out/'config.json',config)
    records=[]
    # Candidate indices are local to a pool. Exclude the actual vocabulary IDs
    # from ALL development pools before generating any confirmation targets.
    seed_ids={}
    forbidden_token_ids=set()
    for seed in args.seeds:
        with np.load(PRIOR/'data'/str(seed)/'features.npz') as raw:
            seed_ids[seed]=raw['ids'].copy()
        for trial in range(args.trials):
            index=int(torch.randint(len(seed_ids[seed]),(),
                generator=torch.Generator().manual_seed(seed+trial*101)))
            forbidden_token_ids.add(int(seed_ids[seed][index]))
    for seed in args.seeds:
        data=PRIOR/'data'/str(seed)
        with np.load(data/'features.npz') as raw:
            embedding=torch.from_numpy(raw['embedding'].copy())
            pca=torch.from_numpy(raw['pca'].copy())
            phi=torch.from_numpy(raw['phi'].copy())
        cases=cases_for_seed(seed,len(embedding),args.trials,seed_ids[seed],forbidden_token_ids)
        for split,trial,pi,target_index,initial,search_seed in cases:
            if split not in args.splits:continue
            with np.load(data/f'prefix_{pi}.npz') as raw:
                tables={layer:torch.from_numpy(raw[f'kv_{layer}'].copy()) for layer in args.layers}
            for layer,table in tables.items():
                key=f'{split}_s{seed}_t{trial}_l{layer}'
                record={'key':key,'split':split,'seed':seed,'trial':trial,'prefix_index':pi,
                        'prefix':prior.PREFIXES[pi],'layer':layer,'target_index':target_index,
                        'initial_indices':initial.tolist(),'search_seed':search_seed,'methods':{}}
                for method in args.methods:
                    trace=search(method,prior.QueryOracle(table),embedding,pca,phi,
                                 initial,table[target_index].clone(),search_seed)
                    metrics=grade(trace,table,target_index)
                    record['methods'][method]={**metrics,'trace':trace}
                records.append(record)
                save(out/'trials.partial.json',records)
                print(key+': '+', '.join(f'{m} first={v["first_hit_batch"]}'
                      for m,v in record['methods'].items()),flush=True)
    assert digest(baseline_path)==original_hash,'Original source changed during run'
    save(out/'trials.json',records)
    (out/'trials.partial.json').unlink(missing_ok=True)
    save(out/'integrity.json',{'original_source_unchanged':True,'original_source_sha256':original_hash,
                             'experiment_source_sha256':digest(Path(__file__)),'cases':len(records)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run'])
    p.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS))
    p.add_argument('--trials',type=int,default=8)
    p.add_argument('--layers',type=int,nargs='+',default=list(LAYERS))
    p.add_argument('--splits',nargs='+',choices=SPLITS,default=list(SPLITS))
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--output',default='results')
    run(p.parse_args())


if __name__=='__main__':main()
