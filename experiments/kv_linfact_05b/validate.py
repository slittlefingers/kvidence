#!/usr/bin/env python3
"""Independent trajectory audit plus exact-linear and interval-logic controls."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import experiment as e


def controls():
    torch.set_num_threads(4)
    torch.manual_seed(771)
    # Verify interval classification on synthetic finite candidate sets. This
    # checks both false exclusion and false inclusion with an unknown minimum.
    for _ in range(50):
        d = torch.rand(150,dtype=torch.float64)*4
        r = torch.rand(150,dtype=torch.float64)*2
        lower,upper = (d-r).clamp_min(0),d+r
        for epsilon in (.1,.7):
            good,bad,active = e.classify(lower,upper,torch.zeros(150,dtype=torch.bool),
                                        torch.zeros(150,dtype=torch.bool),epsilon)
            truth = d <= d.min()+epsilon
            assert not (bad&truth).any()
            assert not (good&~truth).any()
            assert active[d.argmin()]
    # An exactly linear, deterministic model must be solved by both adapters
    # without granting them the target index or unqueried labels.
    n = 700
    z = torch.randn(n,4)
    phi = torch.cat((torch.ones(n,1),z),1).double()
    initial = torch.randperm(n)[:200]
    results = {}
    for method in ('linfact_g_scalar','linfact_g_vector'):
        if 'scalar' in method:
            table = (100+2*z[:,0]+3*z[:,1])[:,None]
            target_index = int(table[:,0].argmin())
            epsilon = .025
        else:
            table = (phi @ torch.randn(5,8,dtype=torch.float64)).float()
            target_index = 619
            epsilon = .15
        target = table[target_index].clone()
        oracle = e.QueryOracle(table)
        snapshots,trace = e.search(method,oracle,z,z,phi,initial,target,epsilon,73,budgets=(200,220))
        scored,_,_ = e.grade(snapshots,table,target,target_index,epsilon)
        assert scored[-1]['stopped'] and scored[-1]['exact_set'], (method,scored[-1])
        assert scored[-1]['queries'] == 200
        results[method] = {'queries':200,'exact_set':True}
    # Verify approximate optimal-design weights and the reported certificate.
    weights,design = e.g_design(phi,limit=1000,tolerance=.03)
    assert abs(float(weights.sum())-1)<1e-9 and bool((weights>=0).all())
    matrix = phi.T @ (weights[:,None]*phi)
    direct_certificate = float((phi@torch.linalg.inv(matrix)*phi).sum(1).max()/phi.shape[1])
    assert abs(direct_certificate-design['max_leverage_over_rank'])<1e-6
    assert direct_certificate<1.04
    return {'interval_classification_cases':100,'linear_controls':results,'g_design':design}


def audit(folder):
    cases = json.loads((folder/'trials.json').read_text())
    count = 0
    tables = {}
    for case in cases:
        target = case['target_index']
        initial = case['initial_indices']
        table_key = (case['seed'],case['prefix_index'],case['layer'])
        if table_key not in tables:
            path = e.ROOT/'data'/str(case['seed'])/f'prefix_{case["prefix_index"]}.npz'
            with np.load(path) as data:
                tables[table_key] = data[f'kv_{case["layer"]}'].copy()
        table = tables[table_key][:case['candidate_count']]
        independently_computed = np.max(np.abs(table-table[target]),axis=1)/max(np.max(np.abs(table[target])),1e-12)
        for method,metrics in case['methods'].items():
            stem = folder/'traces'/f'{case["key"]}_{method}'
            # Keys include a decimal epsilon, so don't treat that as a suffix.
            trace = json.loads(Path(str(stem)+'.json').read_text())
            with np.load(str(stem)+'.npz') as raw:
                truth = raw['truth'].astype(bool)
                distance = raw['distance']
                np.testing.assert_allclose(distance,independently_computed,rtol=1e-7,atol=1e-7)
                assert np.array_equal(truth,distance<=distance.min()+case['epsilon'])
                assert int(distance.argmin())==target
                assert trace['indices'][:200]==initial
                assert len(trace['indices'])==len(set(trace['indices']))<=600
                assert [i for batch in trace['batches'] for i in batch]==trace['indices']
                previous_good = np.zeros(len(truth),dtype=bool)
                previous_bad = previous_good.copy()
                for row in metrics:
                    b = str(row['budget'])
                    queried,good,bad = raw['queried_'+b],raw['good_'+b],raw['bad_'+b]
                    assert queried.tolist()==trace['indices'][:len(queried)]
                    assert len(queried)==row['queries']<=row['budget']
                    assert not np.any(good&bad)
                    if method.startswith('linfact'):
                        assert np.all(good[previous_good]) and np.all(bad[previous_bad])
                        assert row['stopped']==bool(np.all(good|bad))
                    else:
                        assert np.all(np.isin(np.where(good)[0],queried))
                    expected = {'target_queried':target in queried,
                        'nearest_recovered':int(queried[distance[queried].argmin()])==target,
                        'truth_count':int(truth.sum()),'confirmed_count':int(good.sum()),
                        'false_eliminations':int(np.sum(bad&truth)),
                        'target_eliminated':bool(bad[target]),
                        'false_inclusions':int(np.sum(good&~truth)),
                        'possible_count':int(np.sum(~bad)),
                        'exact_set':bool(np.array_equal(good,truth))}
                    for name,value in expected.items():
                        assert row[name]==value,(case['key'],method,b,name,row[name],value)
                    assert abs(row['recall']-np.sum(good&truth)/truth.sum())<1e-7
                    if method.startswith('linfact'):
                        valid = (distance>=raw['lower_'+b]-1e-7)&(distance<=raw['upper_'+b]+1e-7)
                        assert abs(row['interval_coverage']-valid.mean())<1e-10
                        assert row['all_intervals_cover']==bool(valid.all())
                    previous_good,previous_bad = good,bad
                    count+=1
    return {'cases':len(cases),'audited_method_checkpoints':count}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results')
    p.add_argument('--controls-only',action='store_true')
    args=p.parse_args()
    report={'controls':controls()}
    if not args.controls_only:
        report['trajectory_audit']=audit(e.ROOT/args.output)
    path=e.ROOT/args.output/'validation.json'
    e.save_json(path,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
