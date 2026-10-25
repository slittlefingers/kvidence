#!/usr/bin/env python3
"""Algebra checks, independent raw-KV scoring, and frozen-baseline audit."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import experiment as e


def controls():
    torch.set_num_threads(4)
    torch.manual_seed(77103)
    embedding=torch.randn(700,48)
    observed=torch.randperm(700)[:200]
    measured=torch.tanh(embedding[observed]@torch.randn(48,16))
    target=torch.randn(16)
    prediction,state=e.ridge_predict(embedding,observed,measured)
    score=(prediction-target).square().mean(1)
    score[observed]=float('inf')
    ranking=score.argsort()
    reference=e.old.proxy_ranking(embedding,observed,measured,target)
    assert torch.equal(ranking,reference),'Baseline-equivalent predictor changed rankings'
    ones,_=e.ridge_predict(embedding,observed,measured,torch.ones(len(observed)))
    assert torch.allclose(prediction,ones,atol=2e-5,rtol=2e-5)
    permutation=torch.randperm(len(observed))
    nonlinear=e.rbf_predict(embedding,observed,measured)
    reordered=e.rbf_predict(embedding,observed[permutation],measured[permutation])
    assert torch.allclose(nonlinear,reordered,atol=2e-4,rtol=2e-4)
    selected=e.targeted_design(ranking[:500],state)
    assert torch.equal(selected[:16],ranking[:16])
    assert len(set(selected.tolist()))==20
    assert set(selected.tolist())<=set(ranking[:200].tolist())
    # Independently verify conditional covariance against augmented primal ridge.
    x=torch.randn(30,7,dtype=torch.float64)
    p=torch.randn(25,7,dtype=torch.float64)
    lam=.7
    prior=torch.linalg.inv(torch.eye(7,dtype=torch.float64)+x.T@x/lam)
    covariance=p@prior@p.T
    chosen=[2,8,12]
    for index in chosen:
        covariance=e.condition_covariance(covariance,index,lam)
    augmented=torch.cat((x,p[chosen]))
    direct=p@torch.linalg.inv(torch.eye(7,dtype=torch.float64)+augmented.T@augmented/lam)@p.T
    assert torch.allclose(covariance,direct,atol=1e-10,rtol=1e-10)
    return {'original_prediction_ranking_equal':True,'unit_weight_ridge_equal':True,
            'kernel_training_order_invariant':True,'design_batch_constraints':True,
            'conditional_covariance_matches_direct_solve':True}


def audit(folder):
    config=json.loads((folder/'config.json').read_text())
    assert e.digest(Path(config['original_source_path']))==config['original_source_sha256']
    assert e.digest(e.PRIOR/'experiment.py')==config['prior_experiment_sha256']
    assert e.digest(e.ROOT/'experiment.py')==config['source_sha256']
    cases=json.loads((folder/'trials.json').read_text())
    rejected=e.ROOT/'audit_rejected_run'
    previous={}
    if rejected.exists() and folder.name=='results':
        old_config=json.loads((rejected/'config.json').read_text())
        for field in ('parameters','methods','budgets','initial_queries','batch_size','original_source_sha256'):
            assert config[field]==old_config[field],field
        previous={c['key']:c for c in json.loads((rejected/'trials.json').read_text())}
    tables={}
    identities={}
    target_sets={'development':set(),'confirmation':set()}
    count,baseline_trajectories,unchanged_cases=0,0,0
    for case in cases:
        seed,pi,layer=case['seed'],case['prefix_index'],case['layer']
        if seed not in identities:
            with np.load(e.PRIOR/'data'/str(seed)/'features.npz') as raw:
                identities[seed]=raw['ids'].copy()
        target=case['target_index']
        if case['key'] in previous and previous[case['key']]['target_index']==target:
            for method,result in case['methods'].items():
                assert result['trace']['indices']==previous[case['key']]['methods'][method]['trace']['indices']
            unchanged_cases+=1
        target_sets[case['split']].add(int(identities[seed][target]))
        if (seed,pi,layer) not in tables:
            with np.load(e.PRIOR/'data'/str(seed)/f'prefix_{pi}.npz') as raw:
                tables[(seed,pi,layer)]=raw[f'kv_{layer}'].copy()
        table=tables[(seed,pi,layer)]
        t=table[target]
        nmse=np.mean((table-t)**2,axis=1)/max(np.mean(t**2),1e-12)
        linf=np.max(np.abs(table-t),axis=1)/max(np.max(np.abs(t)),1e-12)
        assert nmse.argmin()==target and linf.argmin()==target
        for method,result in case['methods'].items():
            trace=result['trace']
            indices=trace['indices']
            assert indices[:200]==case['initial_indices']
            assert len(indices)==len(set(indices))<=600
            assert indices==[i for batch in trace['batches'] for i in batch]
            assert all(1<=len(batch)<=20 for batch in trace['batches'][1:])
            if not trace['prior_stopped']:assert len(indices)==600
            total=0
            first=None
            for batch in trace['batches']:
                total+=len(batch)
                if first is None and target in batch:first=total
            assert result['first_hit_batch']==first
            assert result['capped_first_hit_queries']==(first or 600)
            for row in result['checkpoints']:
                b=row['budget']
                queried=np.array(indices[:b])
                assert row['queries']==len(queried)
                assert row['hit']==bool(target in queried)
                assert row['nearest_recovered']==bool(queried[nmse[queried].argmin()]==target)
                np.testing.assert_allclose(row['best_nmse'],nmse[queried].min(),rtol=3e-6,atol=1e-8)
                np.testing.assert_allclose(row['best_linf'],linf[queried].min(),rtol=1e-7,atol=1e-7)
                for eps in (.1,.25):
                    truth=linf<=eps
                    found=int(truth[queried].sum())
                    assert row[f'found_{eps:g}']==found
                    assert row[f'truth_count_{eps:g}']==int(truth.sum())
                    assert abs(row[f'recall_{eps:g}']-found/truth.sum())<1e-12
                count+=1
            if case['split']=='development' and method=='original':
                path=e.PRIOR/'results/traces'/f's{seed}_t{case["trial"]}_l{layer}_e0.1_kv_surrogate_full.json'
                if path.exists():
                    prior=json.loads(path.read_text())
                    assert indices==prior['indices'],(case['key'],'Baseline trajectory changed')
                    baseline_trajectories+=1
    assert not target_sets['development']&target_sets['confirmation'],'Target ID reused across split'
    return {'cases':len(cases),'method_budget_checkpoints':count,
            'original_source_unchanged':True,'prior_experiment_source_unchanged':True,
            'baseline_trajectories_identical_to_previous_experiment':baseline_trajectories,
            'development_confirmation_target_ids_disjoint':True,
            'audit_correction_unchanged_target_trajectories':unchanged_cases}


def audit_online(folder):
    obj=json.loads((folder/'online.json').read_text())
    assert e.digest(e.ROOT/'online_check.py')==obj['config']['online_script_sha256']
    replay={c['key']:c for c in json.loads((folder/'trials.json').read_text())}
    count=0
    for case in obj['cases']:
        original=replay[case['key']]
        assert original['split']=='confirmation' and original['seed']==20261019 and original['trial']<4
        for method,result in case['methods'].items():
            indices=result['indices']
            assert len(indices)==len(set(indices))==600
            assert indices[:200]==original['initial_indices']
            assert indices==[i for b in result['batches'] for i in b['indices']]
            total,first,first_time=0,None,None
            previous_time=0.
            for b in result['batches']:
                total+=len(b['indices'])
                assert b['query_seconds']>0 and b['end_to_end_seconds']>previous_time
                previous_time=b['end_to_end_seconds']
                if first is None and case['target_index'] in b['indices']:
                    first,first_time=total,b['end_to_end_seconds']
            assert result['first_hit_batch']==first and result['first_hit_seconds']==first_time
            assert result['total_seconds_600']>=previous_time
            assert result['search_seconds']+result['query_seconds']<=result['total_seconds_600']+1e-5
            assert result['max_kv_difference_from_replay']<3e-4
            assert result['same_indices_as_replay']==(indices==original['methods'][method]['trace']['indices'])
            count+=1
    return {'cases':len(obj['cases']),'actual_model_search_runs':count,'passed':True}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results')
    p.add_argument('--controls-only',action='store_true')
    args=p.parse_args()
    results={'controls':controls()}
    if not args.controls_only:results['audit']=audit(e.ROOT/args.output)
    if not args.controls_only and (e.ROOT/args.output/'online.json').exists():
        results['online_audit']=audit_online(e.ROOT/args.output)
    e.save(e.ROOT/args.output/'validation.json',results)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
