"""Independent adaptive experiment; no changes to any original recovery method."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
LONG = ROOT.parent/'ciphertext_hybrid_l2_long'
sys.path.insert(0, str(LONG))
import run_long as original
from adaptive import FastSurrogate, search, trust_update

shared, exp = original.shared, original.exp
OUT = ROOT/'results'
METHODS = ('hybrid_50_150', 'staged_fast', 'adaptive_trust')


def prepare():
    if OUT.exists():
        raise FileExistsError('Preserve existing results before preparing another experiment')
    shutil.copytree(LONG/'results/observed', OUT/'observed')
    shutil.copy2(LONG/'results/cache_validation.json', OUT/'cache_validation.json')
    protocol = json.loads((LONG/'results/protocol.json').read_text())
    protocol.update(methods=list(METHODS), original_method_unmodified=True,
        source_observation_sha256=exp.digest(OUT/'observed/target.npz'),
        adaptive_design=dict(initial='top50 check, then random150 if unresolved',
            ridge=.01, proposal_batch=20, max_training_anchors=512,
            candidate_query_limit=None, trust_initial_state=1., trust_state_range=[.25,4.],
            base_radius='median standardized embedding RMS distance to current best at first fit',
            residual_factor='clip(sqrt(current_best_L2 / first_fit_best_L2), .25, 1)',
            proposals='18 smallest predicted KV residuals inside radius + 2 best global remaining',
            floor='radius must contain at least 18 unqueried candidates',
            rho='actual squared-L2 decrease / predicted squared-L2 decrease',
            shrink='rho<.25 or nonpositive predicted reduction: state*=.5',
            expand='rho>.75, actual improvement, boundary step: state*=1.6',
            acceptance='real raw KV L2 < 4, no predicted-score acceptance'),
        timing='same JSONL logging; loop excludes model loading/prefix; GPU surrogate setup reported and added',
        frozen_scripts={name:exp.digest(ROOT/name) for name in ('adaptive.py','experiment.py')},
        ablation='staged_fast shares staged acquisition and accelerated ridge, without trust region',
        no_target_labels_in_design=True)
    exp.dump(OUT/'protocol.json', protocol)
    print('Prepared identical 999-token observation and frozen three-method protocol', flush=True)


def guard():
    forbidden=[LONG/'results', original.SHORT/'results_raw_l2_4', exp.ROOT/'results', exp.KVIDENCE/'datasets']
    forbidden=[p.resolve() for p in forbidden]
    def deny(event, args):
        if event!='open' or not isinstance(args[0], (str,bytes,Path)):
            return
        import os
        p=Path(os.fsdecode(args[0])).resolve()
        if 'private_eval' in p.parts or any(p.is_relative_to(d) for d in forbidden):
            raise PermissionError('Recovery cannot open prior results, input datasets or private labels')
    sys.addaudithook(deny)


def preflight():
    guard()
    rt, prior, tokenizer, runtime=exp.load_runtime()
    import torch
    obs=np.load(OUT/'observed/target.npz')
    ids=torch.tensor(json.loads((OUT/'observed/candidate_pool.json').read_text())['token_ids'])
    emb=runtime.embeddings(ids.tolist())
    cache,logits=runtime.prepare(obs['known_prefix_ids'].tolist(),'native')
    order=exp.ordered_probabilities(rt,logits,ids)
    rest=np.setdiff1d(np.arange(len(ids)),order[:50])
    query=np.r_[order[:50],np.random.default_rng(20261019).choice(rest,150,replace=False)]
    before=original.fingerprint(rt,cache)
    values=rt.Oracle(runtime,cache,ids,5,64).query(query)
    assert before==original.fingerprint(rt,cache)
    target=torch.from_numpy(obs['kv'][0])
    distances=np.linalg.norm(values.numpy().astype(np.float64)-target.numpy().astype(np.float64),axis=-1)
    engine=FastSurrogate(rt,emb)
    fit=engine.fit(query.tolist(),values,target,distances,True)
    mean,scale,xm,xc,dual,ymean,ys,ym=fit['reference_factors']
    subset=np.random.default_rng(187).choice(len(ids),1024,replace=False)
    subset=np.unique(np.r_[subset,np.argsort(fit['scores'])[:20]])
    # Direct double-precision reconstruction independently checks the factorization.
    x=((emb[torch.tensor(subset)].double()-mean.double())/scale.double()-xm.double())
    predicted=(x@xc.double().T@dual.double()+ymean.double())*ys.double()+ym.double()
    reference=(predicted-target.double()).square().sum(-1).numpy()
    np.testing.assert_allclose(fit['scores'][subset],reference,rtol=3e-4,atol=.05)
    legacy=shared.rank_candidates(rt,prior,emb,query.tolist(),values,target)
    available=np.ones(len(ids),dtype=bool);available[query]=False
    from adaptive import best_indices
    fast=best_indices(fit['scores'],available,20)
    overlap=len(set(fast)&set(legacy[:20].tolist()))
    assert overlap>=18, overlap
    # Feedback changes step size in both directions, and rejects bad predictions.
    assert trust_update(1.,100.,25.,10.,5.,True)[0]>1
    assert trust_update(1.,100.,25.,10.,11.,True)[0]<1
    assert trust_update(1.,25.,100.,10.,5.,True)[0]<1
    exp.dump(OUT/'preflight.json',dict(passed=True,compared_predictions=len(subset),
        maximum_absolute_squared_l2_error=float(np.max(np.abs(fit['scores'][subset]-reference))),
        fast_legacy_top20_overlap=overlap, prefix_cache_unchanged=True,
        feedback_expand_shrink_and_bad_prediction_checks=True,
        labels_or_private_prefix_used=False))
    print('PREFLIGHT PASSED', overlap, '/20 overlap',flush=True)


def run(args):
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert json.loads((OUT/'preflight.json').read_text())['passed']
    for name,digest in protocol['frozen_scripts'].items():
        assert exp.digest(ROOT/name)==digest, 'Code changed after protocol freeze'
    guard()
    rt,prior,tokenizer,runtime=exp.load_runtime()
    import torch
    obs=np.load(OUT/'observed/target.npz')
    ids=torch.tensor(json.loads((OUT/'observed/candidate_pool.json').read_text())['token_ids'])
    emb=runtime.embeddings(ids.tolist())
    for method in args.methods:
        dest=OUT/method
        if dest.exists():
            raise FileExistsError(f'Preserve completed/partial run: {method}')
        dest.mkdir()
        cache,logits=runtime.prepare(obs['known_prefix_ids'].tolist(),'native')
        setup_started=time.perf_counter()
        engine=FastSurrogate(rt,emb) if method!='hybrid_50_150' else None
        setup_seconds=time.perf_counter()-setup_started
        traces,recovered=[],[]
        advance=0.
        started=time.perf_counter()
        with (dest/'trace.jsonl').open('w',encoding='utf-8') as stream:
            for j,position in enumerate(obs['positions']):
                assert cache[0].offset==int(position)
                tick=time.perf_counter()
                order=exp.ordered_probabilities(rt,logits,ids)
                order_seconds=time.perf_counter()-tick
                target=torch.from_numpy(obs['kv'][j])
                if method=='hybrid_50_150':
                    row=shared.search(rt,prior,runtime,cache,ids,emb,target,order,method,j)
                else:
                    row=search(rt,runtime,cache,ids,emb,target,order,j,engine,method=='adaptive_trust')
                row.update(method=method,position=int(position),probability_ranking_seconds=order_seconds,
                    probability_top50_token_ids=ids[torch.tensor(order[:50])].tolist())
                traces.append(row)
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                stream.flush()
                if row['accepted_token_id'] is None:
                    break
                tid=row['accepted_token_id'];recovered.append(tid)
                tick=time.perf_counter()
                runtime.base.layers=runtime.full_layers
                out=runtime.model(rt.mx.array([[tid]]),cache=cache)
                rt.mx.eval(out);logits=out[0,-1]
                advance+=time.perf_counter()-tick
                if (j+1)%50==0 or j+1==len(obs['positions']):
                    print(method,j+1,'/',len(obs['positions']),'queries',row['queries'],
                        'elapsed',round(time.perf_counter()-started,2),flush=True)
        seconds=time.perf_counter()-started
        exp.dump(dest/'run.json',dict(method=method,recovered_token_ids=recovered,
            recovered_text=tokenizer.decode(recovered,clean_up_tokenization_spaces=False),
            target_tokens=len(obs['positions']),attempted_positions=len(traces),
            total_queries=sum(x['queries'] for x in traces),seconds=seconds,
            surrogate_setup_seconds=setup_seconds,recovery_with_setup_seconds=seconds+setup_seconds,
            forward_seconds=sum(x['forward_seconds'] for x in traces),
            fitting_and_ranking_seconds=sum(x['ranking_seconds']+x['probability_ranking_seconds'] for x in traces),
            advance_seconds=advance,known_correct_prefix=False,true_previous_tokens_used=False,
            private_file_access_guard=True,stopped_unresolved=len(recovered)!=len(obs['positions'])))
        exp.dump(dest/'trace.json',traces)
        del engine,cache,traces
        rt.mx.clear_cache()


def evaluate():
    truth=json.loads((LONG/'results/private_eval/truth.json').read_text())
    actual=truth['payload_ids'];obs=np.load(OUT/'observed/target.npz')
    pool=np.asarray(json.loads((OUT/'observed/candidate_pool.json').read_text())['token_ids'])
    pool_set=set(pool.tolist())
    result={}
    for method in METHODS:
        folder=OUT/method
        run=json.loads((folder/'run.json').read_text())
        traces=json.loads((folder/'trace.json').read_text())
        assert traces==[json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()]
        accepted=[]
        for j,row in enumerate(traces):
            q=row['queried_token_ids']
            assert len(q)==len(set(q))==row['queries'] and set(q)<=pool_set
            top=row['probability_top50_token_ids'];assert q[:50]==top
            if len(q)>50:
                remaining=np.setdiff1d(np.arange(len(pool)),np.flatnonzero(np.isin(pool,top)))
                random=np.random.default_rng(20261019+j).choice(remaining,150,replace=False)
                assert q[50:200]==pool[random].tolist()
            if row['accepted_token_id'] is not None:
                d=float(np.linalg.norm(np.asarray(row['accepted_kv'],dtype=np.float64)-obs['kv'][j].astype(np.float64)))
                np.testing.assert_allclose(d,row['accepted_raw_l2'],rtol=1e-12,atol=1e-12)
                assert d<4 and row['accepted_token_id'] in q
                accepted.append(row['accepted_token_id'])
            else:
                assert row['finite_pool_exhausted']
        assert accepted==run['recovered_token_ids']
        assert run['total_queries']==sum(x['queries'] for x in traces)
        matches=[a==b for a,b in zip(accepted,actual)]
        run.update(correct_tokens=sum(matches),incorrect_acceptances=len(matches)-sum(matches),
            all_exact=accepted==actual,exact_ciphertext_text=run['recovered_text']==truth['payload'],
            maximum_queries_at_one_position=max(x['queries'] for x in traces),
            accepted_l2_max=max(x['accepted_raw_l2'] for x in traces if x['accepted_token_id'] is not None),
            stopped_after_top50=sum(x['queries']==50 for x in traces))
        if method=='adaptive_trust':
            steps=[s for x in traces for s in x['stages'] if s['phase']=='adaptive_trust']
            run['trust_feedback']=dict(rounds=len(steps),
                shrinks=sum(s['radius_state_after']<s['radius_state_before'] for s in steps),
                expands=sum(s['radius_state_after']>s['radius_state_before'] for s in steps),
                invalid_predicted_reductions=sum(s['rho'] is None for s in steps))
        result[method]=run
    exp.dump(OUT/'evaluation.json',dict(sample_id='0037',target_tokens=len(actual),
        epsilon=4,validation_passed=True,methods=result))
    print(json.dumps({m:{k:r[k] for k in ('correct_tokens','total_queries','seconds','recovery_with_setup_seconds')} for m,r in result.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','preflight','run','evaluate'))
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    args=p.parse_args()
    {'prepare':prepare,'preflight':preflight,'run':lambda:run(args),'evaluate':evaluate}[args.stage]()
