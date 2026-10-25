"""Full original vocabulary: Collision+ probability ordering vs staged fast ridge.

Use the same 999-token observation, unmodified old search functions, own recovered
prefixes, and actual raw K+V L2 < 4. No language/alphabet/special-token exclusion.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
LONG=ROOT.parent/'ciphertext_hybrid_l2_long'
FAST=ROOT.parent/'ciphertext_adaptive_trust'
POOL=ROOT.parent/'full_vocabulary_candidates/candidate_pool.json'
sys.path.insert(0,str(LONG))
sys.path.insert(0,str(FAST))
import run_long as original
from adaptive import search as fast_search
from stable_surrogate import StableSurrogate as FastSurrogate
shared,exp=original.shared,original.exp
OUT=ROOT/'results'
METHODS=('collision_plus_l2','staged_fast')
FROZEN=('experiment.py','stable_surrogate.py','../ciphertext_adaptive_trust/adaptive.py',
        '../ciphertext_hybrid_l2/run.py','../kv_guided_search_gemma4_int8/runtime.py')


def prepare():
    if OUT.exists():
        raise FileExistsError('Preserve existing results before a new run')
    pool=json.loads(POOL.read_text())
    assert pool['token_ids']==list(range(262144))
    assert pool['count']==pool['full_vocabulary']==262144
    assert not pool['special_tokens_excluded']
    (OUT/'observed').mkdir(parents=True)
    shutil.copy2(LONG/'results/observed/target.npz',OUT/'observed/target.npz')
    shutil.copy2(POOL,OUT/'observed/candidate_pool.json')
    shutil.copy2(LONG/'results/cache_validation.json',OUT/'cache_validation.json')
    source=json.loads((LONG/'results/protocol.json').read_text())
    exp.dump(OUT/'protocol.json',dict(
        sample_id='0037',target_tokens=source['target_tokens'],
        rendered_chat_tokens=source['rendered_chat_tokens'],known_prefix_tokens=source['known_prefix_tokens'],
        layer_1based=5,metric='raw Euclidean L2 of concatenated K+V, 1024 coordinates',
        epsilon=4.,strict_less_than=True,kv_dtype=source['kv_dtype'],model=source['model'],
        candidate_count=262144,original_embedding_rows=[262144,1536],
        candidate_rule='Every original token ID 0..262143, including all 24 special IDs',
        language_filter=False,alphabet_filter=False,special_tokens_excluded=False,
        no_600_query_limit=True,no_lsb_normalization=True,no_rms_normalization=True,
        no_coordinate_gate=True,methods=list(METHODS),
        collision_plus_policy='Full-model probability order, batches of 20; first real raw L2<4 candidate in order',
        fast_policy='Check probability top50, then uniform random150 from all remaining IDs, then global standardized ridge proposals20',
        ridge=.01,max_regression_anchors=512,query_budget=None,oracle_microbatch=64,
        fast_surrogate='Same standardized ridge objective and staged selection; QR residual factorization for full-vocabulary numerical stability; no trust region',
        numerical_change='First full-vocabulary preflight rejected the float32 expanded quadratic; archived before any timed run, replaced by orthogonal residual scoring without dropping IDs',
        seed=20261019,own_recovered_prefix=True,true_previous_tokens_used=False,
        independent_KV8_quantization=False,native_sliding_cache=True,native_window=512,
        candidate_pool_sha256=exp.digest(OUT/'observed/candidate_pool.json'),
        source_observation_sha256=exp.digest(OUT/'observed/target.npz'),
        reused_observation_source=str(LONG/'results/observed/target.npz'),
        frozen_scripts={name:exp.digest(ROOT/name) for name in FROZEN},
        timing='Loop plus method-specific embedding extraction and surrogate setup; common model loading and initial public prefix excluded'))
    print('PREPARED full vocabulary 262144, same target 999, special IDs included',flush=True)


def guard():
    forbidden=[LONG/'results',FAST/'results',original.SHORT/'results_raw_l2_4',
               exp.ROOT/'results',exp.KVIDENCE/'datasets']
    forbidden=[p.resolve() for p in forbidden]
    def deny(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,Path)):
            return
        path=Path(os.fsdecode(args[0])).resolve()
        if 'private_eval' in path.parts or any(path.is_relative_to(p) for p in forbidden):
            raise PermissionError('Recovery cannot open previous results, datasets or evaluation labels')
    sys.addaudithook(deny)


def inputs(runtime):
    import torch
    obs=np.load(OUT/'observed/target.npz')
    pool=json.loads((OUT/'observed/candidate_pool.json').read_text())
    assert pool['token_ids']==list(range(runtime.lm.args.vocab_size))
    return obs,torch.tensor(pool['token_ids'],dtype=torch.long),pool


def preflight():
    guard()
    rt,prior,tokenizer,runtime=exp.load_runtime()
    import torch
    obs,ids,pool=inputs(runtime)
    emb=runtime.embeddings(ids.tolist())
    assert list(emb.shape)==[262144,1536]
    assert torch.isfinite(emb).all()
    cache,logits=runtime.prepare(obs['known_prefix_ids'].tolist(),'native')
    order=exp.ordered_probabilities(rt,logits,ids)
    remaining=np.setdiff1d(np.arange(len(ids)),order[:50])
    initial=np.r_[order[:50],np.random.default_rng(20261019).choice(remaining,150,replace=False)]
    before=original.fingerprint(rt,cache)
    oracle=rt.Oracle(runtime,cache,ids,5,64)
    values=oracle.query(initial)
    special=oracle.query(pool['special_token_ids'])
    assert torch.isfinite(special).all() and before==original.fingerprint(rt,cache)
    target=torch.from_numpy(obs['kv'][0])
    distances=np.linalg.norm(values.numpy().astype(np.float64)-target.numpy().astype(np.float64),axis=-1)
    engine=FastSurrogate(rt,emb)
    fit=engine.fit(initial.tolist(),values,target,distances,geometry=False)
    assert np.isfinite(fit['scores']).all() and len(fit['scores'])==262144
    mean,scale,xm,xc,dual,ymean,ys,ym=fit['reference_factors']
    subset=np.unique(np.r_[np.random.default_rng(187).choice(len(ids),1024,replace=False),
                           np.argsort(fit['scores'])[:20],pool['special_token_ids'],0,len(ids)-1])
    # Independent float64 residual reconstruction without a full Nx1024 allocation.
    x=(emb[torch.tensor(subset)].double()-mean.double())/scale.double()-xm.double()
    direct=(x@xc.double().T@dual.double()+ymean.double())*ys.double()+ym.double()
    reference=(direct-target.double()).square().sum(-1).numpy()
    np.testing.assert_allclose(fit['scores'][subset],reference,rtol=3e-4,atol=.05)
    exp.dump(OUT/'preflight.json',dict(passed=True,embedding_shape=list(emb.shape),
        candidate_scores=262144,special_ids_tested=len(special),special_candidate_forwards_finite=True,
        prefix_cache_unchanged=True,compared_float64_predictions=len(subset),
        maximum_absolute_squared_l2_error=float(np.max(np.abs(reference-fit['scores'][subset]))),
        no_target_labels_or_true_prefix_used=True))
    print('PREFLIGHT PASSED: full embeddings and scores; all special IDs forward; factorization checked',flush=True)


def run(args):
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert json.loads((OUT/'preflight.json').read_text())['passed']
    for name,digest in protocol['frozen_scripts'].items():
        assert exp.digest(ROOT/name)==digest, 'Frozen code changed'
    guard()
    rt,prior,tokenizer,runtime=exp.load_runtime()
    import torch
    obs,ids,pool=inputs(runtime)
    for method in args.methods:
        dest=OUT/method
        if dest.exists():
            raise FileExistsError(f'Preserve previous/partial run: {method}')
        dest.mkdir()
        cache,logits=runtime.prepare(obs['known_prefix_ids'].tolist(),'native')
        embeddings,engine=None,None
        embedding_setup,surrogate_setup=0.,0.
        if method=='staged_fast':
            tick=time.perf_counter();embeddings=runtime.embeddings(ids.tolist())
            embedding_setup=time.perf_counter()-tick
            tick=time.perf_counter();engine=FastSurrogate(rt,embeddings)
            surrogate_setup=time.perf_counter()-tick
        traces,recovered=[],[]
        advance=0.
        started=time.perf_counter()
        with (dest/'trace.jsonl').open('w',encoding='utf-8') as stream:
            for j,position in enumerate(obs['positions']):
                assert cache[0].offset==int(position)
                tick=time.perf_counter();order=exp.ordered_probabilities(rt,logits,ids)
                probability_seconds=time.perf_counter()-tick
                target=torch.from_numpy(obs['kv'][j])
                if method=='collision_plus_l2':
                    row=shared.search(rt,prior,runtime,cache,ids,None,target,order,method,j)
                else:
                    row=fast_search(rt,runtime,cache,ids,embeddings,target,order,j,engine,False)
                row.update(method=method,position=int(position),
                    probability_ranking_seconds=probability_seconds,
                    probability_top50_token_ids=ids[torch.tensor(order[:50])].tolist())
                traces.append(row)
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                if row['accepted_token_id'] is None:
                    print(method,'UNRESOLVED',j+1,'queries',row['queries'],flush=True)
                    break
                chosen=row['accepted_token_id'];recovered.append(chosen)
                tick=time.perf_counter();runtime.base.layers=runtime.full_layers
                output=runtime.model(rt.mx.array([[chosen]]),cache=cache)
                rt.mx.eval(output);logits=output[0,-1]
                advance+=time.perf_counter()-tick
                if (j+1)%25==0 or j+1==len(obs['positions']):
                    print(method,j+1,'/',len(obs['positions']),'queries',row['queries'],
                        'L2',round(row['accepted_raw_l2'],4),
                        'elapsed',round(time.perf_counter()-started,2),flush=True)
        seconds=time.perf_counter()-started
        exp.dump(dest/'run.json',dict(method=method,recovered_token_ids=recovered,
            recovered_text=tokenizer.decode(recovered,clean_up_tokenization_spaces=False),
            target_tokens=len(obs['positions']),attempted_positions=len(traces),candidate_count=len(ids),
            total_queries=sum(x['queries'] for x in traces),seconds=seconds,
            embedding_setup_seconds=embedding_setup,surrogate_setup_seconds=surrogate_setup,
            recovery_with_setup_seconds=seconds+embedding_setup+surrogate_setup,
            forward_seconds=sum(x['forward_seconds'] for x in traces),
            fitting_and_ranking_seconds=sum(x['ranking_seconds']+x['probability_ranking_seconds'] for x in traces),
            advance_seconds=advance,surrogate_diagnostics=engine.stats if engine is not None else None,
            true_previous_tokens_used=False,private_file_access_guard=True,
            stopped_unresolved=len(recovered)!=len(obs['positions'])))
        exp.dump(dest/'trace.json',traces)
        del embeddings,engine,cache,traces
        rt.mx.clear_cache()


def evaluate():
    truth=json.loads((LONG/'results/private_eval/truth.json').read_text())
    actual=truth['payload_ids'];obs=np.load(OUT/'observed/target.npz')
    pool=json.loads((OUT/'observed/candidate_pool.json').read_text())
    assert pool['token_ids']==list(range(262144))
    results={}
    for method in METHODS:
        dest=OUT/method
        run=json.loads((dest/'run.json').read_text())
        traces=json.loads((dest/'trace.json').read_text())
        assert traces==[json.loads(line) for line in (dest/'trace.jsonl').read_text().splitlines()]
        accepted=[]
        for j,row in enumerate(traces):
            q=row['queried_token_ids'];top=row['probability_top50_token_ids']
            assert len(q)==len(set(q))==row['queries'] and all(0<=t<262144 for t in q)
            if method=='staged_fast':
                assert q[:50]==top
                if len(q)>50:
                    remaining=np.setdiff1d(np.arange(262144),top)
                    random=np.random.default_rng(20261019+j).choice(remaining,150,replace=False)
                    assert q[50:200]==random.tolist()
            else:
                assert q[:min(len(q),50)]==top[:min(len(q),50)]
            assert len(row['raw_l2_distances'])==len(q)
            if row['accepted_token_id'] is not None:
                tid=row['accepted_token_id']
                d=float(np.linalg.norm(np.asarray(row['accepted_kv'],dtype=np.float64)-obs['kv'][j].astype(np.float64)))
                np.testing.assert_allclose(d,row['accepted_raw_l2'],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(d,row['raw_l2_distances'][q.index(tid)],rtol=1e-12,atol=1e-12)
                assert d<4
                accepted.append(tid)
            else:
                assert row['finite_pool_exhausted'] and len(q)==262144
        assert accepted==run['recovered_token_ids']
        assert run['total_queries']==sum(x['queries'] for x in traces)
        matches=[a==b for a,b in zip(accepted,actual)]
        run.update(correct_tokens=sum(matches),incorrect_acceptances=len(matches)-sum(matches),
            all_exact=accepted==actual,exact_ciphertext_text=run['recovered_text']==truth['payload'],
            maximum_queries_at_one_position=max(x['queries'] for x in traces),
            accepted_l2_max=max((x['accepted_raw_l2'] for x in traces if x['accepted_token_id'] is not None),default=None),
            stopped_after_top50=sum(x['queries']==50 for x in traces))
        results[method]=run
    exp.dump(OUT/'evaluation.json',dict(sample_id='0037',target_tokens=len(actual),
        candidate_count=262144,epsilon=4,validation_passed=True,methods=results))
    print(json.dumps({m:{k:r[k] for k in ('correct_tokens','incorrect_acceptances','total_queries','seconds','recovery_with_setup_seconds')} for m,r in results.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','preflight','run','evaluate'))
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    args=p.parse_args()
    {'prepare':prepare,'preflight':preflight,'run':lambda:run(args),'evaluate':evaluate}[args.stage]()
