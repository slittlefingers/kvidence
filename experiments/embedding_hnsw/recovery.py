"""Paired full-vocabulary KV recovery: original full scores vs indexed shortlist.

Observation only during recovery; evaluation labels are read in a separate stage.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
FULL=ROOT.parent/'ciphertext_full_vocabulary'
sys.path.insert(0,str(FULL))
spec=importlib.util.spec_from_file_location('full_vocabulary_runner',FULL/'experiment.py')
full=importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)
from scorer import IndexedSurrogate
from index import ASSETS, digest, dump
exp=full.exp
OUT=ROOT/'results'
METHODS=('staged_fast','hnsw_rerank')
SOURCES=[ROOT/'recovery.py',ROOT/'scorer.py',ROOT/'index.py',FULL/'experiment.py',
         FULL/'stable_surrogate.py',full.FAST/'adaptive.py',
         full.LONG/'run_long.py',full.original.SHORT/'run.py',
         exp.ROOT/'experiment.py',exp.OLD/'runtime.py']


def prepare():
    if OUT.exists():
        raise FileExistsError('Preserve previous results')
    (OUT/'observed').mkdir(parents=True)
    for name in ('target.npz','candidate_pool.json'):
        shutil.copy2(FULL/'results/observed'/name,OUT/'observed'/name)
    p=json.loads((FULL/'results/protocol.json').read_text())
    p.update(methods=list(METHODS),index_manifest=json.loads((ASSETS/'manifest.json').read_text()),
        policy='Same probability50/random150. Indexed: inverse ridge KV->raw embedding; at most two HNSW512/1024 rounds; then full scores for remaining search at this position. Global fit excludes the 40 indexed-only probes; all measured pairs remain logged; real raw KV L2<4 accepts.',
        no_global_query_budget=True,initial_seed=20261019,
        target_label_access=False,original_control_unchanged=True,
        index_metric='squared raw embedding L2',
        acceptance_metric='unsquared raw concatenated KV L2 strictly <4',
        timing='Separate offline index build; online total includes per-method table and index load, excludes common model load/public prefix setup. Fixed order control then HNSW; one sample.',
        frozen_sources={str(x):digest(x) for x in SOURCES},
        frozen_observation=digest(OUT/'observed/target.npz'))
    dump(OUT/'protocol.json',p)


def guard():
    paths=[(ROOT.parent/p).resolve() for p in (
        'ciphertext_full_vocabulary/results','ciphertext_hybrid_l2_long/results',
        'ciphertext_adaptive_trust/results','ciphertext_collision_comparison/results',
        'ciphertext_hybrid_l2/results_raw_l2_4')]
    paths.append((exp.KVIDENCE/'datasets').resolve())
    def deny(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,Path)):
            path=Path(os.fsdecode(args[0])).resolve()
            if 'private_eval' in path.parts or any(path.is_relative_to(p) for p in paths):
                raise PermissionError('No private labels, datasets, previous results during recovery')
    sys.addaudithook(deny)


def inputs():
    obs=np.load(OUT/'observed/target.npz')
    pool=json.loads((OUT/'observed/candidate_pool.json').read_text())
    assert pool['token_ids']==list(range(262144))
    return obs,torch.arange(262144),pool


def preflight():
    guard()
    rt,prior,tokenizer,runtime=exp.load_runtime()
    obs,ids,pool=inputs()
    emb=torch.from_numpy(np.load(ASSETS/'embeddings.npy'))
    probe=np.unique(np.r_[np.arange(0,len(ids),997),pool['special_token_ids']])
    torch.testing.assert_close(emb[probe],runtime.embeddings(probe.tolist()),rtol=0,atol=0)
    cache,logits=runtime.prepare(obs['known_prefix_ids'].tolist(),'native')
    before=full.original.fingerprint(rt,cache)
    order=exp.ordered_probabilities(rt,logits,ids)
    rest=np.setdiff1d(np.arange(len(ids)),order[:50])
    q=np.r_[order[:50],np.random.default_rng(20261019).choice(rest,190,replace=False)]
    measured=rt.Oracle(runtime,cache,ids,5,64).query(q)
    target=torch.from_numpy(obs['kv'][0])
    distances=np.linalg.norm(measured.numpy().astype(np.float64)-target.numpy().astype(np.float64),axis=1)
    engine=IndexedSurrogate(rt,emb)
    checks=[]
    for count in (200,220,240):
        fit=engine.fit(q[:count].tolist(),measured[:count],target,distances[:count])
        finite=np.flatnonzero(np.isfinite(fit['scores']))
        assert len(finite)>=20
        subset=finite if len(finite)<2048 else np.unique(np.r_[finite[::257],np.argsort(fit['scores'])[:20]])
        mean,scale,xm,xc,dual,ymean,ys,ym=fit['reference_factors']
        x=(emb[subset].double()-mean.double())/scale.double()-xm.double()
        direct=(x@xc.double().T@dual.double()+ymean.double())*ys.double()+ym.double()
        ref=(direct-target.double()).square().sum(1).numpy()
        np.testing.assert_allclose(fit['scores'][subset],ref,rtol=3e-4,atol=.05)
        checks.append(dict(queries=count,finite_scores=len(finite),
            max_squared_score_error=float(np.max(np.abs(ref-fit['scores'][subset])))))
    assert len(finite)==262144 and before==full.original.fingerprint(rt,cache)
    denied=0
    for path in (full.LONG/'results/private_eval/truth.json',FULL/'results/evaluation.json'):
        try:
            path.read_text()
        except PermissionError:
            denied+=1
    assert denied==2
    dump(OUT/'preflight.json',dict(passed=True,checks=checks,cache_unchanged=True,
        model_embedding_probes=len(probe),denied_label_checks=denied,
        shortlist_and_full_fallback_checked=True))
    print('PREFLIGHT PASSED',checks,flush=True)


def run():
    p=json.loads((OUT/'protocol.json').read_text())
    assert json.loads((OUT/'preflight.json').read_text())['passed']
    for path,h in p['frozen_sources'].items():
        assert digest(Path(path))==h
    assert digest(OUT/'observed/target.npz')==p['frozen_observation']
    guard()
    rt,prior,tokenizer,runtime=exp.load_runtime()
    obs,ids,pool=inputs()
    for method in METHODS:
        dest=OUT/method
        if dest.exists():
            raise FileExistsError('Preserve previous run')
        dest.mkdir()
        cache,logits=runtime.prepare(obs['known_prefix_ids'].tolist(),'native')
        tick=time.perf_counter()
        embeddings=torch.from_numpy(np.load(ASSETS/'embeddings.npy'))
        engine=(full.FastSurrogate if method=='staged_fast' else IndexedSurrogate)(rt,embeddings)
        setup=time.perf_counter()-tick
        traces,recovered=[],[]
        advance=0.
        started=time.perf_counter()
        with (dest/'trace.jsonl').open('w') as stream:
            for j,position in enumerate(obs['positions']):
                assert cache[0].offset==int(position)
                tick=time.perf_counter()
                order=exp.ordered_probabilities(rt,logits,ids)
                probability=time.perf_counter()-tick
                if method=='hnsw_rerank':
                    engine.records=[]
                target=torch.from_numpy(obs['kv'][j])
                row=full.fast_search(rt,runtime,cache,ids,embeddings,target,order,j,engine,False)
                row.update(method=method,position=int(position),probability_ranking_seconds=probability,
                    probability_top50_token_ids=order[:50].tolist(),
                    index_fits=engine.records if method=='hnsw_rerank' else [])
                traces.append(row)
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                if row['accepted_token_id'] is None:
                    break
                chosen=row['accepted_token_id'];recovered.append(chosen)
                tick=time.perf_counter();runtime.base.layers=runtime.full_layers
                output=runtime.model(rt.mx.array([[chosen]]),cache=cache)
                rt.mx.eval(output);logits=output[0,-1]
                advance+=time.perf_counter()-tick
                if (j+1)%25==0 or j+1==len(obs['positions']):
                    print(method,j+1,'/',len(obs['positions']),'queries',row['queries'],
                        'L2',round(row['accepted_raw_l2'],4),'seconds',round(time.perf_counter()-started,2),flush=True)
        seconds=time.perf_counter()-started
        dump(dest/'run.json',dict(method=method,recovered_token_ids=recovered,
            recovered_text=tokenizer.decode(recovered,clean_up_tokenization_spaces=False),
            target_tokens=len(obs['positions']),total_queries=sum(r['queries'] for r in traces),
            seconds=seconds,setup_seconds=setup,total_online_seconds=seconds+setup,
            forward_seconds=sum(r['forward_seconds'] for r in traces),
            surrogate_ranking_seconds=sum(r['ranking_seconds'] for r in traces),
            probability_seconds=sum(r['probability_ranking_seconds'] for r in traces),
            advance_seconds=advance,diagnostics=engine.stats,own_recovered_prefix=True))
        dump(dest/'trace.json',traces)
        del engine,embeddings,cache,traces
        rt.mx.clear_cache()


def evaluate():
    truth=json.loads((full.LONG/'results/private_eval/truth.json').read_text())
    obs,ids,pool=inputs()
    results={}
    for method in METHODS:
        run=json.loads((OUT/method/'run.json').read_text())
        rows=json.loads((OUT/method/'trace.json').read_text())
        assert rows==[json.loads(x) for x in (OUT/method/'trace.jsonl').read_text().splitlines()]
        accepted=[]
        for j,row in enumerate(rows):
            q=row['queried_token_ids']
            assert len(q)==len(set(q))==row['queries'] and all(0<=t<262144 for t in q)
            assert q[:50]==row['probability_top50_token_ids']
            if len(q)>50:
                rest=np.setdiff1d(np.arange(262144),q[:50])
                assert q[50:200]==np.random.default_rng(20261019+j).choice(rest,150,replace=False).tolist()
            chosen=row['accepted_token_id']
            if chosen is not None:
                d=float(np.linalg.norm(np.asarray(row['accepted_kv'],dtype=np.float64)-obs['kv'][j].astype(np.float64)))
                np.testing.assert_allclose(d,row['accepted_raw_l2'],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(d,row['raw_l2_distances'][q.index(chosen)],rtol=1e-12,atol=1e-12)
                assert d<4
                accepted.append(chosen)
        assert accepted==run['recovered_token_ids']
        assert run['total_queries']==sum(r['queries'] for r in rows)
        correct=sum(a==b for a,b in zip(accepted,truth['payload_ids']))
        run.update(correct_tokens=correct,incorrect_acceptances=len(accepted)-correct,
            exact_ids=accepted==truth['payload_ids'],exact_text=run['recovered_text']==truth['payload'],
            max_queries=max(r['queries'] for r in rows),max_accepted_l2=max(r['accepted_raw_l2'] for r in rows),
            indexed_positions=sum(bool(r['index_fits']) for r in rows),
            positions_with_global_fallback=sum(any(f['full_table'] for f in r['index_fits']) for r in rows))
        results[method]=run
    dump(OUT/'evaluation.json',dict(validation_passed=True,methods=results))
    print(json.dumps({m:{k:v[k] for k in ['correct_tokens','incorrect_acceptances','total_queries','total_online_seconds']} for m,v in results.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','preflight','run','evaluate'])
    args=p.parse_args()
    globals()[args.stage]()
