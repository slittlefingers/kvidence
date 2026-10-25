"""Held-out, true-prefix NPU forward test and equal-query warm-start search.

Truth embeddings are used only for prefix replay and explicit floor controls.
Search receives current target KV and a learned initialization, never its label.
"""
import argparse
import json
import mmap
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from kvrecon.common import WORKSPACE,EXPERIMENTS,digest,dump
from kvrecon.npu import NpuBackend
sys.path.insert(0,str(EXPERIMENTS/'npu_candidate_batch'))
from prepare import FlatReader
EXPERIMENT=WORKSPACE/'kvidence/publications/gemma4-image-prefix-dataset/experiments/20261022_npu_prefix'
sys.path.insert(0,str(EXPERIMENT))
from train import predict_ridge
from infer import load_model


def public_projection(sections):
    path=WORKSPACE/'annoymous/models/gemma-4-E2B-it_Google_Tensor_G5.litertlm'
    with path.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as raw:
        r=FlatReader(raw);begin=sections['tf_lite_vision_adapter']['offset']
        model=begin+r.scalar_at(begin);graphs=r.vector(model,2,tables=True)
        buffers=r.vector(model,4,tables=True);found=[]
        for sig in r.vector(model,7,tables=True):
            if r.string(sig,2)!='vision_adapter_280':continue
            for tensor in r.vector(graphs[r.scalar(sig,4)],0,tables=True):
                if r.vector(tensor,0)==[1536,768]:
                    p=r.table(buffers[r.scalar(tensor,2)],0)
                    if p:
                        assert r.scalar_at(p)==1536*768*4
                        found.append(np.frombuffer(raw,dtype='<f4',count=1536*768,offset=p+4).copy().reshape(1536,768))
    assert len(found)==1
    w=torch.from_numpy(found[0]).double()
    inverse=torch.linalg.solve(w.T@w+torch.eye(768,dtype=torch.float64)*1e-12,w.T).T
    return inverse


def unit(a):
    a=np.asarray(a,dtype=np.float32)
    return a/np.maximum(np.sqrt(np.mean(a.astype(np.float64)**2,axis=-1,keepdims=True)),1e-12)


def metrics(codes,target,quant):
    d=np.asarray(codes,dtype=np.float64)-np.asarray(target,dtype=np.float64)
    scale=np.concatenate([np.full(q['width'],q['scale']) for q in quant])
    return dict(rmse_lsb=float(np.sqrt(np.mean(d*d))),
        k_rmse_lsb=float(np.sqrt(np.mean(d[:512]**2))),
        v_rmse_lsb=float(np.sqrt(np.mean(d[512:]**2))),
        dequantized_rmse=float(np.sqrt(np.mean((d*scale)**2))),
        exact_fraction=float(np.mean(d==0)))


def proposals(directions,values,center_codes,target,radius):
    d=torch.from_numpy(directions.copy()).double()
    y=torch.from_numpy(values.astype(np.float64))
    j=(y[:768]-y[768:])/(2*radius)
    eig,vec=torch.linalg.eigh(j@j.T);eig=eig.clamp_min(0)
    rhs=vec.T@(j@torch.from_numpy((target-center_codes).astype(np.float64)))
    scale=max(float(eig.mean()),1e-8);steps=[]
    for damping in [.001,.01,.1,1.]:
        weights=vec@(rhs/(eig+damping*scale))
        weights*=min(1.,radius/max(float(weights.norm()),1e-12))
        direction=weights@d
        for factor in [1/64,1/32,1/16,1/8,.25,.5,1.,2.,4.]:
            steps.append((direction*factor).float().numpy())
    return np.stack(steps)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=HERE/'results/20261022_inverse_npu')
    ap.add_argument('--rounds',type=int,default=4)
    args=ap.parse_args();torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    data=HERE/'data/20261022_pilot_prefill'
    protocol=json.loads((data/'protocol.json').read_text())
    row=next(x for x in json.loads((data/'index.json').read_text()) if x['split']=='test')
    with np.load(data/row['file']) as f:z=f['embeddings'].copy();target=f['kv_codes'].copy()
    assert digest(data/row['file'])==row['sha256']
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    folder=EXPERIMENT/'results_seed20261022'
    ridge=torch.load(folder/'ridges.pt',weights_only=True,map_location='cpu')['kv']
    model=load_model(folder/'prefix_last.pt','cpu')
    inverse=public_projection(sections)
    def latent(embedding):
        return unit((torch.as_tensor(embedding).double()@inverse).numpy()).astype(np.float32)
    with torch.no_grad():ridge_all=predict_ridge(torch.from_numpy(target),ridge).numpy()
    points=[p for image in row['images'] for p in range(image['start'],image['start']+image['rows'],128) if p>=10]
    search_points=[points[0],points[len(points)//2],points[-1]]
    bases=[];generator=torch.Generator().manual_seed(20261022)
    for _ in range(args.rounds):
        q,_=torch.linalg.qr(torch.randn(768,768,generator=generator,dtype=torch.float64))
        bases.append((q.T*np.sqrt(768)).float().numpy())
    dump(args.output/'protocol.json',dict(test_sequence=row['id'],split='test',
        points=points,search_points=search_points,rounds=args.rounds,
        initial_pool_queries=64,queries_per_round=1572,probe_rms=.08713840313369006,
        seed=20261022,prefix='true replay, original per-image prefill blocks',
        learned_models={p.name:digest(p) for p in [folder/'ridges.pt',folder/'prefix_last.pt']},
        source_sha256=digest(Path(__file__)),
        adapter_projection='public actual adapter matrix left inverse, RMS-normalize, phone CPU adapter',
        search='full 768 paired secants, 4 dampings x 9 step factors, keep best measured; double radius after zero gain up to 4x',
        target_embeddings_used_for='true prefixes, explicit controls and embedding diagnostics only; not candidate/search input'))
    results=[];searches=[]
    with NpuBackend(sections,4,args.output/'device',protocol['model_sha256'],serial='YOUR_DEVICE_SERIAL') as b:
        dump(args.output/'quantization.json',b.quant)
        b.reset(protocol['initial_prefix_ids']);started=time.perf_counter()
        def score(u):
            codes=[]
            for begin in range(0,len(u),128):codes.append(b.query_vision(np.asarray(u[begin:begin+128],dtype=np.float32)))
            return np.concatenate(codes)
        def search(p,arm,initial):
            tick=time.perf_counter();q=score(initial);dist=np.sqrt(np.mean((q.astype(float)-target[p])**2,axis=1))
            index=int(dist.argmin());u=initial[index].copy();codes=q[index].astype(float);best=float(dist[index])
            trace=[dict(round=0,queries=64,rmse=best,seconds=time.perf_counter()-tick)]
            queries=64;radius=.08713840313369006
            for iteration,d in enumerate(bases,1):
                before=best;center=u.copy();center_codes=codes.copy()
                probes=np.concatenate([center+radius*d,center-radius*d]);values=score(probes)
                steps=proposals(d,values,center_codes,target[p].astype(float),radius)
                updates=center+steps;v=score(updates)
                pool=np.concatenate([probes,updates]);all_q=np.concatenate([values,v]).astype(float)
                ds=np.sqrt(np.mean((all_q-target[p])**2,axis=1));j=int(ds.argmin())
                if ds[j]<best:u=pool[j].copy();codes=all_q[j].copy();best=float(ds[j])
                queries+=len(pool)
                trace.append(dict(round=iteration,queries=queries,rmse=best,probe_rms=radius,seconds=time.perf_counter()-tick))
                if best>=before:radius=min(radius*2,.08713840313369006*4)
                print('SEARCH',p,arm,iteration,round(best,5),queries,flush=True)
            np.savez_compressed(args.output/f'search_{p}_{arm}.npz',latent=u,kv=codes)
            result=dict(position=p,arm=arm,trace=trace,**metrics(codes,target[p],b.quant))
            dump(args.output/f'search_{p}_{arm}.json',result)
            return result
        for item in row['images']:
            start=item['start'];end=start+item['rows'];assert b.position==start
            for p in range(start,end,128):
                if p in points:
                    with torch.no_grad():prefix=model(torch.from_numpy(target[p])[None,None],torch.tensor([p]),torch.from_numpy(z[:p])[None])[0,0].numpy()
                    labels=['truth_floor','ridge_direct','prefix_last_direct','mean','previous_true']
                    emb=np.stack([z[p],ridge_all[p],prefix,ridge['ym'].numpy(),z[p-1]])
                    direct=b.query_embeddings(emb)
                    u=latent(emb[:3]);adapted=b.project_vision(u)
                    projected=b.query_vision(u)
                    point=dict(position=p,image_id=item['image_id'],methods={name:metrics(q,target[p],b.quant) for name,q in zip(labels,direct)})
                    for i,name in enumerate(['truth_projected_control','ridge_projected','prefix_last_projected']):
                        point['methods'][name]=dict(**metrics(projected[i],target[p],b.quant),
                            embedding_rmse=float(np.sqrt(np.mean((adapted[i].astype(float)-z[p])**2))))
                    np.savez_compressed(args.output/f'point_{p}.npz',predicted_embeddings=emb[1:3],direct_kv=direct,projected_latents=u,projected_embeddings=adapted,projected_kv=projected,target_kv=target[p])
                    results.append(point);dump(args.output/'forward_results.json',results)
                    print('POINT',p,'ridge',point['methods']['ridge_direct']['rmse_lsb'],'projected',point['methods']['ridge_projected']['rmse_lsb'],'floor',point['methods']['truth_floor']['rmse_lsb'],flush=True)
                    if p in search_points:
                        rng=np.random.default_rng(217+p)
                        random=unit(rng.standard_normal((64,768)).astype(np.float32)).astype(np.float32)
                        noise=rng.standard_normal((63,768)).astype(np.float32)
                        scales=np.resize(np.array([.03,.1,.3,1.],np.float32),63)
                        learned=np.concatenate([u[1:2],unit(u[1]+noise*scales[:,None])]).astype(np.float32)
                        searches.append(search(p,'learned',learned));searches.append(search(p,'random',random))
                count=min(128,end-p);part=np.asarray(z[p:p+count],dtype='<f4')
                raw=b.rpc('collect_embeddings',rows=count,position=p,features_hex=part.tobytes().hex())
                observed=np.frombuffer(raw,np.int8).reshape(count,1024)
                np.testing.assert_array_equal(observed,target[p:p+count])
            if end<len(z):b.commit_embeddings(z[end:end+1])
            if end+1<len(z):b.commit(255999)
        aggregate={name:dict(mean_rmse_lsb=float(np.mean([x['methods'][name]['rmse_lsb'] for x in results])),
            median_rmse_lsb=float(np.median([x['methods'][name]['rmse_lsb'] for x in results]))) for name in results[0]['methods']}
        summary=dict(points=len(results),searches=searches,aggregate=aggregate,
            seconds=time.perf_counter()-started,canonical_prefix_replay_exact=True,
            limits=['true prefixes only','block-start subset of one held-out sequence','projected/direct spaces differ','search is finite-budget full-basis pilot, not historical run comparison'])
        dump(args.output/'summary.json',summary);print('COMPLETE',json.dumps(aggregate),flush=True)


if __name__=='__main__':main()
