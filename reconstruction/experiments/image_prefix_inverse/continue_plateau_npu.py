"""Continue a measured warm start to an operational multiscale plateau."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from kvrecon.common import EXPERIMENTS,dump,digest
from kvrecon.npu import NpuBackend


def updates(directions,values,center,target,radius):
    d=torch.from_numpy(directions).double()
    y=torch.from_numpy(values.astype(np.float64))
    j=(y[:768]-y[768:])/(2*radius)
    eig,vec=torch.linalg.eigh(j@j.T);eig=eig.clamp_min(0)
    rhs=vec.T@(j@torch.from_numpy(target-center))
    scale=max(float(eig.mean()),1e-8);steps=[]
    for damping in [.001,.01,.1,1.]:
        weights=vec@(rhs/(eig+damping*scale))
        weights*=min(1.,radius/max(float(weights.norm()),1e-12))
        direction=weights@d
        for factor in [1/1024,1/512,1/256,1/128,1/64,1/32,1/16,1/8,.25,.5,1.,2.,4.,8.]:
            steps.append((direction*factor).float().numpy())
    return np.stack(steps)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=HERE/'results/20261023_position1976_plateau')
    ap.add_argument('--max-rounds',type=int,default=60)
    ap.add_argument('--plateau-rounds',type=int,default=8)
    ap.add_argument('--warm',type=Path,default=HERE/'results/20261022_inverse_npu/search_1976_learned.npz')
    ap.add_argument('--base-radius',type=float,default=.08713840313369006)
    ap.add_argument('--seed',type=int,default=20261023)
    args=ap.parse_args();torch.set_num_threads(1)
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    warm=args.warm
    with np.load(warm) as f:u=f['latent'].copy();expected=f['kv'].copy()
    data=HERE/'data/20261022_pilot_prefill';p=1976
    config=json.loads((data/'protocol.json').read_text())
    row=next(x for x in json.loads((data/'index.json').read_text()) if x['split']=='test')
    with np.load(data/row['file']) as f:
        prefix=f['embeddings'][:p].copy();target=f['kv_codes'][p].astype(np.float64)
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    base=args.base_radius;seed=args.seed
    if not np.isfinite(base) or base<=0:raise ValueError('Positive finite base radius required')
    dump(out/'protocol.json',dict(position=p,seed=seed,warm_start_sha256=digest(warm),
        source_sha256=digest(Path(__file__)),target='original INT8 KV, LSB RMSE',
        prefix='true prefix, exact original prefill blocks; current true embedding/latent not used',
        fine_anchor_scales=[base/64,base/4],main_scale=base,
        expanded_main='2x after first flat round; 4x after second, capped',
        per_round='four scales of full 768 paired probes, fine line search, 128 tiny random jitters',
        plateau_rounds=args.plateau_rounds,max_rounds=args.max_rounds,epsilon=.125))
    rng=np.random.default_rng(seed);generator=torch.Generator().manual_seed(seed)
    queries=0;best=float('inf');best_u=None;best_q=None;trace=[];flat=0
    with NpuBackend(sections,4,out/'device',config['model_sha256'],serial='YOUR_DEVICE_SERIAL') as b:
        b.reset(config['initial_prefix_ids'])
        for image in row['images']:
            start=image['start'];end=min(image['start']+image['rows'],p)
            for a in range(start,end,128):
                chunk=np.asarray(prefix[a:min(a+128,end)],dtype='<f4')
                b.rpc('collect_embeddings',rows=len(chunk),position=a,features_hex=chunk.tobytes().hex())
            if b.position==p:break
            b.commit_embeddings(prefix[end:end+1]);b.commit(255999)
        assert b.position==p
        started=time.perf_counter()
        def measure(rows,label):
            nonlocal queries,best,best_u,best_q
            rows=np.asarray(rows,dtype=np.float32);codes=[]
            for a in range(0,len(rows),128):
                codes.append(b.query_vision(rows[a:a+128]))
                assert b.position==p
            codes=np.concatenate(codes).astype(np.float64);queries+=len(rows)
            distance=np.sqrt(np.mean((codes-target)**2,axis=1));i=int(distance.argmin())
            if distance[i]<best:
                best=float(distance[i]);best_u=rows[i].copy();best_q=codes[i].copy()
                np.savez_compressed(out/'best.npz',latent=best_u,kv=best_q)
            np.savez_compressed(out/f'{label}.npz',inputs=rows,kv=codes.astype(np.int8))
            return codes,distance
        first,_=measure(u[None],'initial')
        np.testing.assert_array_equal(first[0],expected)
        initial=best
        trace.append(dict(round=0,rmse=best,queries=queries,flat_rounds=0,seconds=0))
        status='round_budget'
        for iteration in range(1,args.max_rounds+1):
            if best<.125:status='threshold';break
            before=best;center=best_u.copy();center_q=best_q.copy()
            q,_=torch.linalg.qr(torch.randn(768,768,generator=generator,dtype=torch.float64))
            d=(q.T*np.sqrt(768)).float().numpy()
            main=base*(1 if flat==0 else 2 if flat==1 else 4)
            scales=[base/64,base/4,main,4*main];candidates=[];scale_stats=[]
            for si,radius in enumerate(scales):
                probes=np.concatenate([center+radius*d,center-radius*d])
                measured,dist=measure(probes,f'round_{iteration:03d}_scale_{si}_probes')
                candidates.extend(center+updates(d,measured,center_q,target,radius))
                scale_stats.append(dict(probe_rms=radius,best_probe_rmse=float(dist.min()),
                    code_change_fraction=float(np.mean(measured[:768]!=measured[768:]))))
            jitter=rng.standard_normal((128,768)).astype(np.float32)
            jitter/=np.sqrt(np.mean(jitter.astype(np.float64)**2,axis=1,keepdims=True))
            jitter_scale=np.repeat(np.array([base/65536,base/4096,base/256,base/16],np.float32),32)
            candidates.extend(center+jitter*jitter_scale[:,None]);candidates.append(center)
            _,dist=measure(np.stack(candidates),f'round_{iteration:03d}_updates')
            flat=0 if best<before else flat+1
            row_info=dict(round=iteration,rmse=best,gain=before-best,queries=queries,
                main_radius=main,flat_rounds=flat,seconds=time.perf_counter()-started,scales=scale_stats)
            trace.append(row_info);dump(out/'trace.json',trace)
            np.savez_compressed(out/'state.npz',rng_state=np.array(json.dumps(rng.bit_generator.state)),
                torch_rng_state=generator.get_state().numpy(),best_u=best_u,best_q=best_q)
            dump(out/'progress.json',row_info)
            print('ROUND',iteration,'RMSE',round(best,9),'gain',round(before-best,9),'flat',flat,'queries',queries,flush=True)
            if flat>=args.plateau_rounds:status='operational_plateau';break
        repeated=b.query_vision(np.repeat(best_u[None],4,axis=0)).astype(np.float64)
        queries+=4
        assert np.array_equal(repeated,np.repeat(best_q[None],4,axis=0))
        result=dict(status=status,initial_rmse=initial,best_rmse=best,
            improvement_fraction=(initial-best)/initial,rounds=len(trace)-1,
            final_flat_rounds=flat,queries=queries,seconds=time.perf_counter()-started,
            best_single_candidate_rmse=float(np.sqrt(np.mean((repeated[0]-target)**2))),
            repeat_four_exact=True,passes=best<.125,
            scope='operational plateau of this strategy, not proof of global minimum; true prefix')
        dump(out/'result.json',result);print('COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
