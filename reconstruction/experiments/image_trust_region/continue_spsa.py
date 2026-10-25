"""Continue a recorded SPSA run, reconstructing and verifying its Adam/RNG state."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from kvrecon.common import EXPERIMENTS,code_values,dump,digest,validate_quantization_grid
from kvrecon.cli import private_guard
from kvrecon.npu import NpuBackend


def restore(source,target):
    protocol=json.loads((source/'protocol.json').read_text())
    result=json.loads((source/'result.json').read_text())
    trace=json.loads((source/'optimization.json').read_text())
    rng=np.random.default_rng(protocol['seed'])
    with np.load(source/'initial.npz') as d:u=d['inputs'][0].copy()
    m=np.zeros_like(u,dtype=float);v=m.copy()
    for row in trace:
        step=row['round'];c=row['perturbation']
        delta=(2*rng.integers(0,2,(protocol['directions'],len(u)))-1).astype(np.float32)
        with np.load(source/f'round_{step:04d}_probes.npz') as d:
            np.testing.assert_array_equal(d['inputs'],np.concatenate([u+c*delta,u-c*delta]))
            loss=.5*np.mean((d['kv']-target)**2,axis=1)
        n=protocol['directions']
        g=np.mean(((loss[:n]-loss[n:])/(2*c))[:,None]*delta,axis=0)
        m=.9*m+.1*g;v=.999*v+.001*g*g
        u=(u-row['lr']*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)).astype(np.float32)
        with np.load(source/f'round_{step:04d}_update.npz') as d:
            np.testing.assert_array_equal(u,d['inputs'][0])
    return protocol,result,trace,rng,u,m,v


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--observation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--total-rounds',type=int,default=1000);p.add_argument('--plateau-rounds',type=int,default=100)
    args=p.parse_args();private_guard()
    meta=json.loads((args.observation/'metadata.json').read_text())
    if digest(args.observation/'observed.npz')!=meta['observation_sha256']:raise ValueError('Observation hash mismatch')
    with np.load(args.observation/'observed.npz') as d:target=code_values(d['codes'][0])
    cfg,old,trace,rng,u,m,v=restore(args.source,target)
    if cfg['observation']['observation_sha256']!=meta['observation_sha256']:raise ValueError('Source observation mismatch')
    if args.total_rounds<=len(trace) or args.plateau_rounds<1:raise ValueError('Invalid extension')
    best_u=np.load(args.source/'best_input.npy');best_q=np.load(args.source/'best_kv.npy')
    best=old['best_rmse'];epsilon=cfg['epsilon'];queries=old['queries'];oracle=old['oracle_seconds']
    last_improvement=next(row['round'] for row in trace if row['best_rmse']==best)
    base_seconds=old['optimizer_seconds'];n=cfg['directions'];start_round=len(trace)
    args.output.mkdir(parents=True,exist_ok=False)
    dump(args.output/'protocol.json',dict(**cfg,continuation_source=str(args.source),
        continuation_source_sha256=digest(args.source/'result.json'),total_rounds=args.total_rounds,
        plateau_rounds=args.plateau_rounds,plateau_rule='no strictly smaller measured best RMSE for 100 consecutive rounds',
        lr_after_original_schedule=cfg['lr']*.1,
        state_reconstruction='all prior probe inputs and Adam update inputs verified bit-for-bit',
        continuation_source_code_sha256=digest(Path(__file__))))
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    with NpuBackend(sections,meta['layer_index'],args.output/'device',meta['model_sha256'],install=False,serial='YOUR_DEVICE_SERIAL') as b:
        validate_quantization_grid(meta['quantization'],b.quant);b.reset(meta['public_prefix'])
        with np.load(args.source/f'round_{start_round:04d}_update.npz') as d:current_reference=d['kv'][0].copy()
        verification=code_values(b.query_vision(np.stack([u,best_u])))
        np.testing.assert_array_equal(verification,np.stack([current_reference,best_q]))
        np.savez_compressed(args.output/'resume_state.npz',current=u,m=m,v=v,best_input=best_u,best_kv=best_q)
        started=time.perf_counter();stop='iteration_limit'
        def measure(rows,label):
            nonlocal best,best_u,best_q,queries,oracle,last_improvement
            rows=np.asarray(rows,dtype=np.float32);tick=time.perf_counter()
            values=code_values(b.query_vision(rows));oracle+=time.perf_counter()-tick
            if b.position!=meta['media_start'] or not np.isfinite(values).all():raise ValueError('Invalid replay')
            queries+=len(rows);loss=.5*np.mean((values-target)**2,axis=1);distance=np.sqrt(2*loss)
            j=int(distance.argmin())
            if distance[j]<best:
                best=float(distance[j]);best_u=rows[j].copy();best_q=values[j].copy();last_improvement=step
            np.savez_compressed(args.output/f'{label}.npz',inputs=rows,kv=values)
            return loss,distance
        for step in range(start_round+1,args.total_rounds+1):
            if best<epsilon:stop='threshold';break
            c=cfg['perturbation']*(1+(step-1)/100)**(-.101)
            delta=(2*rng.integers(0,2,(n,len(u)))-1).astype(np.float32)
            loss,_=measure(np.concatenate([u+c*delta,u-c*delta]),f'round_{step:04d}_probes')
            gradient=np.mean(((loss[:n]-loss[n:])/(2*c))[:,None]*delta,axis=0)
            m=.9*m+.1*gradient;v=.999*v+.001*gradient**2
            # Original cosine schedule has finished. Hold its declared floor;
            # do not restart annealing or change the original 300-round path.
            rate=cfg['lr']*.1
            u=(u-rate*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)).astype(np.float32)
            _,current=measure(u[None],f'round_{step:04d}_update')
            trace.append(dict(round=step,queries=queries,elapsed_seconds=base_seconds+time.perf_counter()-started,
                oracle_seconds=oracle,best_rmse=best,current_rmse=float(current[0]),perturbation=c,lr=rate,
                gradient_norm=float(np.linalg.norm(gradient)),zero_slope_fraction=float(np.mean(loss[:n]==loss[n:])),
                rounds_without_improvement=step-last_improvement))
            dump(args.output/'optimization.json',trace)
            if step%25==0:
                np.savez_compressed(args.output/'checkpoint.npz',current=u,m=m,v=v,best_input=best_u,best_kv=best_q)
                dump(args.output/'rng_state.json',rng.bit_generator.state)
                print(f"seed={cfg['seed']} round={step} best_RMSE={best:.6f} idle={step-last_improvement}",flush=True)
            if step-last_improvement>=args.plateau_rounds:stop='empirical_plateau';break
        seconds=base_seconds+time.perf_counter()-started
        np.save(args.output/'best_input.npy',best_u);np.save(args.output/'best_kv.npy',best_q)
        repeat=code_values(b.query_vision(np.repeat(best_u[None],32,axis=0)))
        result=dict(status='threshold' if best<epsilon else stop,seed=cfg['seed'],initial_rmse=old['initial_rmse'],
            initial_extension_rmse=old['best_rmse'],best_rmse=best,passes_query=best<epsilon,epsilon=epsilon,
            rounds=len(trace),queries=queries,last_improvement_round=last_improvement,
            unchanged_tail_rounds=len(trace)-last_improvement,optimizer_seconds=seconds,
            extension_seconds=seconds-base_seconds,oracle_seconds=oracle,repeat_exact=bool(np.all(repeat==best_q)),
            resume_state_verified_exact=True,additional_diagnostic_candidates=34,
            whole_image_recovered=False,subsequent_text_recovered=False)
        if best<epsilon:
            committed=code_values(b.commit_embeddings(b.project_vision(best_u[None])))[0]
            distance=float(np.sqrt(np.mean((committed-target)**2)))
            result.update(commit_rmse=distance,passes_commit=distance<epsilon)
        dump(args.output/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
