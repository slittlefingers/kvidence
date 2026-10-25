"""Matched-budget averaged SPSA gradient + Adam; real phone KV, one image slot."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from kvrecon.common import EXPERIMENTS,code_values,dump,digest,validate_quantization_grid
from kvrecon.npu import NpuBackend
from kvrecon.cli import private_guard


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-run',type=Path,required=True)
    p.add_argument('--observation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=300);p.add_argument('--directions',type=int,default=16)
    p.add_argument('--lr',type=float,default=.05);p.add_argument('--perturbation',type=float,default=.1)
    args=p.parse_args()
    if args.steps<1 or not 1<=args.directions<=64 or args.lr<=0 or args.perturbation<=0:raise ValueError('Invalid configuration')
    private_guard()
    meta=json.loads((args.observation/'metadata.json').read_text())
    reference=json.loads((args.reference_run/'protocol.json').read_text())
    if reference['observation']['observation_sha256']!=meta['observation_sha256']:raise ValueError('Different observation')
    if digest(args.observation/'observed.npz')!=meta['observation_sha256']:raise ValueError('Observation integrity mismatch')
    if reference['distance_space']!='lsb' or reference['optimizer']['metric']!='rmse':raise ValueError('Require LSB RMSE reference')
    with np.load(args.observation/'observed.npz') as d:target=code_values(d['codes'][0])
    initial_path=args.reference_run/'recovery/position_0005/initial.npz'
    with np.load(initial_path) as d:u=d['inputs'][0].copy();reference_initial=d['kv'][0].copy()
    seed=reference['optimizer']['seed'];epsilon=reference['optimizer']['epsilon']
    rng=np.random.default_rng(seed);m=np.zeros_like(u,dtype=float);v=m.copy()
    args.output.mkdir(parents=True,exist_ok=False)
    dump(args.output/'protocol.json',dict(optimizer='averaged_spsa_adam',seed=seed,steps=args.steps,
        directions=args.directions,lr=args.lr,perturbation=args.perturbation,
        perturbation_schedule='c=.1*(1+(step-1)/100)^(-.101), with .1 configurable',
        lr_schedule='cosine to 0.1 * initial lr',query_budget=1+args.steps*(2*args.directions+1),
        distance_space='lsb',distance_metric='rmse',epsilon=epsilon,loss='0.5 * MSE',
        initial_sha256=digest(initial_path),reference_run=str(args.reference_run),observation=meta,
        update_projection='none; same unconstrained float input space as trust-region',
        no_model_gradients=True,original_media_accessed=False,original_text_accessed=False,
        source_sha256=digest(Path(__file__))))
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    queries=0;oracle_seconds=0.;best=float('inf');best_u=None;best_q=None;trace=[]
    with NpuBackend(sections,meta['layer_index'],args.output/'device',meta['model_sha256'],install=False,serial='YOUR_DEVICE_SERIAL') as b:
        validate_quantization_grid(meta['quantization'],b.quant);b.reset(meta['public_prefix'])
        started=time.perf_counter()
        def measure(rows,label):
            nonlocal queries,oracle_seconds,best,best_u,best_q
            rows=np.asarray(rows,dtype=np.float32);tick=time.perf_counter()
            values=code_values(b.query_vision(rows));oracle_seconds+=time.perf_counter()-tick
            if b.position!=meta['media_start'] or values.shape!=(len(rows),len(target)) or not np.isfinite(values).all():raise ValueError('Invalid candidate replay')
            queries+=len(rows);loss=.5*np.mean((values-target)**2,axis=1);dist=np.sqrt(2*loss)
            j=int(dist.argmin())
            if dist[j]<best:best=float(dist[j]);best_u=rows[j].copy();best_q=values[j].copy()
            np.savez_compressed(args.output/f'{label}.npz',inputs=rows,kv=values)
            return loss,dist,values
        _,_,initial=measure(u[None],'initial')
        initial_exact=bool(np.array_equal(initial[0],reference_initial));initial_distance=best
        if not initial_exact:raise ValueError('Initial candidate differs from reference replay')
        for step in range(1,args.steps+1):
            if best<epsilon:break
            c=args.perturbation*(1+(step-1)/100)**(-.101)
            delta=(2*rng.integers(0,2,(args.directions,len(u)))-1).astype(np.float32)
            probes=np.concatenate([u+c*delta,u-c*delta])
            losses,_,_=measure(probes,f'round_{step:04d}_probes')
            plus,minus=losses[:args.directions],losses[args.directions:]
            gradient=np.mean(((plus-minus)/(2*c))[:,None]*delta,axis=0)
            if not np.isfinite(gradient).all():raise ValueError('Nonfinite gradient estimate')
            m=.9*m+.1*gradient;v=.999*v+.001*gradient**2
            rate=args.lr*(.1+.9*.5*(1+np.cos(np.pi*(step-1)/args.steps)))
            u=(u-rate*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)).astype(np.float32)
            _,current,_=measure(u[None],f'round_{step:04d}_update')
            trace.append(dict(round=step,queries=queries,elapsed_seconds=time.perf_counter()-started,
                oracle_seconds=oracle_seconds,best_rmse=best,current_rmse=float(current[0]),
                perturbation=c,lr=rate,gradient_norm=float(np.linalg.norm(gradient)),
                zero_slope_fraction=float(np.mean(plus==minus))))
            dump(args.output/'optimization.json',trace)
            if step%25==0:print(f'SPSA round={step} queries={queries} best_RMSE={best:.6f} current={current[0]:.6f}',flush=True)
        optimization_seconds=time.perf_counter()-started
        np.save(args.output/'best_input.npy',best_u);np.save(args.output/'best_kv.npy',best_q)
        # Repeatability is diagnostic, counted separately from search budget.
        repeats=code_values(b.query_vision(np.repeat(best_u[None],32,axis=0)))
        report=dict(status='threshold' if best<epsilon else 'iteration_limit',passes_query=best<epsilon,
            initial_rmse=initial_distance,best_rmse=best,queries=queries,rounds=len(trace),
            optimizer_seconds=optimization_seconds,oracle_seconds=oracle_seconds,
            diagnostic_candidates=32,repeat_exact=bool(np.all(repeats==best_q)),
            seed=seed,initial_exact_reference=initial_exact,epsilon=epsilon,
            whole_image_recovered=False,subsequent_text_recovered=False)
        if best<epsilon:
            feature=b.project_vision(best_u[None]);actual=code_values(b.commit_embeddings(feature))[0]
            commit_rmse=float(np.sqrt(np.mean((actual-target)**2)))
            report.update(commit_rmse=commit_rmse,passes_commit=commit_rmse<epsilon)
        dump(args.output/'result.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
