"""Official DFBGN 0.1 against measured Pixel KV, with explicit wall-time budget."""
import argparse
import importlib.metadata
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

class StopSearch(Exception):
    pass

def load_solver():
    # DFBGN 0.1 uses aliases removed in NumPy 1.24. Same builtin types,
    # process-local compatibility only; installed solver source is unchanged.
    if 'int' not in np.__dict__:np.int=int
    if 'float' not in np.__dict__:np.float=float
    import dfbgn
    return dfbgn

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference-run',type=Path,required=True)
    p.add_argument('--observation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warm-run',type=Path)
    p.add_argument('--seconds',type=float,default=173.)
    p.add_argument('--maxfun',type=int,default=45001)
    p.add_argument('--block',type=int,default=64)
    p.add_argument('--rhobeg',type=float,default=None)
    p.add_argument('--rhoend',type=float,default=1e-8)
    p.add_argument('--serial',default='YOUR_DEVICE_SERIAL')
    args=p.parse_args()
    if args.seconds<=0 or args.maxfun<=args.block or not 1<=args.block<=768:raise ValueError('Invalid budget/block')
    private_guard();solver=load_solver()
    meta=json.loads((args.observation/'metadata.json').read_text())
    ref=json.loads((args.reference_run/'protocol.json').read_text())
    if digest(args.observation/'observed.npz')!=meta['observation_sha256'] or ref['observation']!=meta:raise ValueError('Observation mismatch')
    if ref['distance_space']!='lsb' or ref['optimizer']['metric']!='rmse':raise ValueError('Expected LSB RMSE')
    with np.load(args.observation/'observed.npz') as d:target=code_values(d['codes'][0])
    start_path=args.reference_run/'recovery/position_0005/initial.npz'
    with np.load(start_path) as d:u=d['inputs'][0].copy();expected=d['kv'][0].copy()
    warm_seconds=0.;warm_queries=0
    if args.warm_run:
        prev_protocol=json.loads((args.warm_run/'protocol.json').read_text())
        if prev_protocol['observation']!=meta:raise ValueError('Warm-start observation mismatch')
        start_path=args.warm_run/'best_input.npy';u=np.load(start_path);expected=np.load(args.warm_run/'best_kv.npy')
        prev=json.loads((args.warm_run/'result.json').read_text())
        warm_seconds=prev['total_seconds_including_warmup'];warm_queries=prev['total_queries_including_warmup']
    seed=ref['optimizer']['seed'];epsilon=ref['optimizer']['epsilon'];np.random.seed(seed)
    args.output.mkdir(parents=True,exist_ok=False)
    params={'logging.save_diagnostic_info':False}
    package_path=Path(solver.__file__).parent
    dump(args.output/'protocol.json',dict(optimizer='official_dfbgn',package_version=importlib.metadata.version('dfbgn'),
        source_sha256=digest(Path(__file__)),package_source_sha256={f.name:digest(f) for f in package_path.glob('*.py')},
        numpy_compatibility=['np.int=int','np.float=float'],numpy_version=np.__version__,seed=seed,
        observation=meta,distance_space='lsb',distance_metric='rmse',epsilon=epsilon,
        residual='(measured integer KV codes - target codes) / sqrt(1024); sum of squares is MSE',
        seconds=args.seconds,maxfun=args.maxfun,block=args.block,rhobeg=args.rhobeg,
        effective_rhobeg=args.rhobeg if args.rhobeg is not None else .1*max(float(np.max(np.abs(u))),1.),
        rhoend=args.rhoend,user_params=params,objfun_has_noise=False,
        diagnostics_note='No expensive solver diagnostics; every physical query logged separately',
        batch_size=1,initial_sha256=digest(start_path),reference_run=str(args.reference_run),
        warm_run=str(args.warm_run) if args.warm_run else None,warm_seconds=warm_seconds,warm_queries=warm_queries,
        original_media_accessed=False,original_text_accessed=False))
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    queries=0;oracle_seconds=0.;best=float('inf');best_u=None;best_q=None;initial=None;status=None;solution=None
    with NpuBackend(sections,meta['layer_index'],args.output/'device',meta['model_sha256'],install=False,serial=args.serial) as b:
        validate_quantization_grid(meta['quantization'],b.quant);b.reset(meta['public_prefix'])
        started=time.perf_counter()
        with (args.output/'optimization.jsonl').open('w') as trace:
            def objective(x):
                nonlocal queries,oracle_seconds,best,best_u,best_q,initial
                if queries and best<epsilon:raise StopSearch('threshold')
                if queries and time.perf_counter()-started>=args.seconds:raise StopSearch('wall_time_budget')
                rows=np.asarray(x,dtype=np.float32).reshape(1,-1)
                if not np.isfinite(rows).all():raise ValueError('Nonfinite solver proposal')
                tick=time.perf_counter();q=code_values(b.query_vision(rows));oracle_seconds+=time.perf_counter()-tick
                if b.position!=meta['media_start'] or q.shape!=(1,len(target)) or not np.isfinite(q).all():raise ValueError('Invalid replay')
                queries+=1;r=(q[0]-target)/np.sqrt(len(target));rmse=float(np.linalg.norm(r))
                if queries==1:
                    if not np.array_equal(q[0],expected):raise ValueError('Initial replay differs')
                    initial=rmse
                if rmse<best:
                    best=rmse;best_u=rows[0].copy();best_q=q[0].copy()
                    np.save(args.output/'best_input.npy',best_u);np.save(args.output/'best_kv.npy',best_q)
                np.savez_compressed(args.output/f'query_{queries:05d}.npz',inputs=rows,kv=q)
                item=dict(queries=queries,elapsed_seconds=time.perf_counter()-started,oracle_seconds=oracle_seconds,
                    current_rmse=rmse,best_rmse=best)
                trace.write(json.dumps(item)+'\n');trace.flush()
                if queries%50==0:print(f'DFBGN query={queries} sec={item["elapsed_seconds"]:.1f} best_RMSE={best:.6f}',flush=True)
                return r
            try:
                solution=solver.solve(objective,u.astype(float),fixed_block=args.block,rhobeg=args.rhobeg,
                    rhoend=args.rhoend,maxfun=args.maxfun,user_params=params,objfun_has_noise=False)
                status='solver_exit'
            except StopSearch as exc:status=str(exc)
        elapsed=time.perf_counter()-started
        repeats=code_values(b.query_vision(np.repeat(best_u[None],32,axis=0)))
        report=dict(status=status,passes_query=best<epsilon,initial_rmse=initial,best_rmse=best,
            queries=queries,optimizer_seconds=elapsed,oracle_seconds=oracle_seconds,
            total_queries_including_warmup=queries+warm_queries,total_seconds_including_warmup=elapsed+warm_seconds,
            seed=seed,epsilon=epsilon,initial_exact_reference=True,repeat_exact=bool(np.all(repeats==best_q)),
            diagnostic_candidates=32,whole_image_recovered=False,subsequent_text_recovered=False)
        if solution is not None:report.update(solver_flag=int(solution.flag),solver_msg=solution.msg,solver_nf=int(solution.nf))
        if best<epsilon:
            actual=code_values(b.commit_embeddings(b.project_vision(best_u[None])))[0]
            error=float(np.sqrt(np.mean((actual-target)**2)))
            report.update(commit_rmse=error,passes_commit=error<epsilon)
        dump(args.output/'result.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
