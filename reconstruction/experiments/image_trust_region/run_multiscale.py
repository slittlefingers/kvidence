"""Measured multiscale secant/SPSA refinement with bounded local restarts."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from kvrecon.common import EXPERIMENTS,code_values,dump,digest,validate_quantization_grid
from kvrecon.npu import NpuBackend
from kvrecon.cli import private_guard

SCALES=np.array([.0001,.001,.01,.1])

def proposals(center,center_q,delta,values,target):
    """Fit each scale separately; use measured center residual, not probe mean."""
    result=[]
    for group,c in enumerate(SCALES):
        start=group*16;dirs=delta[start:start+16]
        plus=values[start:start+16];minus=values[64+start:64+start+16]
        a=(plus-minus)/(2*c);r=center_q-target
        gram=np.einsum('ik,jk->ij',a,a,optimize=False)
        rhs=-np.einsum('ik,k->i',a,r,optimize=False)
        scale=max(float(np.trace(gram)/16),1e-12)
        for reg in (.001,.01,.1,1.):
            w=np.linalg.solve(gram+reg*scale*np.eye(16),rhs)
            step=np.einsum('i,ij->j',w,dirs,optimize=False)
            rms=float(np.sqrt(np.mean(step**2)))
            if rms>.5:step*=.5/rms
            for factor in (.0625,.25,1.,2.):result.append(center+factor*step)
        lossplus=.5*np.mean((plus-target)**2,axis=1);lossminus=.5*np.mean((minus-target)**2,axis=1)
        g=np.einsum('i,ij->j',(lossplus-lossminus)/(2*c),dirs,optimize=False)/16
        g/=max(float(np.sqrt(np.mean(g*g))),1e-12)
        for factor in (.1,.3,1.,3.):result.append(center-c*factor*g)
    return np.asarray(result,dtype=np.float32)

def main():
    p=argparse.ArgumentParser();p.add_argument('--warm-run',type=Path,required=True)
    p.add_argument('--observation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=300);p.add_argument('--rounds',type=int,default=1000)
    p.add_argument('--epsilon',type=float,default=.1);p.add_argument('--seed',type=int,default=117)
    args=p.parse_args();private_guard()
    meta=json.loads((args.observation/'metadata.json').read_text());old=json.loads((args.warm_run/'protocol.json').read_text())
    if old['observation']!=meta or digest(args.observation/'observed.npz')!=meta['observation_sha256']:raise ValueError('Observation mismatch')
    u=np.load(args.warm_run/'best_input.npy');expected=np.load(args.warm_run/'best_kv.npy')
    prior=json.loads((args.warm_run/'result.json').read_text())
    with np.load(args.observation/'observed.npz') as f:target=code_values(f['codes'][0])
    args.output.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(args.seed)
    dump(args.output/'protocol.json',dict(optimizer='multiscale_centered_secant_spsa_restarts',observation=meta,
        seed=args.seed,scales=SCALES.tolist(),directions_per_scale=16,seconds=args.seconds,rounds=args.rounds,
        epsilon=args.epsilon,acceptance='measured LSB RMSE <= epsilon, plus commit validation',distance_space='lsb',distance_metric='rmse',
        warm_run=str(args.warm_run),warm_seconds=prior['total_seconds_including_warmup'],warm_queries=prior['total_queries_including_warmup'],
        source_sha256=digest(Path(__file__)),restart_after_nonimproving_rounds=20,
        center_response='exact measured center KV',proposal_batch=128,probe_batch=128,
        original_media_accessed=False,original_text_accessed=False))
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    best=float('inf');best_u=None;best_q=None;queries=0;oracle=0.;trace=[];restarts=0;stale=0
    with NpuBackend(sections,meta['layer_index'],args.output/'device',meta['model_sha256'],install=False,serial='YOUR_DEVICE_SERIAL') as b:
        validate_quantization_grid(meta['quantization'],b.quant);b.reset(meta['public_prefix']);started=time.perf_counter()
        def measure(rows,label):
            nonlocal best,best_u,best_q,queries,oracle
            rows=np.asarray(rows,dtype=np.float32);tick=time.perf_counter();q=code_values(b.query_vision(rows));oracle+=time.perf_counter()-tick
            if b.position!=meta['media_start'] or q.shape!=(len(rows),len(target)) or not np.isfinite(q).all():raise ValueError('Invalid replay')
            d=np.sqrt(np.mean((q-target)**2,axis=1));j=int(d.argmin());queries+=len(rows)
            if d[j]<best:
                best=float(d[j]);best_u=rows[j].copy();best_q=q[j].copy()
                np.save(args.output/'best_input.npy',best_u);np.save(args.output/'best_kv.npy',best_q)
            np.savez_compressed(args.output/f'{label}.npz',inputs=rows,kv=q)
            return q,d
        q,d=measure(u[None],'initial');uq=q[0];current=float(d[0]);initial=current
        if not np.array_equal(uq,expected):raise ValueError('Initial replay mismatch')
        status='round_limit'
        for step in range(1,args.rounds+1):
            if best<=args.epsilon:status='threshold';break
            if time.perf_counter()-started>=args.seconds:status='time_budget';break
            before=best
            delta=(2*rng.integers(0,2,(64,len(u)))-1).astype(np.float32)
            scale=np.repeat(SCALES,16)[:,None]
            probes=np.concatenate([u+scale*delta,u-scale*delta]).astype(np.float32)
            probe_q,probe_d=measure(probes,f'round_{step:04d}_probes')
            trial=proposals(u,uq,delta,probe_q,target)
            # 80 directed proposals + 46 random local explorations + two incumbents.
            jitters=rng.standard_normal((46,len(u))).astype(np.float32)
            jitters/=np.sqrt(np.mean(jitters**2,axis=1,keepdims=True))
            radii=np.resize(SCALES,46)[:,None]
            trial=np.concatenate([trial,u+radii*jitters,u[None],best_u[None]]).astype(np.float32)
            tq,td=measure(trial,f'round_{step:04d}_trials')
            j=int(td[:127].argmin());k=int(probe_d.argmin())
            if probe_d[k]<td[j]:u=probes[k].copy();uq=probe_q[k].copy();current=float(probe_d[k])
            else:u=trial[j].copy();uq=tq[j].copy();current=float(td[j])
            stale=0 if best<before else stale+1;restarted=False
            if stale>=20 and best>args.epsilon:
                # Start a measured, slightly worse branch. Keep global incumbent
                # separately; omit it from branch selection on subsequent rounds.
                choices=np.arange(80,126);eligible=choices[(td[choices]>best)&(td[choices]<=best*1.25)]
                if len(eligible):
                    j=int(rng.choice(eligible));u=trial[j].copy();uq=tq[j].copy();current=float(td[j]);restarts+=1;restarted=True
                stale=0
            trace.append(dict(round=step,queries=queries,elapsed_seconds=time.perf_counter()-started,
                oracle_seconds=oracle,best_rmse=best,current_rmse=current,restarted=restarted,restarts=restarts))
            dump(args.output/'optimization.json',trace)
            if step%25==0:print(f'Multiscale round={step} best={best:.6f} current={current:.6f} sec={trace[-1]["elapsed_seconds"]:.1f}',flush=True)
        elapsed=time.perf_counter()-started
        repeat=code_values(b.query_vision(np.repeat(best_u[None],32,axis=0)))
        report=dict(status=status,initial_rmse=initial,best_rmse=best,passes_query=best<=args.epsilon,epsilon=args.epsilon,
            queries=queries,rounds=len(trace),optimizer_seconds=elapsed,oracle_seconds=oracle,restarts=restarts,
            total_queries_including_warmup=queries+prior['total_queries_including_warmup'],
            total_seconds_including_warmup=elapsed+prior['total_seconds_including_warmup'],
            repeat_exact=bool(np.all(repeat==best_q)),diagnostic_candidates=32,whole_image_recovered=False,subsequent_text_recovered=False)
        if best<=args.epsilon:
            actual=code_values(b.commit_embeddings(b.project_vision(best_u[None])))[0]
            error=float(np.sqrt(np.mean((actual-target)**2)));report.update(commit_rmse=error,passes_commit=error<=args.epsilon)
        dump(args.output/'result.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
