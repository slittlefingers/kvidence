"""Experimental paired-probe ridge proposals jointly measured with SPSA steps."""
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
from kvrecon.media.image_geometry import orthogonal_directions, centered_steps
from kvrecon.media.image_surrogate import LocalSurrogate
from kvrecon.media.image_checkpoint import restore_state, save_state


def ridge_steps(delta, values, target, c):
    """Secant fit in the sampled directions; no model derivatives required."""
    n=len(delta)
    responses=(values[:n]-values[n:])/(2*c)
    residual=np.mean((values[:n]+values[n:])*.5,axis=0)-target
    gram=np.einsum('ik,jk->ij',responses,responses,optimize=False)
    rhs=-np.einsum('ik,k->i',responses,residual,optimize=False)
    scale=max(float(np.trace(gram)/n),1e-8)
    steps=[]
    for damping in (.001,.01,.1,1.):
        weights=np.linalg.solve(gram+damping*scale*np.eye(n),rhs)
        direction=np.einsum('i,ij->j',weights,delta,optimize=False)
        rms=float(np.sqrt(np.mean(direction**2)))
        if rms>0.5:direction*=.5/rms
        steps.extend([direction*a for a in (.25,.5,1.,2.)])
    return np.asarray(steps,dtype=np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-run',type=Path,required=True)
    p.add_argument('--observation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=300);p.add_argument('--directions',type=int,default=64)
    p.add_argument('--lr',type=float,default=.05);p.add_argument('--perturbation',type=float,default=.1)
    p.add_argument('--center-every',type=int,default=1,help='Compatibility option; joint search measures proposals every round')
    p.add_argument('--warm-start',action='store_true')
    p.add_argument('--geometry',action='store_true',help='Orthogonal probes, measured center residual and probe-radius step cap')
    p.add_argument('--surrogate',action='store_true',help='Replace ridge proposals with fitted tiny-surrogate proposals; requires geometry')
    p.add_argument('--nonlinear',type=float,default=.1,help='Tanh kernel weight; 0 gives a matched linear-only control')
    p.add_argument('--resume-run',type=Path,help='Resume a saved joint/geometry run, counting its preceding cost')
    p.add_argument('--max-queries',type=int,help='Exact candidate cap; schedules follow query-budget progress (300 baseline rounds)')
    p.add_argument('--continue-tail',action='store_true',help='Restore Adam/RNG and hold the preceding run final LR/perturbation')
    p.add_argument('--plateau-rounds',type=int,default=0,help='Stop after this many full rounds without any measured best improvement; 0 disables')
    args=p.parse_args()
    if args.center_every<1 or args.steps<1 or not 1<=args.directions<=768 or args.lr<=0 or args.perturbation<=0:raise ValueError('Invalid configuration')
    if args.max_queries is not None and args.max_queries<25:raise ValueError('Require at least 25 search candidates')
    if args.surrogate and args.directions>64:raise ValueError('Tiny surrogate retains at most 256 probe rows; use <=64 directions')
    if args.surrogate and not args.geometry:raise ValueError('Surrogate requires orthogonal geometry')
    if not np.isfinite(args.nonlinear) or args.nonlinear<0:raise ValueError('Invalid nonlinear weight')
    if args.resume_run and args.warm_start:raise ValueError('Choose one warm-start source')
    if args.plateau_rounds<0:raise ValueError('Invalid plateau window')
    if args.continue_tail and (not args.resume_run or not args.geometry or args.surrogate or args.max_queries):
        raise ValueError('Tail continuation requires resumed geometry, without surrogate or query schedule')
    private_guard()
    meta=json.loads((args.observation/'metadata.json').read_text())
    reference=json.loads((args.reference_run/'protocol.json').read_text())
    if reference['observation']['observation_sha256']!=meta['observation_sha256']:raise ValueError('Different observation')
    if digest(args.observation/'observed.npz')!=meta['observation_sha256']:raise ValueError('Observation integrity mismatch')
    if reference['distance_space']!='lsb' or reference['optimizer']['metric']!='rmse':raise ValueError('Require LSB RMSE reference')
    with np.load(args.observation/'observed.npz') as d:target=code_values(d['codes'][0])
    initial_path=args.reference_run/'recovery/position_0005/initial.npz'
    with np.load(initial_path) as d:u=d['inputs'][0].copy();reference_initial=d['kv'][0].copy()
    warm_seconds=0.;warm_queries=0
    if args.warm_start:
        folder=args.reference_run/'recovery/position_0005'
        u=np.load(folder/'best_input.npy');reference_initial=np.load(folder/'best_kv.npy')
        initial_path=folder/'best_input.npy'
        previous=json.loads((folder/'result.json').read_text())
        warm_seconds=previous['seconds'];warm_queries=previous['queries']
    if args.resume_run:
        prior=json.loads((args.resume_run/'protocol.json').read_text())
        previous=json.loads((args.resume_run/'result.json').read_text())
        if prior['observation']!=meta or prior['seed']!=reference['optimizer']['seed']:
            raise ValueError('Resume observation/seed mismatch')
        initial_path=args.resume_run/'best_input.npy'
        u=np.load(initial_path);reference_initial=np.load(args.resume_run/'best_kv.npy')
        warm_seconds=previous['total_seconds_including_warmup'];warm_queries=previous['total_queries_including_warmup']
    seed=reference['optimizer']['seed'];epsilon=reference['optimizer']['epsilon']
    rng=np.random.default_rng(seed);m=np.zeros_like(u,dtype=float);v=m.copy()
    adam_offset=0;restore_seconds=0.;restored=None;tail=None
    if args.continue_tail:
        tick=time.perf_counter()
        rng,m,v,adam_offset,restored=restore_state(args.resume_run,target,len(u))
        restore_seconds=time.perf_counter()-tick
        tail=json.loads((args.resume_run/'optimization.json').read_text())[-1]
    args.output.mkdir(parents=True,exist_ok=False)
    optimizer=('tiny_tanh_joint' if args.nonlinear>0 else 'tiny_linear_joint') if args.surrogate else ('centered_orthogonal_joint' if args.geometry else 'joint_secant_ridge_spsa')
    dump(args.output/'protocol.json',dict(optimizer=optimizer,seed=seed,steps=args.steps,
        surrogate=args.surrogate,nonlinear=args.nonlinear,hidden=64,surrogate_ridge=.1,
        surrogate_seed=217,surrogate_iterations=12,surrogate_training='current and preceding probe batches, max 256 rows',
        resume_run=str(args.resume_run) if args.resume_run else None,
        continue_tail=args.continue_tail,restored_state=restored,adam_offset=adam_offset,state_restore_seconds=restore_seconds,
        plateau_rounds=args.plateau_rounds,plateau_rule='consecutive full rounds with zero measured best-RMSE improvement; phase-local',
        fixed_tail_lr=tail['lr'] if tail else None,fixed_tail_perturbation=tail['perturbation'] if tail else None,
        geometry=args.geometry,probe_distribution='orthogonal RMS-one' if args.geometry else 'Rademacher',
        ridge_residual='measured center' if args.geometry else 'mean paired midpoint',
        directions=args.directions,lr=args.lr,perturbation=args.perturbation,
        perturbation_schedule='c=.1*(1+(step-1)/100)^(-.101), with .1 configurable',
        lr_schedule='cosine to 0.1 * initial lr',query_budget=args.max_queries or 1+args.steps*(2*args.directions+22),
        schedule_basis='candidate-budget progress; perturbation horizon 300 baseline rounds' if args.max_queries else 'round index',
        probe_batch_capacity=128,final_round='shrink direction count to remaining budget',
        center_every=1,proposal_batch_size=22,ridge_damping=[.001,.01,.1,1.],
        ridge_multipliers=[.25,.5,1.,2.],ridge_step_rms_cap_before_multiplier='current perturbation' if args.geometry else .5,
        selection="best measured incumbent including probes",warm_start=bool(args.warm_start or args.resume_run),warm_start_seconds=warm_seconds,warm_start_queries=warm_queries,
        distance_space='lsb',distance_metric='rmse',epsilon=epsilon,loss='0.5 * MSE',
        initial_sha256=digest(initial_path),reference_run=str(args.reference_run),observation=meta,
        update_projection='none; same unconstrained float input space as trust-region',
        no_model_gradients=True,surrogate_gradients=args.surrogate,original_media_accessed=False,original_text_accessed=False,
        source_sha256=digest(Path(__file__)),geometry_source_sha256=digest(ROOT/'kvrecon/media/image_geometry.py'),
        surrogate_source_sha256=digest(ROOT/'kvrecon/media/image_surrogate.py')))
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    queries=0;oracle_seconds=0.;best=float('inf');best_u=None;best_q=None;trace=[];previous_probes=None;stale=0;plateau=False
    with NpuBackend(sections,meta['layer_index'],args.output/'device',meta['model_sha256'],install=False,serial='YOUR_DEVICE_SERIAL') as b:
        validate_quantization_grid(meta['quantization'],b.quant);b.reset(meta['public_prefix'])
        started=time.perf_counter()
        def measure(rows,label):
            nonlocal queries,oracle_seconds,best,best_u,best_q
            rows=np.asarray(rows,dtype=np.float32);tick=time.perf_counter()
            capacity=min(128,getattr(b,'batch_capacity',128))
            chunks=[]
            for begin in range(0,len(rows),capacity):
                chunk=rows[begin:begin+capacity]
                value=code_values(b.query_vision(chunk))
                if b.position!=meta['media_start'] or value.shape!=(len(chunk),len(target)) or not np.isfinite(value).all():raise ValueError('Invalid candidate chunk replay')
                chunks.append(value)
            values=np.concatenate(chunks);oracle_seconds+=time.perf_counter()-tick
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
            count=args.directions
            progress=(step-1)/args.steps
            c=args.perturbation*(1+(step-1)/100)**(-.101)
            if args.max_queries is not None:
                count=min(count,(args.max_queries-queries-22)//2)
                if count<1:break
                progress=(queries-1)/(args.max_queries-1)
                c=args.perturbation*(1+3*progress)**(-.101)
            if tail:c=tail['perturbation']
            delta=(orthogonal_directions(rng,count,len(u)) if args.geometry else
                (2*rng.integers(0,2,(count,len(u)))-1).astype(np.float32))
            center_q=best_q.copy()
            before=best
            probes=np.concatenate([u+c*delta,u-c*delta])
            losses,_,probe_values=measure(probes,f'round_{step:04d}_probes')
            plus,minus=losses[:count],losses[count:]
            gradient=np.mean(((plus-minus)/(2*c))[:,None]*delta,axis=0)
            if not np.isfinite(gradient).all():raise ValueError('Nonfinite gradient estimate')
            m=.9*m+.1*gradient;v=.999*v+.001*gradient**2
            rate=args.lr*(.1+.9*.5*(1+np.cos(np.pi*progress)))
            if tail:rate=tail['lr']
            adam_step=adam_offset+step
            adam=-rate*(m/(1-.9**adam_step))/(np.sqrt(v/(1-.999**adam_step))+1e-8)
            directions=(centered_steps(delta,probe_values,center_q,target,c) if args.geometry else
                ridge_steps(delta,probe_values,target,c))
            surrogate_info={}
            if args.surrogate:
                tick=time.perf_counter()
                train_x,train_y=probes,probe_values
                if previous_probes is not None:
                    train_x=np.concatenate([previous_probes[0],train_x])
                    train_y=np.concatenate([previous_probes[1],train_y])
                model=LocalSurrogate(hidden=64,nonlinear=args.nonlinear,ridge=.1,seed=217).fit(train_x,train_y,u,center_q,c)
                starts=(u[None]+directions).astype(np.float32)
                proposals=model.proposals(target,starts,max_rms=2*c,iterations=12)
                surrogate_info=dict(surrogate_fit_seconds=model.fit_seconds,
                    surrogate_seconds=time.perf_counter()-tick,surrogate_train_rows=len(train_x))
                directions=proposals-u[None]
                previous_probes=(probes.copy(),probe_values.copy())
            candidates=np.concatenate([u[None]+directions,
                u[None]+np.asarray([.25,.5,1.,2.])[:,None]*adam,
                u[None],best_u[None]]).astype(np.float32)
            _,dist,update_values=measure(candidates,f'round_{step:04d}_update')
            if args.surrogate:
                tick=time.perf_counter()
                prediction=model.predict(candidates[:16])
                surrogate_info['surrogate_seconds']+=time.perf_counter()-tick
                surrogate_info['surrogate_holdout_rmse']=float(np.sqrt(np.mean((prediction-update_values[:16])**2)))
                surrogate_info['surrogate_constant_rmse']=float(np.sqrt(np.mean((center_q-update_values[:16])**2)))
            elif args.geometry:
                weights=np.einsum('ij,kj->ik',candidates[:16].astype(float)-u,delta.astype(float),optimize=False)/len(u)
                responses=(probe_values[:count]-probe_values[count:])/(2*c)
                prediction=center_q+np.einsum('ij,jk->ik',weights,responses,optimize=False)
                surrogate_info['local_prediction_rmse']=float(np.sqrt(np.mean((prediction-update_values[:16])**2)))
                surrogate_info['local_constant_rmse']=float(np.sqrt(np.mean((center_q-update_values[:16])**2)))
            # Advance only to a measured global incumbent, including probe wins.
            u=best_u.copy()
            current=best
            stale=0 if best<before else stale+int(count==args.directions)
            trace.append(dict(round=step,queries=queries,directions=count,elapsed_seconds=time.perf_counter()-started,
                adam_step=adam_step,stale_rounds=stale,improvement=before-best,
                oracle_seconds=oracle_seconds,best_rmse=best,current_rmse=current,
                perturbation=c,lr=rate,gradient_norm=float(np.linalg.norm(gradient)),
                changed_code_fraction=float(np.mean(probe_values[:count]!=probe_values[count:])),
                midpoint_bias_rmse=float(np.sqrt(np.mean((np.mean((probe_values[:count]+probe_values[count:])*.5,axis=0)-center_q)**2))),
                zero_slope_fraction=float(np.mean(plus==minus)),**surrogate_info))
            dump(args.output/'optimization.json',trace)
            save_state(args.output/'optimizer_state.npz',rng,m,v,adam_step)
            np.save(args.output/'best_input.npy',best_u);np.save(args.output/'best_kv.npy',best_q)
            if step%25==0 or count>64:print(f'SPSA round={step} directions={count} queries={queries} best_RMSE={best:.6f} current={current}',flush=True)
            if args.plateau_rounds and stale>=args.plateau_rounds:
                plateau=True;break
        optimization_seconds=time.perf_counter()-started
        np.save(args.output/'best_input.npy',best_u);np.save(args.output/'best_kv.npy',best_q)
        # Repeatability is diagnostic, counted separately from search budget.
        repeats=code_values(b.query_vision(np.repeat(best_u[None],32,axis=0)))
        report=dict(status='threshold' if best<epsilon else ('plateau' if plateau else ('query_budget' if args.max_queries and args.max_queries-queries<24 else 'iteration_limit')),passes_query=best<epsilon,
            plateau_detected=plateau,stale_rounds=stale,plateau_rounds=args.plateau_rounds,state_restore_seconds=restore_seconds,
            initial_rmse=initial_distance,best_rmse=best,queries=queries,rounds=len(trace),
            warm_start=bool(args.warm_start or args.resume_run),total_queries_including_warmup=queries+warm_queries,
            total_seconds_including_warmup=optimization_seconds+warm_seconds+restore_seconds,
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
