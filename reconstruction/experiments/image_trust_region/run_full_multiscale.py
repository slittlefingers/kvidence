"""Full-basis multiscale secants and broad, actually measured line search."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from kvrecon.common import EXPERIMENTS,code_values,dump,digest,validate_quantization_grid
from kvrecon.npu import NpuBackend
from kvrecon.cli import private_guard
from kvrecon.media.image_geometry import orthogonal_directions,centered_steps
from kvrecon.media.image_checkpoint import restore_state,save_state
from kvrecon.media.image_expansion import ExpansionPolicy
SCALE_FACTORS=(.25,1.,4.)
STEP_FACTORS=(1/64,1/32,1/16,1/8,.25,.5,1.,2.,4.)


def main():
    p=argparse.ArgumentParser();p.add_argument('--warm-run',type=Path,required=True)
    p.add_argument('--observation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=5);p.add_argument('--serial',default='YOUR_DEVICE_SERIAL')
    p.add_argument('--adaptive',action='store_true')
    p.add_argument('--plateau-rounds',type=int,default=1)
    p.add_argument('--expansion-factor',type=float,default=2.)
    p.add_argument('--expansion-trial-rounds',type=int,default=1)
    p.add_argument('--max-radius-multiplier',type=float,default=4.)
    p.add_argument('--reset-expansion-policy',action='store_true',help='Apply new policy settings while retaining optimizer/best state')
    p.add_argument('--mlp-test',action='store_true',help='Add 16 trained-MLP and 16 matched linear proposals after standard candidates')
    p.add_argument('--probe-radius',type=float,help='Explicit fixed main probe radius for nonadaptive controlled trials')
    args=p.parse_args()
    if args.rounds<1:raise ValueError('Positive round budget required')
    if args.probe_radius is not None and (args.adaptive or not np.isfinite(args.probe_radius) or args.probe_radius<=0):
        raise ValueError('Probe radius override requires finite positive nonadaptive trial')
    private_guard();meta=json.loads((args.observation/'metadata.json').read_text())
    prior=json.loads((args.warm_run/'protocol.json').read_text());old=json.loads((args.warm_run/'result.json').read_text())
    if prior['observation']!=meta or digest(args.observation/'observed.npz')!=meta['observation_sha256']:
        raise ValueError('Observation mismatch')
    if prior['distance_space']!='lsb' or prior['distance_metric']!='rmse':raise ValueError('Expected LSB RMSE')
    with np.load(args.observation/'observed.npz') as d:target=code_values(d['codes'][0])
    u=np.load(args.warm_run/'best_input.npy');expected=np.load(args.warm_run/'best_kv.npy')
    if u.shape!=(768,):raise ValueError('Expected pre-adapter width 768')
    tail=json.loads((args.warm_run/'optimization.json').read_text())[-1]
    radius=tail.get('next_radius',tail.get('perturbation',prior.get('probe_rms')))
    rate=tail.get('lr',prior.get('lr'));epsilon=prior['epsilon']
    if args.probe_radius is not None:radius=args.probe_radius
    mlp=None;mlp_import_seconds=0.
    if args.mlp_test:
        tick=time.perf_counter()
        import torch
        from kvrecon.media.image_mlp import TinyMLPSurrogate
        from kvrecon.media.image_surrogate import LocalSurrogate
        torch.set_num_threads(1);mlp=TinyMLPSurrogate(steps=100,warm_steps=20)
        mlp_import_seconds=time.perf_counter()-tick
    policy=None
    if args.adaptive:
        state_path=args.warm_run/'expansion_state.json'
        if state_path.exists():
            policy=ExpansionPolicy(**json.loads(state_path.read_text()))
            if args.reset_expansion_policy:
                policy=ExpansionPolicy(policy.initial_radius,args.plateau_rounds,args.expansion_factor,args.expansion_trial_rounds,args.max_radius_multiplier,
                    radius=policy.radius,good_radius=policy.good_radius)
            elif (policy.patience,policy.factor,policy.trial_rounds,policy.max_multiplier)!=(args.plateau_rounds,args.expansion_factor,args.expansion_trial_rounds,args.max_radius_multiplier):
                raise ValueError('Resumed expansion settings differ')
        else:policy=ExpansionPolicy(radius,args.plateau_rounds,args.expansion_factor,args.expansion_trial_rounds,args.max_radius_multiplier)
        radius=policy.radius
    tick=time.perf_counter();rng,m,v,offset,state_source=restore_state(args.warm_run,target,768)
    restore_seconds=time.perf_counter()-tick
    args.output.mkdir(parents=True,exist_ok=False)
    dump(args.output/'protocol.json',dict(optimizer='full_basis_multiscale_measured_line_search',observation=meta,
        warm_run=str(args.warm_run),seed=prior['seed'],rounds=args.rounds,directions=768,geometry=True,
        scale_factors=SCALE_FACTORS,step_factors=STEP_FACTORS,probe_rms=radius,lr=rate,epsilon=epsilon,
        adaptive=args.adaptive,expansion_policy=policy.state() if policy else None,
        reset_expansion_policy=args.reset_expansion_policy,
        mlp_test=args.mlp_test,mlp_import_seconds=mlp_import_seconds,
        mlp_training='42 measured paired directions per scale; 252 rows; 100 first updates, 20 warm updates',
        small_probe_anchor=policy.initial_radius*.25 if policy else radius*.25,
        adam_scale_selection='gradient at scale of measured winning candidate; nominal scale if no improvement',
        adam_offset=offset,restored_state=state_source,restore_seconds=restore_seconds,
        candidates_per_round=3*1536+121+(32 if args.mlp_test else 0),query_budget=1+args.rounds*(3*1536+121+(32 if args.mlp_test else 0)),
        distance_space='lsb',distance_metric='rmse',original_media_accessed=False,original_text_accessed=False,
        source_sha256=digest(Path(__file__)),geometry_source_sha256=digest(ROOT/'kvrecon/media/image_geometry.py'),
        checkpoint_source_sha256=digest(ROOT/'kvrecon/media/image_checkpoint.py'),
        expansion_source_sha256=digest(ROOT/'kvrecon/media/image_expansion.py')))
    if args.mlp_test:
        dump(args.output/'mlp_provenance.json',dict(mlp_source_sha256=digest(ROOT/'kvrecon/media/image_mlp.py'),
            baseline_source_sha256=digest(ROOT/'kvrecon/media/image_surrogate.py'),torch_version=torch.__version__,device='CPU',threads=1))
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    queries=0;oracle=0.;best=float('inf');best_u=None;best_q=None;best_origin=None;trace=[]
    with NpuBackend(sections,meta['layer_index'],args.output/'device',meta['model_sha256'],install=False,serial=args.serial) as b:
        validate_quantization_grid(meta['quantization'],b.quant);b.reset(meta['public_prefix']);started=time.perf_counter()
        def measure(rows,label,metadata):
            nonlocal queries,oracle,best,best_u,best_q,best_origin
            rows=np.asarray(rows,dtype=np.float32);tick=time.perf_counter();chunks=[]
            capacity=min(128,b.batch_capacity)
            for begin in range(0,len(rows),capacity):
                chunk=rows[begin:begin+capacity];q=code_values(b.query_vision(chunk))
                if b.position!=meta['media_start'] or q.shape!=(len(chunk),len(target)) or not np.isfinite(q).all():
                    raise ValueError('Invalid candidate query or advanced prefix')
                chunks.append(q);queries+=len(chunk)
            q=np.concatenate(chunks);oracle+=time.perf_counter()-tick
            distance=np.sqrt(np.mean((q-target)**2,axis=1));j=int(distance.argmin())
            if distance[j]<best:
                best=float(distance[j]);best_u=rows[j].copy();best_q=q[j].copy();best_origin=dict(metadata[j],file=label,index=j)
            np.savez_compressed(args.output/f'{label}.npz',inputs=rows,kv=q)
            dump(args.output/f'{label}.json',[dict(row,rmse=float(d)) for row,d in zip(metadata,distance)])
            return q,distance
        initial,_=measure(u[None],'initial',[dict(kind='incumbent')])
        if not np.array_equal(initial[0],expected):raise ValueError('Initial replay mismatch')
        initial_rmse=best
        for step in range(1,args.rounds+1):
            if best<epsilon:break
            if policy and policy.stopped:break
            if policy:radius=policy.radius
            center=best_u.copy();center_q=best_q.copy();before=best
            delta=orthogonal_directions(rng,768,768);proposals=[];metadata=[];moments={};scale_stats=[];train_x=[];train_y=[]
            sample_rng=np.random.default_rng(217+offset+step)
            scales=(policy.initial_radius*.25/radius,1.,4.) if policy else SCALE_FACTORS
            for scale in scales:
                c=radius*scale;probes=np.concatenate([center+c*delta,center-c*delta])
                probe_metadata=[dict(kind='probe',scale=scale,direction=i%768,sign=1 if i<768 else -1) for i in range(1536)]
                q,dist=measure(probes,f'round_{step:04d}_scale_{scale:g}_probes',probe_metadata)
                if args.mlp_test:
                    ids=sample_rng.choice(768,42,replace=False);ids=np.concatenate([ids,ids+768])
                    train_x.append(probes[ids]);train_y.append(q[ids])
                bases=centered_steps(delta,q,center_q,target,c,multipliers=(1.,))
                for damping_index,direction in enumerate(bases):
                    for factor in STEP_FACTORS:
                        change=direction*factor
                        proposals.append(center+change)
                        metadata.append(dict(kind='ridge',scale=scale,damping=(.001,.01,.1,1.)[damping_index],
                            factor=factor,rms=float(np.sqrt(np.mean(change.astype(float)**2)))))
                loss=.5*np.mean((q-target)**2,axis=1)
                gradient=np.mean(((loss[:768]-loss[768:])/(2*c))[:,None]*delta,axis=0)
                ms=.9*m+.1*gradient;vs=.999*v+.001*gradient**2;moments[scale]=(ms,vs)
                direction=-rate*(ms/(1-.9**(offset+step)))/(np.sqrt(vs/(1-.999**(offset+step)))+1e-8)
                for factor in (.25,.5,1.,2.):
                    change=direction*factor;proposals.append(center+change)
                    metadata.append(dict(kind='adam',scale=scale,factor=factor,rms=float(np.sqrt(np.mean(change**2)))))
                scale_stats.append(dict(scale=scale,probe_rms=c,best_probe_rmse=float(dist.min()),
                    changed_code_fraction=float(np.mean(q[:768]!=q[768:]))))
            proposals.append(center);metadata.append(dict(kind='incumbent'))
            q,dist=measure(proposals,f'round_{step:04d}_updates',metadata)
            mlp_stats={}
            if args.mlp_test:
                standard_best=best
                candidates_array=np.asarray(proposals,dtype=np.float32)
                # Both models optimize the exact same starts, chosen by measured standard-candidate loss.
                order=np.argsort(dist,kind='stable');unique=[];seen=set()
                for i in order:
                    key=candidates_array[i].tobytes()
                    if key not in seen:unique.append(int(i));seen.add(key)
                    if len(unique)==16:break
                while len(unique)<16:unique.append(int(order[0]))
                starts=candidates_array[unique]
                tx=np.concatenate(train_x);ty=np.concatenate(train_y)
                linear=LocalSurrogate(nonlinear=0.).fit(tx,ty,center,center_q,radius)
                tick=time.perf_counter();recovery=None
                try:mlp.fit(tx,ty,center,center_q,radius)
                except (ValueError,RuntimeError) as error:
                    if not any(word in str(error).lower() for word in ('nonfinite','non-finite')):raise
                    recovery=str(error)
                    mlp=TinyMLPSurrogate(steps=100,warm_steps=20,lr=.0003)
                    mlp.fit(tx,ty,center,center_q,radius)
                mlp_fit_seconds=time.perf_counter()-tick
                mlp_stats=dict(linear_fit_seconds=linear.fit_seconds,mlp_fit_seconds=mlp_fit_seconds,
                    mlp_recovery=recovery,mlp_lr=mlp.lr,
                    mlp_train_seconds=mlp.train_seconds,mlp_parameters=mlp.parameter_count,standard_best_rmse=standard_best)
                # Held-out standard updates exclude any exact duplicate training input.
                keep=np.array([not np.any(np.max(np.abs(tx-a),axis=1)<1e-6) for a in candidates_array])
                for name,model in (('linear',linear),('mlp',mlp)):
                    pred=model.predict(candidates_array[keep])
                    mlp_stats[name+'_heldout_rmse']=float(np.sqrt(np.mean((pred-q[keep])**2))) if keep.any() else None
                tick=time.perf_counter();linear_rows=linear.proposals(target,starts,max_rms=8*radius,iterations=12)
                mlp_stats['linear_proposal_seconds']=time.perf_counter()-tick
                tick=time.perf_counter();mlp_rows=mlp.proposals(target,starts,max_rms=8*radius,iterations=12)
                mlp_stats['mlp_proposal_seconds']=time.perf_counter()-tick
                rows=np.concatenate([linear_rows,mlp_rows])
                extra_meta=[dict(kind='linear_surrogate' if i<16 else 'trained_mlp',start_index=unique[i%16],
                    rms=float(np.sqrt(np.mean((rows[i].astype(float)-center)**2)))) for i in range(32)]
                _,extra_dist=measure(rows,f'round_{step:04d}_surrogate_updates',extra_meta)
                mlp_stats.update(linear_proposal_best_rmse=float(extra_dist[:16].min()),mlp_proposal_best_rmse=float(extra_dist[16:].min()))
                np.savez_compressed(args.output/f'round_{step:04d}_mlp_weights.npz',
                    **{k:v.detach().numpy() for k,v in mlp.net.state_dict().items()})
                dump(args.output/f'round_{step:04d}_mlp_training_indices.json',dict(start_indices=unique,train_rows=len(tx)))
            improved=best<before;chosen_scale=best_origin.get('scale',1.) if improved else 1.
            m,v=moments[chosen_scale]
            by_scale=[]
            for scale in scales:
                ids=[i for i,a in enumerate(metadata) if a.get('scale')==scale]
                nominal=[i for i in ids if metadata[i]['kind']=='adam' or metadata[i].get('factor') in (.25,.5,1.,2.)]
                by_scale.append(dict(scale=scale,best_broad_step_rmse=float(dist[ids].min()),
                    best_original_step_subset_rmse=float(dist[nominal].min())))
            event=policy.update(improved) if policy else None
            trace.append(dict(round=step,queries=queries,best_rmse=best,improvement=before-best,
                perturbation=radius,lr=rate,next_radius=policy.radius if policy else radius,
                expansion_event=event,stale_rounds=policy.stale if policy else None,
                mlp=mlp_stats,
                selected=best_origin if improved else dict(kind='incumbent'),scale_probes=scale_stats,
                scale_updates=by_scale,elapsed_seconds=time.perf_counter()-started,oracle_seconds=oracle))
            dump(args.output/'optimization.json',trace);save_state(args.output/'optimizer_state.npz',rng,m,v,offset+step)
            np.save(args.output/'best_input.npy',best_u);np.save(args.output/'best_kv.npy',best_q)
            if policy:dump(args.output/'expansion_state.json',policy.state())
            print(f'Multiscale full round={step} queries={queries} RMSE={best:.6f} radius={radius:.6g} event={event} selected={trace[-1]["selected"]}',flush=True)
        elapsed=time.perf_counter()-started
        np.save(args.output/'best_input.npy',best_u);np.save(args.output/'best_kv.npy',best_q)
        save_state(args.output/'optimizer_state.npz',rng,m,v,offset+len(trace))
        if policy:dump(args.output/'expansion_state.json',policy.state())
        repeats=code_values(b.query_vision(np.repeat(best_u[None],32,axis=0)))
        result=dict(status='threshold' if best<epsilon else (policy.reason if policy and policy.stopped else 'round_budget'),initial_rmse=initial_rmse,best_rmse=best,
            adaptive=args.adaptive,expansion_state=policy.state() if policy else None,
            passes_query=best<epsilon,epsilon=epsilon,queries=queries,rounds=len(trace),optimizer_seconds=elapsed,
            oracle_seconds=oracle,restore_seconds=restore_seconds,total_queries_including_warmup=old['total_queries_including_warmup']+queries,
            total_seconds_including_warmup=old['total_seconds_including_warmup']+elapsed+restore_seconds,
            repeat_exact=bool(np.all(repeats==best_q)),initial_exact_reference=True,diagnostic_candidates=32,
            whole_image_recovered=False,subsequent_text_recovered=False)
        if best<epsilon:
            actual=code_values(b.commit_embeddings(b.project_vision(best_u[None])))[0]
            error=float(np.sqrt(np.mean((actual-target)**2)));result.update(commit_rmse=error,passes_commit=error<epsilon)
        dump(args.output/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
