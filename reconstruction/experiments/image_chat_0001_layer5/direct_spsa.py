"""Single-position SPSA/Adam over unconstrained post-adapter coordinates."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from spsa_adam import estimate_gradient


def fit_direct(query, target, initial_features, config, output):
    c=config
    initial=np.asarray(initial_features,dtype=np.float32)
    target=np.asarray(target,dtype=np.float64)
    if (initial.shape!=(c.initial_candidates,c.dimension) or not np.isfinite(initial).all()
            or target.ndim!=1 or not np.isfinite(target).all()
            or not 1<=c.directions<=64 or c.steps<1 or c.lr<=0 or c.perturbation<=0
            or c.acceptance_l2<=0):
        raise ValueError('Invalid single-position direct optimization inputs')
    # One scalar change of units, from public random initial candidates only.
    # No projection or normalization is applied to subsequent candidate vectors.
    scale=float(np.sqrt(np.mean(initial.astype(np.float64)**2)))
    if not np.isfinite(scale) or scale<=1e-12:
        raise ValueError('Degenerate initial feature scale')
    out=Path(output);out.mkdir(exist_ok=False)
    rng=np.random.default_rng(c.seed)
    calls=queries=0;oracle_seconds=0.;best=float('inf');best_z=best_kv=None
    started=time.perf_counter()
    def evaluate_features(z):
        nonlocal calls,queries,oracle_seconds,best,best_z,best_kv
        z=np.asarray(z,dtype=np.float32)
        if not np.isfinite(z).all():raise ValueError('Nonfinite proposed features')
        tick=time.perf_counter();kv=np.asarray(query(z),dtype=np.float64)
        oracle_seconds+=time.perf_counter()-tick;calls+=1;queries+=len(z)
        if kv.shape!=(len(z),len(target)) or not np.isfinite(kv).all():
            raise ValueError('Invalid oracle output')
        diff=kv-target;dist=np.linalg.norm(diff,axis=1)
        loss=.5*np.mean(diff*diff,axis=1);i=int(dist.argmin())
        if dist[i]<best:
            best=float(dist[i]);best_z=z[i].copy();best_kv=kv[i].copy()
        return loss,dist
    _,dist=evaluate_features(initial)
    x=initial[int(dist.argmin())].astype(np.float64)/scale
    initial_l2=current=best;m=np.zeros(c.dimension);v=np.zeros(c.dimension)
    def record(step,**extra):
        row=dict(step=step,current_l2=float(current),best_l2=best,queries=queries,
            oracle_calls=calls,oracle_seconds=oracle_seconds,seconds=time.perf_counter()-started,**extra)
        with (out/'trace.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if step%25==0 or best<c.acceptance_l2:
            np.save(out/'best_features.npy',best_z)
            np.save(out/'best_kv.npy',best_kv)
            print(f'DIRECT step={step} current_L2={current:.5f} best_L2={best:.5f} candidates={queries}',flush=True)
    record(0)
    step=0
    for step in range(1,c.steps+1):
        if best<c.acceptance_l2:step-=1;break
        radius=c.perturbation*(1+(step-1)/100)**(-c.perturbation_decay)
        delta=(2*rng.integers(0,2,(c.directions,c.dimension))-1).astype(np.float64)
        loss,_=evaluate_features(scale*np.concatenate([x+radius*delta,x-radius*delta]))
        plus,minus=loss[:c.directions],loss[c.directions:]
        gradient=estimate_gradient(plus,minus,delta,radius)
        m=c.beta1*m+(1-c.beta1)*gradient
        v=c.beta2*v+(1-c.beta2)*gradient*gradient
        rate=c.lr*(c.lr_floor+(1-c.lr_floor)*.5*(1+np.cos(np.pi*(step-1)/c.steps)))
        x-=rate*(m/(1-c.beta1**step))/(np.sqrt(v/(1-c.beta2**step))+c.adam_epsilon)
        _,dist=evaluate_features(scale*x[None]);current=float(dist[0])
        record(step,lr=rate,perturbation_in_scaled_coordinates=radius,
            gradient_norm=float(np.linalg.norm(gradient)),
            zero_slope_fraction=float(np.mean(plus==minus)))
    result=dict(status='threshold' if best<c.acceptance_l2 else 'iteration_limit',
        initial_l2=initial_l2,best_l2=best,current_l2=current,passes_query=best<c.acceptance_l2,
        steps=step,queries=queries,oracle_calls=calls,oracle_seconds=oracle_seconds,
        seconds=time.perf_counter()-started,config=asdict(c),coordinate_scale=scale,
        adapter_used_in_iterations=False,vector_projection_used=False,
        optimization_scope='one position; alternatives share the same fixed recovered prefix',
        exact_gradient_used=False)
    np.save(out/'best_features.npy',best_z);np.save(out/'best_kv.npy',best_kv)
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return best_z,result
