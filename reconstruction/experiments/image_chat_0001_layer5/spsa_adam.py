"""Averaged two-sided SPSA + Adam, using only a black-box KV oracle.

Separate experimental optimizer; no changes to the original recovery methods.
"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import time
import numpy as np


@dataclass(frozen=True)
class Config:
    dimension: int = 768
    directions: int = 64
    initial_candidates: int = 128
    steps: int = 1000
    lr: float = .05
    lr_floor: float = .1
    perturbation: float = .1
    perturbation_decay: float = .101
    beta1: float = .9
    beta2: float = .999
    adam_epsilon: float = 1e-8
    acceptance_l2: float = 4.
    seed: int = 17


def unit(x):
    x=np.asarray(x,dtype=np.float64)
    return (x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-8)).astype(np.float32)


def estimate_gradient(plus_loss,minus_loss,directions,c):
    delta=np.asarray(directions,dtype=np.float64)
    slope=(np.asarray(plus_loss)-np.asarray(minus_loss))/(2*c)
    return np.mean(slope[:,None]*delta,axis=0)


def fit(query,target,config,output):
    """query accepts candidate rows, returning actual original-unit KV rows."""
    c=config
    if (c.dimension<1 or not 1<=c.directions<=64 or not 1<=c.initial_candidates<=128
            or c.steps<1 or c.lr<=0 or c.perturbation<=0 or c.acceptance_l2<=0):
        raise ValueError('Invalid SPSA configuration')
    out=Path(output);out.mkdir(exist_ok=False)
    target=np.asarray(target,dtype=np.float64)
    if target.ndim!=1 or not np.isfinite(target).all():raise ValueError('Invalid KV target')
    rng=np.random.default_rng(c.seed);calls=0;queries=0;oracle_seconds=0.
    best=float('inf');best_u=None;best_kv=None;started=time.perf_counter()
    def evaluate(rows):
        nonlocal calls,queries,oracle_seconds,best,best_u,best_kv
        rows=np.asarray(rows,dtype=np.float32)
        tick=time.perf_counter();values=np.asarray(query(rows),dtype=np.float64)
        oracle_seconds+=time.perf_counter()-tick;calls+=1;queries+=len(rows)
        if values.shape!=(len(rows),len(target)) or not np.isfinite(values).all():
            raise ValueError('Invalid oracle output')
        diff=values-target;loss=.5*np.mean(diff*diff,axis=1)
        distances=np.sqrt(np.sum(diff*diff,axis=1));i=int(distances.argmin())
        if distances[i]<best:
            best=float(distances[i]);best_u=rows[i].copy();best_kv=values[i].copy()
        return loss,distances
    initial=unit(rng.standard_normal((c.initial_candidates,c.dimension)))
    _,distances=evaluate(initial);u=initial[int(distances.argmin())].copy()
    initial_l2=best;current_l2=best;m=np.zeros(c.dimension);v=np.zeros(c.dimension)
    trace=[]
    def record(step,**extra):
        row=dict(step=step,current_l2=float(current_l2),best_l2=best,queries=queries,
            oracle_calls=calls,oracle_seconds=oracle_seconds,seconds=time.perf_counter()-started,**extra)
        trace.append(row)
        with (out/'trace.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if step%25==0 or best<c.acceptance_l2:
            print(f'SPSA seed={c.seed} step={step} current_L2={current_l2:.5f} best_L2={best:.5f} candidates={queries}',flush=True)
        if step%25==0 or best<c.acceptance_l2:
            np.save(out/'best_latent.npy',best_u);np.save(out/'best_kv.npy',best_kv)
    record(0)
    step=0
    for step in range(1,c.steps+1):
        if best<c.acceptance_l2:step-=1;break
        radius=c.perturbation*(1+(step-1)/100)**(-c.perturbation_decay)
        delta=(2*rng.integers(0,2,(c.directions,c.dimension))-1).astype(np.float32)
        # Adapter itself normalizes the perturbed inputs. Do not normalize each
        # perturbation here: the estimated function includes the actual adapter.
        probes=np.concatenate([u+radius*delta,u-radius*delta])
        loss,_=evaluate(probes)
        plus,minus=loss[:c.directions],loss[c.directions:]
        g=estimate_gradient(plus,minus,delta,radius)
        if not np.isfinite(g).all():raise ValueError('Nonfinite SPSA gradient')
        m=c.beta1*m+(1-c.beta1)*g;v=c.beta2*v+(1-c.beta2)*g*g
        rate=c.lr*(c.lr_floor+(1-c.lr_floor)*.5*(1+np.cos(np.pi*(step-1)/c.steps)))
        update=rate*(m/(1-c.beta1**step))/(np.sqrt(v/(1-c.beta2**step))+c.adam_epsilon)
        u=unit(u-update)
        _,updated=evaluate(u[None]);current_l2=float(updated[0])
        record(step,lr=rate,perturbation=radius,gradient_norm=float(np.linalg.norm(g)),
            zero_slope_fraction=float(np.mean(plus==minus)),
            mean_absolute_loss_difference=float(np.mean(np.abs(plus-minus))))
    result=dict(status='threshold' if best<c.acceptance_l2 else 'iteration_limit',
        initial_l2=initial_l2,best_l2=best,current_l2=current_l2,passes_query=best<c.acceptance_l2,
        steps=step,queries=queries,oracle_calls=calls,oracle_seconds=oracle_seconds,
        seconds=time.perf_counter()-started,config=asdict(c),
        exact_gradient_used=False,gradient_source='paired phone-forward loss differences',
        note='Threshold matching does not imply unique latent or original image reconstruction')
    np.save(out/'best_latent.npy',best_u);np.save(out/'best_kv.npy',best_kv)
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return best_u,result
