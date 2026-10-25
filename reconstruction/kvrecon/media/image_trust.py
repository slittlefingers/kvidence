"""Query-only local ridge/trust-region optimizer for ONE causal image slot."""
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import numpy as np
from ..common import dump
from ..search import kv_distance


@dataclass(frozen=True)
class TrustConfig:
    directions: int = 16
    max_queries: int = 2048
    radius: float = 2.0
    min_radius: float = .01
    max_radius: float = 16.0
    ridge: float = .01
    damping: float = .001
    seed: int = 29
    epsilon: float = .125
    metric: str = 'rmse'
    batch_size: int = 128

    def validate(self, width):
        if any(type(v) is not int or v < 1 for v in (self.directions,self.max_queries,self.batch_size)):
            raise ValueError('Positive integer directions, budget and batch size required')
        if self.directions > width or self.max_queries < 4:
            raise ValueError('Too many directions or insufficient query budget')
        if not all(np.isfinite(v) and v > 0 for v in (self.radius,self.min_radius,self.max_radius,self.ridge,self.damping,self.epsilon)):
            raise ValueError('Finite positive trust-region parameters required')
        if not self.min_radius <= self.radius <= self.max_radius or self.metric not in ('rmse','l2'):
            raise ValueError('Invalid radius bounds or metric')


def product(left, right):
    # Small local systems: explicit contractions avoid Accelerate/BLAS floating
    # status warnings on this host, without suppressing numerical errors.
    signature=('i,ij->j' if left.ndim==1 else 'ij,jk->ik') if right.ndim==2 else 'ij,j->i'
    value=np.einsum(signature,left,right,optimize=False)
    if not np.isfinite(value).all():raise ValueError('Nonfinite local model product')
    return value


def trust_step(mapping, residual, radius, damping):
    """Solve ||B u-r||² + mu||u||² subject to ||u|| <= radius."""
    h=product(mapping.T,mapping)
    eigen, vectors=np.linalg.eigh(h)
    eigen=np.maximum(eigen,0.)
    rhs=product(vectors.T,product(mapping.T,residual))
    def solve(extra):return product(vectors,rhs/(eigen+damping+extra))
    step=solve(0.)
    if np.linalg.norm(step)<=radius:return step
    low,high=0.,1.
    while np.linalg.norm(solve(high))>radius:high*=2
    for _ in range(60):
        mid=(low+high)/2
        if np.linalg.norm(solve(mid))>radius:low=mid
        else:high=mid
    return solve(high)


def fit_local(query, target, initial, config, output):
    """query(rows) returns actual KV in declared units, with NO prefix advance.

    Operates on float input coordinates; no unjustified INT8 input assumption.
    Each round rotates a random subspace and includes the previous useful step.
    All actual queried rows/KV are persisted, never surrogate-accepted.
    """
    z=np.asarray(initial,dtype=np.float32).copy()
    target=np.asarray(target,dtype=np.float64)
    if z.ndim!=1 or target.ndim!=1 or not np.isfinite(z).all() or not np.isfinite(target).all():
        raise ValueError('Expected finite one-position vectors')
    config.validate(len(z))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(config.seed);queries=0;round_index=0;last_step=None
    started=time.perf_counter();radius=config.radius
    def measure(rows, label):
        nonlocal queries
        rows=np.asarray(rows,dtype=np.float32)
        if queries+len(rows)>config.max_queries:raise ValueError('Query budget exceeded')
        chunks=[]
        for begin in range(0,len(rows),config.batch_size):
            chunk=rows[begin:begin+config.batch_size]
            values=np.asarray(query(chunk),dtype=np.float64)
            if values.shape!=(len(chunk),len(target)) or not np.isfinite(values).all():
                raise ValueError('Invalid measured candidate KV')
            chunks.append(values);queries+=len(chunk)
        values=np.concatenate(chunks)
        np.savez_compressed(output/f'{label}.npz',inputs=rows,kv=values)
        return values
    q=measure(z[None],'initial')[0]
    loss=float(np.mean((q-target)**2));trace=[]
    while float(kv_distance(q,target,config.metric))>=config.epsilon:
        remaining=config.max_queries-queries
        if remaining<3:break
        k=min(config.directions,(remaining-1)//2)
        raw=rng.standard_normal((len(z),k))
        if last_step is not None:raw[:,0]=last_step
        basis=np.linalg.qr(raw,mode='reduced')[0]
        coordinates=np.concatenate([np.eye(k),-np.eye(k)])*radius
        rows=(z[None]+product(coordinates,basis.T)).astype(np.float32)
        actual_coordinates=product(rows.astype(np.float64)-z,basis)
        measured=measure(rows,f'round_{round_index:04d}_probes')
        # Ridge fit of changes; scale coordinates so lambda remains meaningful
        # as the radius changes. B maps subspace displacement to KV change.
        x=actual_coordinates/radius
        coefficients=np.linalg.solve(product(x.T,x)+config.ridge*len(x)*np.eye(k),product(x.T,measured-q))
        mapping=coefficients.T/radius
        step=trust_step(mapping,target-q,radius,config.damping)
        trial=(z+product(basis,step)).astype(np.float32)
        effective_step=product(trial.astype(np.float64)-z,basis)
        predicted=q+product(mapping,effective_step)
        predicted_gain=loss-float(np.mean((predicted-target)**2))
        trial_q=measure(trial[None],f'round_{round_index:04d}_trial')[0]
        trial_loss=float(np.mean((trial_q-target)**2))
        actual_gain=loss-trial_loss
        ratio=actual_gain/predicted_gain if predicted_gain>1e-15 else None
        old_radius=radius
        if ratio is None or ratio<.25:radius=max(config.min_radius,radius*.5)
        elif ratio>.75 and np.linalg.norm(step)>.8*old_radius:radius=min(config.max_radius,radius*2)
        # All measured probes are valid alternatives, including when the local
        # model failed; retain the actual best, not the predicted best.
        all_rows=np.concatenate([rows,trial[None]])
        all_values=np.concatenate([measured,trial_q[None]])
        losses=np.mean((all_values-target)**2,axis=1);best=int(np.argmin(losses))
        improved=bool(losses[best]<loss)
        if improved:
            last_step=all_rows[best].astype(np.float64)-z
            z=all_rows[best].copy();q=all_values[best].copy();loss=float(losses[best])
        trace.append(dict(round=round_index,queries=queries,directions=k,radius=old_radius,
            next_radius=radius,predicted_gain=predicted_gain,actual_gain=actual_gain,ratio=ratio,
            improved=improved,selected='trial' if best==len(rows) else 'probe',
            distance=float(kv_distance(q,target,config.metric)),mse=loss,
            elapsed_seconds=time.perf_counter()-started))
        dump(output/'optimization.json',trace)
        round_index+=1
    distance=float(kv_distance(q,target,config.metric))
    np.save(output/'best_input.npy',z);np.save(output/'best_kv.npy',q)
    report=dict(status='threshold' if distance<config.epsilon else 'query_budget',
        passes_query=distance<config.epsilon,distance=distance,rmse=float(kv_distance(q,target)),
        l2=float(kv_distance(q,target,'l2')),queries=queries,rounds=round_index,
        config=asdict(config),seconds=time.perf_counter()-started,
        optimizer='local_ridge_trust_region',original_representation_recovered=False)
    dump(output/'result.json',report)
    return z,report
