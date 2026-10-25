"""Discrete search. Proposals never satisfy acceptance without a real KV query."""
from dataclasses import dataclass, asdict
import time
import numpy as np
from .common import code_values, resolve_distance_space


@dataclass(frozen=True)
class SearchConfig:
    method: str = "guided"
    epsilon: float = 0.125
    batch_size: int = 128
    probability_seeds: int = 50
    random_seeds: int = 150
    proposals: int = 20
    seed: int = 20261019
    max_queries: int | None = None
    kv_thresholds: tuple[float, float] | None = None
    k_width: int | None = None
    stop_partition: int = 1
    distance_space: str = "auto"
    distance_metric: str = "rmse"

    @property
    def metric(self):
        return "l2" if self.method == "collision_plus_cpa" else self.distance_metric

    def space_for(self, dtype):
        # Official CPA retains its calibrated original-value metric.
        return "original" if self.method == "collision_plus_cpa" else resolve_distance_space(self.distance_space, dtype)

    def validate(self):
        if self.distance_metric not in ("rmse", "l2"):
            raise ValueError("Unknown KV distance metric")
        if self.distance_space not in ("auto", "lsb", "original"):
            raise ValueError("Unknown KV distance space")
        if self.method not in ("guided","collision_plus","collision_plus_cpa","exhaustive"):
            raise ValueError("Unknown search method")
        if not np.isfinite(self.epsilon) or self.epsilon<=0 or self.batch_size<=0:
            raise ValueError("Invalid epsilon or batch size")
        if self.method=='guided' and (min(self.probability_seeds,self.random_seeds)<0 or self.probability_seeds+self.random_seeds<1 or self.proposals<=0):
            raise ValueError("Seed counts must be nonnegative with a positive total; proposals must be positive")
        if self.method=="guided" and self.probability_seeds+self.random_seeds<2:
            raise ValueError("Guided ridge search requires at least two requested initial candidates")
        if self.max_queries is not None and self.max_queries<=0:
            raise ValueError("An explicit resource budget must be positive")
        if self.stop_partition < 1:
            raise ValueError("stop_partition must be positive")
        if self.method=='collision_plus_cpa':
            if self.kv_thresholds is None or len(self.kv_thresholds)!=2 or not np.isfinite(self.kv_thresholds).all() or min(self.kv_thresholds)<0 or not self.k_width or self.k_width<1:
                raise ValueError("CPA requires calibrated K/V thresholds and K width")
        elif self.stop_partition!=1:
            raise ValueError("Probability pruning is only implemented for CPA")
        return self


def measured_values(backend, codes, config):
    return code_values(codes) if config.space_for(np.asarray(codes).dtype) == "lsb" else backend.values(codes)


def kv_distance(values, target, metric="rmse"):
    """Widen before subtraction; RMSE normalises by retained coordinate count."""
    delta=np.asarray(values,dtype=np.float64)-np.asarray(target,dtype=np.float64)
    if delta.shape[-1] == 0:raise ValueError("Empty KV vector")
    if metric not in ("rmse","l2"):raise ValueError("Unknown KV distance metric")
    distance=np.linalg.norm(delta,axis=-1)
    return distance/np.sqrt(delta.shape[-1]) if metric == "rmse" else distance


def acceptance(values, target, config):
    if config.method=='collision_plus_cpa':
        from .cpa import component_distances
        distances=component_distances(values,target,config.k_width)
        tau=np.asarray(config.kv_thresholds)
        return np.all(distances < tau+0.01*tau,axis=-1),distances
    distances=kv_distance(values,target,config.metric)
    return distances<config.epsilon,distances


def lowest_scored_ids(indices, scores, count):
    """Exact lexsort(score, original ID) prefix without sorting the full pool.

    indices must be ascending unique IDs, as produced by flatnonzero(available).
    Include boundary ties in ID order before sorting the selected prefix.
    """
    if count<=0:return indices[:0]
    values=scores[indices]
    if count>=len(indices):return indices[np.lexsort((indices,values))]
    boundary=np.partition(values,count-1)[count-1]
    below=indices[values<boundary]
    tied=indices[values==boundary][:count-len(below)]
    selected=np.concatenate((below,tied))
    return selected[np.lexsort((selected,scores[selected]))]


def recover_position(backend, target, order, index, config, surrogate=None, candidate_ids=None, on_batch=None):
    config.validate();n=backend.vocab_size;order=np.asarray(order,dtype=np.int64)
    if order.shape!=(n,) or not np.array_equal(np.sort(order),np.arange(n)):
        raise ValueError("Probability ordering must include the entire original vocabulary exactly once")
    target=np.asarray(target,dtype=np.float64)
    if target.ndim!=1 or not np.isfinite(target).all():
        raise ValueError("Invalid target KV")
    if config.batch_size>getattr(backend,"batch_capacity",128):
        raise ValueError("Requested batch exceeds the compiled model capacity")
    pool=np.arange(n) if candidate_ids is None else np.asarray(candidate_ids)
    if pool.ndim!=1 or pool.dtype.kind not in "iu" or not len(pool) or len(np.unique(pool))!=len(pool) or pool.min()<0 or pool.max()>=n:
        raise ValueError("Candidate pool must contain unique original model token IDs")
    started=time.perf_counter();available=np.zeros(n,dtype=bool);available[pool]=True
    probability_method=config.method in ('collision_plus','collision_plus_cpa')
    if config.method=='collision_plus_cpa':
        # Match upstream range(0, len(vocab)//partition, logical_batch),
        # including its possible final-batch overshoot.
        ranked=order[available[order]];limit=len(ranked)//config.stop_partition
        stop=((limit+config.batch_size-1)//config.batch_size)*config.batch_size
        available[ranked[stop:]]=False
    queried=[];measured=[];distances=[];stages=[];ranking=0.;forward=0.
    distance_space="original" if config.method=='collision_plus_cpa' else config.distance_space
    top=np.array([],dtype=np.int64);random=np.array([],dtype=np.int64)
    if config.method=='guided':
        top=order[available[order]][:min(config.probability_seeds,len(pool))]
        rest=np.setdiff1d(pool,top)
        random=np.random.default_rng(config.seed+index).choice(rest,min(config.random_seeds,len(rest)),replace=False)
    initial=[("probability",top.tolist()),("random",random.tolist())]
    accepted=None;reason="probability_pruned" if config.stop_partition>1 else "exhausted";passing_ids=[];stage_passing=[];accepted_codes=None;accepted_distance=None
    while available.any():
        remaining=int(available.sum())
        if config.max_queries is not None:
            remaining=min(remaining,config.max_queries-len(queried))
        if remaining<=0:
            reason="resource_budget";break
        tick=time.perf_counter()
        if config.method=='exhaustive':
            phase='exhaustive';chosen=np.flatnonzero(available)[:min(config.batch_size,remaining)].tolist()
        elif probability_method:
            phase="probability";chosen=order[available[order]][:min(config.batch_size,remaining)].tolist()
        elif initial:
            phase,chosen=initial.pop(0)
            if not chosen:
                continue
        else:
            if surrogate is None:
                raise ValueError("Guided search requires a surrogate")
            phase="ridge";scores=np.asarray(surrogate.scores(queried,np.asarray(measured),target,np.asarray(distances)))
            if config.metric=="rmse":scores=scores/len(target)  # squared L2 -> MSE; ranking unchanged
            if scores.shape!=(n,) or not np.isfinite(scores).all():
                raise ValueError("Surrogate must score all original candidates finitely")
            indices=np.flatnonzero(available)
            chosen=lowest_scored_ids(indices,scores,min(config.proposals,remaining)).tolist()
        ranking+=time.perf_counter()-tick
        chosen=chosen[:remaining]
        stage_passing=[]
        for start in range(0,len(chosen),config.batch_size):
            ids=chosen[start:start+config.batch_size]
            if len(set(ids))!=len(ids) or not available[ids].all():
                raise ValueError("Duplicate or already queried candidate")
            tick=time.perf_counter()
            if config.method=='collision_plus_cpa':
                query=getattr(backend,'query_retained',None)
                if not callable(query):raise ValueError("CPA backend must retain candidate cache for acceptance")
                codes=query(ids)
            else:codes=backend.query(ids)
            values=measured_values(backend,codes,config)
            distance_space=config.space_for(np.asarray(codes).dtype)
            forward+=time.perf_counter()-tick
            if values.shape!=(len(ids),len(target)) or not np.isfinite(values).all():
                raise ValueError("Malformed real candidate KV")
            d=kv_distance(values,target,config.metric)
            queried.extend(ids)
            if config.method=='guided':measured.extend(values)
            distances.extend(d.tolist());available[ids]=False
            mask,component_distances=acceptance(values,target,config)
            passing=np.flatnonzero(mask)
            stages.append(dict(phase=phase,queries=len(queried),valid_candidates=len(ids),physical_slots=getattr(backend,"physical_batch_capacity",getattr(backend,"batch_capacity",128)),
                minimum_distance=float(d.min()),distance_metric=config.metric,passing_ids=[ids[i] for i in passing]))
            if config.method=='collision_plus_cpa':
                stages[-1]['kv_distances']=component_distances.tolist()
            if on_batch is not None:
                on_batch(dict(type="batch", index=index, batch=len(stages), phase=phase,
                    distance_space=config.space_for(np.asarray(codes).dtype),distance_metric=config.metric,
                    queries=len(queried), token_ids=list(ids), distances=d.tolist(),
                    passing_ids=[ids[i] for i in passing],
                    kv_distances=component_distances.tolist() if config.method=='collision_plus_cpa' else None,
                    elapsed_seconds=time.perf_counter()-started))
            if len(passing):
                best=int(passing[0] if probability_method else passing[np.argmin(d[passing])])
                stage_passing.extend(ids[i] for i in passing)
                if accepted_distance is None or float(d[best])<accepted_distance:
                    accepted=ids[best];accepted_distance=float(d[best]);accepted_codes=codes[best].tolist()
                    passing_ids=[ids[i] for i in passing]
                reason="threshold"
                # Finish the complete configured guided stage before choosing
                # its best, even across NPU chunks.
                if probability_method:
                    break
        if accepted is not None:
            break
    return dict(index=index,distance_space=distance_space,distance_metric=config.metric,accepted_token_id=accepted,accepted_distance=accepted_distance,
        accepted_l2=(accepted_distance*np.sqrt(len(target)) if config.metric=="rmse" else accepted_distance) if accepted_distance is not None else None,
        accepted_codes=accepted_codes,passing_ids_in_accepting_batch=passing_ids,
        passing_ids_in_accepting_stage=stage_passing,
        uniqueness_proven=False,stop_reason=reason,queries=len(queried),queried_token_ids=queried,
        distances=distances,stages=stages,ranking_seconds=ranking,query_transport_seconds=forward,
        search_seconds=time.perf_counter()-started,candidate_pool_size=len(pool),
        requested_initial_candidates=config.probability_seeds+config.random_seeds if config.method=='guided' else 0,
        effective_probability_seeds=len(top),effective_random_seeds=len(random),config=asdict(config))
