"""Discrete surrogate trust-region search; acceptance always uses real raw KV."""
import time
import numpy as np
import torch

EPSILON = 4.0
SEED = 20261019
ANCHORS = 512  # Surrogate training size, NEVER a query budget.


def best_indices(scores, allowed, count):
    indices = np.flatnonzero(allowed)
    count = min(count, len(indices))
    if not count:
        return []
    if count < len(indices):
        indices = indices[np.argpartition(scores[indices], count - 1)[:count]]
    return indices[np.lexsort((indices, scores[indices]))].tolist()


class FastSurrogate:
    """Same standardized ridge objective; factor the squared residual in query space.

    Instead of materializing 53,150 x 1,024 predicted KV values, use
    ||g B - t||^2 = g(B B^T)g^T - 2g(Bt) + t^T t.
    Embeddings and their squares stay on the GPU across positions.
    """
    def __init__(self, rt, embeddings):
        self.rt, self.embeddings = rt, embeddings
        self.e = rt.mx.array(embeddings.contiguous().numpy())
        self.e2 = self.e.square()
        rt.mx.eval(self.e, self.e2)

    def fit(self, queried, measured, target, distances, geometry=False):
        mx = self.rt.mx
        q = np.asarray(queried)
        # Bound only regression memory; all unqueried tokens remain searchable.
        keep = np.arange(len(q))
        if len(keep) > ANCHORS:
            nearest = np.argsort(distances, kind="stable")[:ANCHORS // 2]
            rest = np.setdiff1d(keep, nearest)
            diverse = rest[np.linspace(0, len(rest)-1, ANCHORS-len(nearest), dtype=int)]
            keep = np.r_[nearest, diverse]
        x = self.embeddings[torch.tensor(q[keep])]
        mean, scale = x.mean(0), x.std(0).clamp_min(1e-4)
        x = (x-mean)/scale
        y = measured[torch.tensor(keep)]
        ym, ys = y.mean(0), y.std(0).clamp_min(1e-4)
        y = (y-ym)/ys
        xm, ymean = x.mean(0), y.mean(0)
        xc, yc = x-xm, y-ymean
        system = xc @ xc.T + torch.eye(len(x))*(.01*len(x))
        dual = torch.linalg.solve(system, yc)
        b = dual*ys
        t = target-(ym+ymean*ys)
        gpu = lambda a: mx.array(a.contiguous().numpy())
        # g = ((E-mean)/scale-xm) @ xc.T, with no N x d normalization temporary.
        g = self.e @ gpu((xc/scale).T) - gpu((mean/scale+xm) @ xc.T)
        scores = mx.sum((g @ gpu(b @ b.T))*g, axis=-1) - 2*(g @ gpu(b @ t)) + float(t @ t)
        mx.eval(scores)
        result = dict(scores=np.maximum(np.array(scores), 0.), anchors=len(x))
        if geometry:
            center = int(q[int(np.argmin(distances))])
            c = self.embeddings[center]
            inv = scale.reciprocal().square()
            d2 = (self.e2 @ gpu(inv) - 2*(self.e @ gpu(c*inv)) + float((c*c*inv).sum()))/len(c)
            mx.eval(d2)
            result.update(center=center, embedding_distances=np.sqrt(np.maximum(np.array(d2), 0.)))
        # Used only by independent numerical preflight, not the candidate policy.
        result["reference_factors"] = (mean, scale, xm, xc, dual, ymean, ys, ym)
        return result


def trust_update(state, predicted_before, predicted_after, before_l2, after_l2, boundary):
    predicted = predicted_before-predicted_after
    actual = before_l2**2-after_l2**2
    rho = actual/predicted if predicted > 1e-8 else None
    if rho is None or rho < .25:
        state *= .5
    elif rho > .75 and boundary and actual > 0:
        state *= 1.6
    return float(np.clip(state, .25, 4.)), rho


def search(rt, runtime, cache, ids, embeddings, target, order, index, engine, adaptive):
    oracle = rt.Oracle(runtime, cache, ids, 5, 64)
    queried, measured, all_l2, stages = [], None, [], []
    available = np.ones(len(ids), dtype=bool)
    top = order[:50]
    remaining = np.setdiff1d(np.arange(len(ids)), top)
    random_part = np.random.default_rng(SEED+index).choice(remaining, 150, replace=False)
    target64 = target.numpy().astype(np.float64)
    ticks = dict(forward=0., ranking=0.)
    selected, selected_vector, selected_distance = None, None, None
    state, initial_gap, base_radius = 1., None, None
    started = time.perf_counter()
    while len(queried) < len(ids):
        tick = time.perf_counter()
        fit, detail = None, {}
        if not queried:
            phase, chosen = "probability_50", top.tolist()
        elif len(queried) == 50:
            phase, chosen = "random_150", random_part.tolist()
        else:
            phase = "adaptive_trust" if adaptive else "global_ridge"
            fit = engine.fit(queried, measured, target, all_l2, geometry=adaptive)
            scores = fit["scores"]
            if not adaptive:
                chosen = best_indices(scores, available, 20)
            else:
                current_gap = min(all_l2)
                if initial_gap is None:
                    initial_gap = current_gap
                    base_radius = float(np.median(fit["embedding_distances"][available]))
                # Shrink when real residual shrinks; trust feedback can expand or contract.
                gap_factor = max(.25, min(1., (current_gap/initial_gap)**.5))
                radius = base_radius*state*gap_factor
                ed = fit["embedding_distances"]
                floor = float(np.partition(ed[available], min(17, available.sum()-1))[min(17, available.sum()-1)])
                effective = max(radius, floor*(1+1e-6))
                local = best_indices(scores, available & (ed <= effective), 18)
                global_available = available.copy()
                global_available[local] = False
                # Two global proposals avoid a permanently closed local neighborhood.
                chosen = local+best_indices(scores, global_available, 20-len(local))
                detail = dict(radius_state_before=state, radius=radius, effective_radius=effective,
                              base_radius=base_radius, gap_factor=gap_factor,
                              center_candidate_index=fit["center"], best_l2_before=current_gap)
        ticks["ranking"] += time.perf_counter()-tick
        assert chosen and len(chosen)==len(set(chosen)) and available[chosen].all()
        tick = time.perf_counter()
        new = oracle.query(chosen)
        ticks["forward"] += time.perf_counter()-tick
        distances = np.linalg.norm(new.numpy().astype(np.float64)-target64, axis=-1)
        assert np.isfinite(distances).all()
        if adaptive and fit is not None:
            i = int(np.argmin(distances))
            candidate = chosen[i]
            state, rho = trust_update(state, float(fit["scores"][fit["center"]]),
                float(fit["scores"][candidate]), min(all_l2), float(distances[i]),
                bool(fit["embedding_distances"][candidate] >= .8*detail["effective_radius"]))
            detail.update(radius_state_after=state, rho=rho,
                          predicted_center_squared_l2=float(fit["scores"][fit["center"]]),
                          predicted_best_new_squared_l2=float(fit["scores"][candidate]))
        queried.extend(chosen)
        available[chosen] = False
        all_l2.extend(distances.tolist())
        measured = new if measured is None else torch.cat([measured, new])
        passing = np.flatnonzero(distances < EPSILON)
        stages.append(dict(phase=phase, queries=len(queried), batch_size=len(chosen),
            batch_min_raw_l2=float(distances.min()), passing_in_batch=int(len(passing)), **detail))
        if len(passing):
            best = int(passing[np.argmin(distances[passing])])
            selected = int(ids[chosen[best]])
            selected_vector, selected_distance = new[best].tolist(), float(distances[best])
            break
    return dict(index=index, queries=len(queried), queried_token_ids=ids[torch.tensor(queried)].tolist(),
        raw_l2_distances=all_l2, accepted_token_id=selected, accepted_raw_l2=selected_distance,
        accepted_kv=selected_vector, stages=stages, ranking_seconds=ticks["ranking"],
        forward_seconds=ticks["forward"], search_seconds=time.perf_counter()-started,
        finite_pool_exhausted=selected is None and len(queried)==len(ids))
