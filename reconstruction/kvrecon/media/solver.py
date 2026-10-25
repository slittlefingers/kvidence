"""Adapter-constrained latent fitting, then real-backend verification/replay.

The Adam schedule follows the earlier MLX pilot. A model adapter supplies a
differentiable objective and projection; no gradient through an INT8 NPU is
assumed. The current text-only NPU backend deliberately lacks these hooks.
"""
import time
import numpy as np
from ..common import dump, digest
from ..search import acceptance, measured_values
from . import MediaUnavailable


def fit(problem):
    native_fit = getattr(problem, 'fit_native', None)
    if callable(native_fit):
        z, loss, trace = native_fit()
        z = np.asarray(z)
        if not np.isfinite(z).all() or not np.isfinite(loss):
            raise ValueError('Nonfinite native media optimization result')
        return z, float(loss), trace
    u = np.asarray(problem.initial, dtype=np.float32).copy()
    steps, lr = getattr(problem, 'steps', 3000), getattr(problem, 'lr', 0.05)
    if type(steps) is not int or steps < 0 or not np.isfinite(lr) or lr <= 0 or not np.isfinite(u).all():
        raise ValueError('Invalid media optimization configuration')
    m, v = np.zeros_like(u), np.zeros_like(u)
    best, best_u, trace = float('inf'), None, []
    for i in range(steps + 1):
        loss, grad = problem.loss_and_grad(u)
        loss, grad = float(loss), np.asarray(grad, dtype=np.float32)
        if not np.isfinite(loss) or grad.shape != u.shape or not np.isfinite(grad).all():
            raise ValueError('Invalid media objective or gradient')
        if loss < best:
            best, best_u = loss, u.copy()
        if i % 25 == 0 or i == steps:
            trace.append(dict(step=i, loss=loss, best_loss=best))
        if i == steps:
            break
        m = .9*m + .1*grad
        v = .999*v + .001*grad*grad
        rate = lr*(.15+.85*.5*(1+np.cos(np.pi*i/steps)))
        u -= rate*(m/(1-.9**(i+1)))/(np.sqrt(v/(1-.999**(i+1)))+1e-8)
        # Enable only when the model adapter declares this RMS invariance.
        if getattr(problem, 'normalize_latent', False):
            u /= np.sqrt(np.mean(u*u, axis=-1, keepdims=True)+1e-8)
    z = np.asarray(problem.project(best_u))
    if not np.isfinite(z).all():
        raise ValueError('Nonfinite recovered media features')
    return z, best, trace


def recover_latent(backend, request, problem_factory=None):
    context_factory = getattr(backend, 'media_context', None)
    use_context = callable(context_factory) and problem_factory is not None
    required = ('query_media', 'commit_media') + (() if use_context else ('media_problem',))
    missing = [name for name in required if not callable(getattr(backend, name, None))]
    if missing:
        raise MediaUnavailable('media_backend_unsupported: missing ' + ', '.join(missing))
    # Existing differentiable Gemma pilots produce original-value KV. Do not
    # silently fit them against integer-code targets after a metric change.
    if request['config'].distance_space == 'lsb' and getattr(backend,'media_distance_space',None) != 'lsb':
        raise MediaUnavailable('media_distance_space_unsupported: adapter must declare an LSB-space objective')
    start = backend.position
    if start != request['span']['start']:
        raise ValueError('Media start disagrees with recovered prefix')
    tick = time.perf_counter()
    # Only recovered-prefix backend state and the selected observed KV enter.
    if use_context:
        context = context_factory(request['modality'], request['positions'])
        problem = problem_factory(media_positions=request['positions'], targets=request['targets'], **context)
    else:
        problem = backend.media_problem(request['modality'], request['positions'], request['targets'])
    z, loss, trace = fit(problem)
    codes = np.asarray(backend.query_media(z, request['modality']))
    if backend.position != start:
        raise ValueError('Media fitting/query must not advance the prefix')
    values = measured_values(backend,codes,request['config'])
    if values.shape != request['targets'].shape or not np.isfinite(values).all():
        raise ValueError('Malformed media KV verification output')
    valid, _ = acceptance(values, request['targets'], request['config'])
    output = request['output']; output.mkdir(parents=True, exist_ok=False)
    np.save(output/'features.npy', z)
    if getattr(problem, 'best_u', None) is not None:
        np.save(output/'latent.npy', problem.best_u)
    dump(output/'optimization.json', trace)
    from ..search import kv_distance
    distances = kv_distance(values,request['targets'],request['config'].metric)
    report = dict(status='unresolved', distance_space=request['config'].space_for(codes.dtype),fit_loss=loss, distance_metric=request['config'].metric,per_position_distance=distances.tolist(),per_position_l2=np.linalg.norm(values-request['targets'],axis=-1).tolist(),
                  optimizer=getattr(problem, 'fit_metadata', None),
                  features_sha256=digest(output/'features.npy'),
                  feature_kind='recovered replacement representation, not original media',
                  seconds=time.perf_counter()-tick)
    if not np.all(valid):
        dump(output/'result.json', report)
        raise MediaUnavailable('media_verification_failed: real-backend KV exceeds acceptance threshold')
    committed = np.asarray(backend.commit_media(z, request['modality']))
    actual = measured_values(backend,committed,request['config'])
    if actual.shape != values.shape or not np.isfinite(actual).all():
        raise ValueError('Malformed committed media KV')
    valid, _ = acceptance(actual, request['targets'], request['config'])
    if (backend.position != request['span']['end'] or not np.all(valid)
            or not np.array_equal(committed, codes)):
        dump(output/'result.json', dict(report, status='commit_mismatch'))
        raise MediaUnavailable('media_commit_mismatch: stop before recovering subsequent text')
    report.update(status='complete', seconds=time.perf_counter()-tick, commit_matches_query=True)
    dump(output/'result.json', report)
    return report
