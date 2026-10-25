"""Collision+ CPA statistics and threshold fitting (Shadow in the Cache).

Equations/branching follow kvcloak commit 6b40f36, attack/collision.py.
Calibration is a separate, labelled experiment, never an attack-time oracle.
"""
import json
from pathlib import Path
import time
import numpy as np
from .common import dump, digest

UPSTREAM='6b40f36edb2f337557543e7e60b10022308883d4'
METRIC='separate K/V Frobenius norms, original values, float32 accumulation'


def component_distances(values, target, k_width):
    values=np.asarray(values,dtype=np.float32);target=np.asarray(target,dtype=np.float32)
    if not 0<k_width<values.shape[-1] or target.shape!=(values.shape[-1],):
        raise ValueError('Invalid K/V boundary or target')
    delta=values-target
    return np.stack([np.linalg.norm(delta[...,:k_width],axis=-1),
                     np.linalg.norm(delta[...,k_width:],axis=-1)],axis=-1)


def fit_thresholds(stats, rank=8):
    """Same branches/objective/bounds as official get_collision_threshold.

    Fail closed for the upstream optimizer's undefined zero-variance OTHER
    branch. A zero threshold is retained exactly (strict < then accepts none).
    """
    from scipy.stats import norm
    from scipy.optimize import minimize_scalar
    if type(rank) is not int or rank<2:raise ValueError('rank must be an integer >= 2')
    keys=('target_max','target_mean','target_std','others_min','others_mean','others_std')
    a={k:np.asarray(stats[k],dtype=float) for k in keys}
    if any(v.shape!=(2,) or not np.isfinite(v).all() or (v<0).any() for v in a.values()):
        raise ValueError('Need finite nonnegative K/V statistics')
    if np.any(a['target_max']<a['target_mean']) or np.any(a['others_min']>a['others_mean']):
        raise ValueError('Inconsistent calibration extrema')
    separated=bool(np.all(a['target_max']<a['others_min']))
    thresholds=[];branches=[]
    for i in range(2):
        if separated:
            t=(a['target_max'][i]+a['others_min'][i])/2;branch='separated_midpoint'
        else:
            mt,st,mf,sf=(a[k][i] for k in ('target_mean','target_std','others_mean','others_std'))
            if st==0:t=mt;branch='zero_target_std_mean'
            else:
                if sf==0:raise ValueError('Official optimizer undefined for zero other std; do not invent a threshold')
                result=minimize_scalar(lambda x: -(norm.cdf(x,mt,st)*(1-norm.cdf(x,mf,sf))**(rank-1)),
                                       bounds=(min(mt,mf),max(mt,mf)),method='bounded')
                if not result.success or not np.isfinite(result.fun):raise ValueError('CPA optimization failed')
                t=result.x;branch='gaussian_rank_objective'
        thresholds.append(float(t));branches.append(branch)
    return dict(rank=rank,thresholds=thresholds,effective_thresholds=[t+.01*t for t in thresholds],
                branches=branches,strict_inequality=True,upstream_commit=UPSTREAM,
                note='Rank is an assumed probability rank, not sequence length. Gaussian fit is not a uniqueness proof.')


def summarize_distances(distances, token_ids, candidates):
    """Position-major [position,candidate,K_or_V], no K/V averaging."""
    d=np.asarray(distances,dtype=np.float64);ids=np.asarray(candidates)
    if d.shape!=(len(token_ids),len(ids),2) or len(token_ids)<2 or len(ids)<2 or not np.isfinite(d).all() or (d<0).any():
        raise ValueError('Need >=2 calibration positions and >=2 candidates with finite distances')
    true=[];other=[]
    for row,token in zip(d,token_ids):
        mask=ids==token
        if mask.sum()!=1:raise ValueError('Each known token must occur exactly once in the candidate pool')
        true.append(row[mask][0]);other.append(row[~mask])
    true=np.asarray(true);other=np.concatenate(other)
    return dict(target_mean=true.mean(0).tolist(),target_std=true.std(0,ddof=1).tolist(),target_max=true.max(0).tolist(),
                others_mean=other.mean(0).tolist(),others_std=other.std(0,ddof=1).tolist(),others_min=other.min(0).tolist(),
                target_count=len(true),others_count=len(other),std_ddof=1)


def collect(backend, targets, positions, prefix, tokens, candidates, k_width, output, identity, rank=8, chunk_size=4096):
    """Full-vocabulary by default at caller. Resume completed positions only.

    Labels are used to advance the KNOWN calibration prefix and label the two
    populations; they are never available to recovery search.
    """
    from .backend import CandidateBatchBackend
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    tokens=list(map(int,tokens));ids=np.asarray(candidates,dtype=np.int64)
    if len(tokens)!=len(targets) or len(tokens)<2 or not np.array_equal(positions,np.arange(len(prefix),len(prefix)+len(tokens))):
        raise ValueError('Calibration needs >=2 contiguous positions with exact known labels')
    if len(np.unique(ids))!=len(ids) or len(ids)<2 or min(ids)<0 or max(ids)>=backend.vocab_size or not set(tokens).issubset(set(ids.tolist())):
        raise ValueError('Invalid calibration candidate pool')
    if not 1<=chunk_size<=4096:raise ValueError('chunk_size must be in 1..4096')
    protocol=dict(schema=1,identity=identity,known_token_ids=tokens,public_prefix_ids=prefix,
                  candidates=ids.tolist(),k_width=k_width,metric=METRIC,rank=rank,
                  implementation_sha256=digest(Path(__file__)),numpy_version=np.__version__,
                  upstream_commit=UPSTREAM,cache_policy='known calibration prefix replay; candidate KV measured by query')
    protocol_path=output/'calibration_protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=protocol:raise ValueError('Resume calibration identity changed')
    else:dump(protocol_path,protocol)
    started=time.perf_counter();backend.reset(prefix);all_dist=[]
    wrapped=CandidateBatchBackend(backend,chunk_size)
    for i,token in enumerate(tokens):
        path=output/f'position_{i:05d}.npy';record=output/f'position_{i:05d}.json'
        if path.exists() and record.exists():
            if digest(path)!=json.loads(record.read_text())['sha256']:raise ValueError('Calibration checkpoint digest mismatch')
            distances=np.load(path,allow_pickle=False)
        else:
            tick=time.perf_counter();distances=[]
            for start in range(0,len(ids),chunk_size):
                chunk=ids[start:start+chunk_size].tolist()
                bulk=getattr(backend,'query_bulk',None)
                codes=bulk(chunk) if callable(bulk) else wrapped.query(chunk)
                distances.append(component_distances(backend.values(codes),targets[i],k_width))
                print(f'calibration position={i+1}/{len(tokens)} candidates={min(start+chunk_size,len(ids))}/{len(ids)}',flush=True)
            distances=np.concatenate(distances)
            temporary=path.with_suffix('.tmp')
            with temporary.open('wb') as f:np.save(f,distances,allow_pickle=False)
            temporary.replace(path)
            dump(record,dict(sha256=digest(path),seconds=time.perf_counter()-tick,position=int(positions[i])))
        if distances.shape!=(len(ids),2) or not np.isfinite(distances).all():raise ValueError('Invalid calibration checkpoint')
        all_dist.append(distances)
        backend.commit(token)
    stats=summarize_distances(all_dist,tokens,ids)
    fit=fit_thresholds(stats,rank)
    full=np.array_equal(np.sort(ids),np.arange(backend.vocab_size))
    asset=dict(schema=1,identity=identity,metric=METRIC,k_width=k_width,statistics=stats,fit=fit,
               coverage=dict(positions=len(tokens),candidates=len(ids),full_vocabulary=full),
               calibration_protocol_sha256=digest(protocol_path),
               source_observation_sha256=identity['observation_sha256'],
               seconds_this_invocation=time.perf_counter()-started,
               scope='full candidate vocabulary on supplied calibration positions' if full else 'pilot candidate subset; not benchmark calibration',
               checkpoint_hashes=[digest(output/f'position_{i:05d}.npy') for i in range(len(tokens))])
    dump(output/'calibration.json',asset)
    # Schema consumed by upstream get_collision_threshold(layer_idx). Null
    # uncalibrated layers fail rather than accidentally reuse another layer.
    upstream=[None]*(identity['layer_index']+1);upstream[-1]=stats
    dump(output/'upstream_statistics.json',upstream)
    return asset


def load_calibration(path, expected, observation_sha256, allow_pilot=False):
    asset=json.loads(Path(path).read_text())
    if asset.get('schema')!=1 or asset.get('metric')!=METRIC:raise ValueError('Unsupported CPA artifact')
    for k,v in expected.items():
        if asset['identity'].get(k)!=v:raise ValueError('CPA calibration identity mismatch: '+k)
    if asset['source_observation_sha256']==observation_sha256:
        raise ValueError('Evaluation observation must be independent from calibration observation')
    quant=expected.get('quantization')
    if quant and asset['k_width']!=quant[0]['width']:raise ValueError('Calibration K/V split differs from observation')
    if not asset['coverage']['full_vocabulary'] and not allow_pilot:
        raise ValueError('Subset calibration is pilot-only; explicitly pass --allow-pilot-calibration for smoke tests')
    fitted=fit_thresholds(asset['statistics'],asset['fit']['rank'])
    if fitted!=asset['fit']:raise ValueError('Stored CPA thresholds do not match calibration statistics')
    return asset
