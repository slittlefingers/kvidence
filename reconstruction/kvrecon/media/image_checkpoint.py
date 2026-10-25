"""Recover joint search Adam/RNG state without restarting its random sequence."""
import json
from pathlib import Path
import numpy as np
from .image_geometry import orthogonal_directions


def save_state(path,rng,m,v,steps):
    np.savez_compressed(path,m=m,v=v,steps=np.array(steps),rng=np.array(json.dumps(rng.bit_generator.state)))


def restore_state(run,target,width):
    run=Path(run);protocol=json.loads((run/'protocol.json').read_text())
    rng=np.random.default_rng(protocol['seed'])
    checkpoint=run/'optimizer_state.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as data:
            m=data['m'].copy();v=data['v'].copy();steps=int(data['steps'])
            rng.bit_generator.state=json.loads(str(data['rng'].item()))
        if m.shape!=(width,) or v.shape!=(width,) or steps<0 or not np.isfinite(m).all() or not np.isfinite(v).all() or (v<0).any():
            raise ValueError('Invalid optimizer checkpoint')
        return rng,m,v,steps,'checkpoint'
    if protocol.get('resume_run') or protocol.get('warm_start') or protocol.get('surrogate') or not protocol.get('geometry'):
        raise ValueError('Legacy state replay requires a cold-start geometry run')
    m=np.zeros(width);v=m.copy();trace=json.loads((run/'optimization.json').read_text())
    for row in trace:
        count=row.get('directions',protocol['directions']);c=row['perturbation']
        delta=orthogonal_directions(rng,count,width)
        with np.load(run/f"round_{row['round']:04d}_probes.npz") as data:
            inputs=data['inputs'];values=data['kv'].astype(float)
        if values.shape!=(2*count,len(target)) or not np.allclose(inputs[:count]-inputs[count:],2*c*delta,atol=3e-6):
            raise ValueError('Legacy probe sequence does not match optimizer replay')
        losses=.5*np.mean((values-target)**2,axis=1)
        gradient=np.mean(((losses[:count]-losses[count:])/(2*c))[:,None]*delta,axis=0)
        m=.9*m+.1*gradient;v=.999*v+.001*gradient**2
    return rng,m,v,len(trace),'replayed measured probes'
