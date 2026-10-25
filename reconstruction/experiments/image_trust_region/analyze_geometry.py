"""Held-out KV prediction checks on saved joint-search probes and updates.

Fits only paired probes; scores separately measured update candidates. This
measures local prediction, not the unqueried performance of a new optimizer.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from kvrecon.common import dump, digest


def contract(a,b):
    return np.einsum('ij,jk->ik',a,b,optimize=False)


def analyze(run, rounds):
    protocol=json.loads((run/'protocol.json').read_text())
    result=[]
    for round_index in rounds:
        probes=run/f'round_{round_index:04d}_probes.npz'
        updates=run/f'round_{round_index:04d}_update.npz'
        with np.load(probes) as data:x=data['inputs'].astype(float);y=data['kv'].astype(float)
        with np.load(updates) as data:xt=data['inputs'].astype(float);yt=data['kv'].astype(float)
        # The runner measures its current center as the penultimate update row.
        center=xt[-2];center_q=yt[-2];n=len(x)//2
        if len(x)!=2*n or not np.allclose((x[:n]+x[n:])*.5,center,atol=2e-6):
            raise ValueError('Not paired probes at the declared center')
        c=protocol['perturbation']*(1+(round_index-1)/100)**(-.101)
        delta=(x[:n]-x[n:])/(2*c)
        response=(y[:n]-y[n:])/(2*c)
        gram=contract(delta,delta.T)
        eigen,vectors=np.linalg.eigh(gram)
        midpoint=np.mean((y[:n]+y[n:])*.5,axis=0)
        base=dict(round=round_index,probe_sha256=digest(probes),update_sha256=digest(updates),
            changed_code_fraction=float(np.mean(y[:n]!=y[n:])),
            midpoint_bias_rmse=float(np.sqrt(np.mean((midpoint-center_q)**2))),
            baseline_prediction_rmse=float(np.sqrt(np.mean((yt-center_q)**2))),
            input_rank=int(np.sum(eigen>max(eigen[-1],1e-12)*1e-6)))
        fits=[]
        for ridge in (.01,.1,1.):
            inverse=contract(vectors/(np.maximum(eigen,0)+ridge*np.mean(eigen)),vectors.T)
            coefficients=contract(contract(xt-center,delta.T),inverse)
            change=contract(coefficients,response)
            fits.append(dict(ridge=ridge,
                centered_prediction_rmse=float(np.sqrt(np.mean((center_q+change-yt)**2))),
                midpoint_prediction_rmse=float(np.sqrt(np.mean((midpoint+change-yt)**2)))))
        base['fits']=fits;result.append(base)
    return dict(run=str(run),seed=protocol['seed'],rounds=result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Output must be new')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    dump(args.output,dict(method='paired secant model; separately measured update holdout',
        note='Within-run correlated holdout; not an optimizer speedup or feature recovery claim',
        runs=[analyze(run,(1,25,100,300)) for run in args.runs]))

if __name__=='__main__':main()
