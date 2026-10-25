"""Offline, held-out comparison of tiny tanh residual vs linear-only ridge."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from kvrecon.common import dump,digest
from kvrecon.media.image_surrogate import LocalSurrogate


def benchmark(run,rounds):
    protocol=json.loads((run/'protocol.json').read_text());rows=[]
    for r in rounds:
        files=[run/f'round_{j:04d}_probes.npz' for j in range(max(1,r-1),r+1)]
        xx=[];yy=[]
        for path in files:
            with np.load(path) as d:xx.append(d['inputs']);yy.append(d['kv'])
        x=np.concatenate(xx);y=np.concatenate(yy)
        update=run/f'round_{r:04d}_update.npz'
        with np.load(update) as d:xt=d['inputs'][:-2];yt=d['kv'][:-2];center=d['inputs'][-2];q=d['kv'][-2]
        # Exclude exact or float-roundoff duplicate training rows from holdout.
        keep=np.array([not np.any(np.max(np.abs(x-a),axis=1)<1e-6) for a in xt])
        xt=xt[keep];yt=yt[keep]
        if not len(xt):raise ValueError('No independent holdout rows')
        c=protocol['perturbation']*(1+(r-1)/100)**(-.101)
        for nonlinear in (0.,.1,1.,10.):
            model=LocalSurrogate(hidden=64,nonlinear=nonlinear,ridge=.1).fit(x,y,center,q,c)
            start=time.perf_counter();pred=model.predict(xt);predict_seconds=time.perf_counter()-start
            rows.append(dict(round=r,nonlinear=nonlinear,train_rows=len(x),test_rows=len(xt),
                kv_prediction_rmse=float(np.sqrt(np.mean((pred-yt)**2))),
                constant_prediction_rmse=float(np.sqrt(np.mean((q-yt)**2))),
                fit_seconds=model.fit_seconds,predict_seconds=predict_seconds,
                probe_sha256=[digest(f) for f in files],update_sha256=digest(update)))
    return dict(run=str(run),seed=protocol['seed'],rows=rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--rounds',nargs='+',type=int,default=[25,100,300]);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Output must be new')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(model='64 fixed tanh features, full linear skip, local dual ridge; max 256 training queries',
        hidden=64,ridge=.1,model_seed=217,source_sha256=digest(ROOT/'kvrecon/media/image_surrogate.py'),
        note='Fits probes only; scores distinct measured updates. Correlated within-run holdout; not optimizer recovery evidence.',
        runs=[benchmark(r,args.rounds) for r in args.runs])
    dump(args.output,result)
    for run in result['runs']:
        for r in run['rows']:print(run['seed'],r['round'],r['nonlinear'],round(r['kv_prediction_rmse'],4),round(r['fit_seconds']*1000,1),'ms')

if __name__=='__main__':main()
