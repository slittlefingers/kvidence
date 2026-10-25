"""Timed trainable-MLP vs matched linear/fixed-tanh local prediction checks."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from kvrecon.common import dump,digest
from kvrecon.media.image_surrogate import LocalSurrogate
start_import=time.perf_counter()
import torch
from kvrecon.media.image_mlp import TinyMLPSurrogate
IMPORT_SECONDS=time.perf_counter()-start_import


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Output must be new')
    torch.set_num_threads(1);records=[]
    for seed in (17,29):
        run=Path(__file__).parent/'results'/f'0001_geometry300_seed{seed}_20261021'
        proto=json.loads((run/'protocol.json').read_text())
        for r in (25,100,300):
            xx=[];yy=[]
            for k in (r-1,r):
                with np.load(run/f'round_{k:04d}_probes.npz') as d:xx.append(d['inputs']);yy.append(d['kv'])
            x=np.concatenate(xx);y=np.concatenate(yy)
            with np.load(run/f'round_{r:04d}_update.npz') as d:
                test_x=d['inputs'][:-2];test_y=d['kv'][:-2];center=d['inputs'][-2];q=d['kv'][-2]
            keep=[not np.any(np.max(np.abs(x-a),axis=1)<1e-6) for a in test_x]
            test_x=test_x[keep];test_y=test_y[keep]
            radius=proto['perturbation']*(1+(r-1)/100)**(-.101)
            for label,model in [('linear',LocalSurrogate(nonlinear=0.)),('fixed_tanh',LocalSurrogate(nonlinear=.1)),
                    ('mlp100',TinyMLPSurrogate(steps=100)),('mlp300',TinyMLPSurrogate(steps=300))]:
                model.fit(x,y,center,q,radius);tick=time.perf_counter();pred=model.predict(test_x);predict=time.perf_counter()-tick
                row=dict(seed=seed,round=r,model=label,prediction_rmse=float(np.sqrt(np.mean((pred-test_y)**2))),
                    fit_seconds=model.fit_seconds,predict_seconds=predict,train_rows=len(x),test_rows=len(test_x),
                    train_seconds=getattr(model,'train_seconds',None),parameters=getattr(model,'parameter_count',None))
                if label=='mlp100':
                    model.fit(x,y,center,q,radius);row['warm_fit_seconds']=model.fit_seconds;row['warm_train_seconds']=model.train_seconds
                records.append(row);print(row,flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    dump(args.output,dict(torch_version=torch.__version__,device='CPU',torch_threads=1,
        torch_import_seconds=IMPORT_SECONDS,source_sha256=digest(ROOT/'kvrecon/media/image_mlp.py'),
        note='Correlated trajectory holdouts, no test data used for training or early stopping. Warm timing uses same training batch again.',records=records))

if __name__=='__main__':main()
