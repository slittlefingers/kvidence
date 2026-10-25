"""Matched-budget SPSA-vs-local-ridge reports; no historical result rewriting."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[29,17]);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--spsa-rounds',type=int,default=300)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.4));rows=[];milestones=[]
    for seed in args.seeds:
        for method in ('ridge','spsa'):
            name=f'0001_{"rounds300" if method=="ridge" else f"spsa{args.spsa_rounds}"}_seed{seed}_20261020'
            run=args.root/name;folder=run/'recovery/position_0005' if method=='ridge' else run
            trace=json.loads((folder/'optimization.json').read_text());result=json.loads((folder/'result.json').read_text())
            if method=='spsa':initial=result['initial_rmse'];seconds=result['optimizer_seconds']
            else:
                with np.load(folder/'initial.npz') as d:initial_values=d['kv'][0]
                with np.load(args.root.parent/'input/0001/observed.npz') as d:target=d['codes'][0].astype(float)
                initial=float(np.sqrt(np.mean((initial_values-target)**2)));seconds=result['seconds']
            values=[initial]+[t['distance'] if method=='ridge' else t['best_rmse'] for t in trace]
            times=[0.]+[t['elapsed_seconds'] for t in trace]
            earliest=next(i for i,v in enumerate(values) if v==values[-1])
            row=dict(seed=seed,method=method,initial_rmse=initial,final_rmse=values[-1],
                rounds=len(trace),queries=result['queries'],seconds=seconds,last_improvement_round=earliest,
                unchanged_tail_rounds=len(trace)-earliest,passes_query=result['passes_query'],
                last50_improvement=values[-51]-values[-1] if len(trace)>=50 else None)
            rows.append(row)
            for n in (0,2,5,10,20,50,100,200,300,400,500,600,750,1000):
                if n<=len(trace):milestones.append(dict(seed=seed,method=method,round=n,rmse=values[n],seconds=times[n],queries=1+n*33))
            label=f'{method}, seed {seed}'
            axes[0].plot(np.arange(len(values))*33+1,values,label=label)
            axes[1].plot(times,values,label=label)
    for ax in axes:ax.grid(alpha=.25);ax.legend();ax.set_ylabel('Best measured KV RMSE (LSB steps)')
    axes[0].set_xlabel('Candidate measurements');axes[1].set_xlabel('Optimizer seconds (worker setup excluded)')
    fig.suptitle('Pixel NPU — image_chat_100 / 0001, layer 5, first image position')
    fig.tight_layout();fig.savefig(args.output/'comparison.png',dpi=180);fig.savefig(args.output/'comparison.pdf')
    for name,data in [('summary',rows),('milestones',milestones)]:
        (args.output/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
        with (args.output/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
