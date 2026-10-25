"""Report actual convergence traces; a finite plateau is not a global certificate."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def summarize(run, observation):
    folder=run/'recovery/position_0005'
    trace=json.loads((folder/'optimization.json').read_text())
    result=json.loads((folder/'result.json').read_text())
    with np.load(observation/'observed.npz') as source,np.load(folder/'initial.npz') as initial:
        target=source['codes'][0].astype(np.float64)
        start=float(np.sqrt(np.mean((initial['kv'][0]-target)**2)))
    distances=np.array([start]+[r['distance'] for r in trace])
    final=float(distances[-1]);first_final=int(np.flatnonzero(distances==final)[0])
    milestones=[]
    for n in sorted({0,1,2,5,10,20,30,50,100,200,300,500,1000,len(trace)}):
        if n>len(trace):continue
        row=trace[n-1] if n else None
        milestones.append(dict(round=n,queries=row['queries'] if row else 1,
            elapsed_seconds=row['elapsed_seconds'] if row else None,rmse=float(distances[n]),
            radius=row['next_radius'] if row else result['config']['radius']))
    windows=[]
    for width in (25,50,100):
        if len(trace)>=width:
            old=float(distances[-1-width])
            windows.append(dict(rounds=width,absolute_improvement=old-final,
                relative_improvement=(old-final)/old if old else 0.,
                seconds=trace[-1]['elapsed_seconds']-trace[-width-1]['elapsed_seconds'] if len(trace)>width else trace[-1]['elapsed_seconds']))
    summary=dict(run=str(run),initial_rmse=start,final_rmse=final,
        reduction_percent=100*(start-final)/start,rounds=len(trace),queries=result['queries'],
        seconds=result['seconds'],first_round_at_final_best=first_final,
        unchanged_tail_rounds=len(trace)-first_final,passes_query=result['passes_query'],
        final_radius=trace[-1]['next_radius'],windows=windows,milestones=milestones,
        caveat='Finite-run plateau or budget exhaustion, not a global convergence guarantee.')
    (run/'convergence.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (run/'convergence.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['round','queries','elapsed_seconds','rmse','radius','improved','ratio'])
        writer.writeheader();writer.writerow(dict(round=0,queries=1,rmse=start,radius=result['config']['radius']))
        for r in trace:
            writer.writerow(dict(round=r['round']+1,queries=r['queries'],elapsed_seconds=r['elapsed_seconds'],
                rmse=r['distance'],radius=r['next_radius'],improved=r['improved'],ratio=r['ratio']))
    return summary,trace


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runs',nargs='+',type=Path,required=True)
    parser.add_argument('--observation',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    summaries=[]
    for run in args.runs:
        summary,trace=summarize(run,args.observation);summaries.append(summary)
        config=json.loads((run/'protocol.json').read_text())['optimizer']
        label=f"seed {config['seed']}, {len(trace)} rounds"
        distances=[summary['initial_rmse']]+[r['distance'] for r in trace]
        axes[0].plot(range(len(distances)),distances,label=label)
        axes[1].plot([0]+[r['elapsed_seconds'] for r in trace],distances,label=label)
        axes[2].plot(range(1,len(trace)+1),[r['next_radius'] for r in trace],label=label)
    axes[0].set(xlabel='Guided rounds',ylabel='Best measured KV RMSE (LSB steps)')
    axes[1].set(xlabel='Optimizer elapsed seconds (setup excluded)',ylabel='Best measured KV RMSE (LSB steps)')
    axes[2].set(xlabel='Guided rounds',ylabel='Trust-region radius',yscale='log')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('image_chat_100 / 0001 — first image position, Gemma layer 5')
    fig.tight_layout();fig.savefig(args.output/'convergence.png',dpi=180);fig.savefig(args.output/'convergence.pdf')
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
