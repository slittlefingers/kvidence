"""Standalone scientific plot from logged oracle measurements only."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    args=parser.parse_args();p=args.run
    result=json.loads((p/'result.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    colors=['#2874a6','#b65f24','#278054']
    for r,color in zip(result['runs'],colors):
        rows=[json.loads(x) for x in (p/r['kind']/'trace.jsonl').read_text().splitlines()]
        label=r['kind'].replace('_',' ')
        for ax,key,scale in ((axes[0],'queries',1000),(axes[1],'seconds',1)):
            xs=[x[key]/scale for x in rows]
            ax.plot(xs,[x['current_l2'] for x in rows],color=color,alpha=.18,lw=.65)
            ax.plot(xs,[x['best_l2'] for x in rows],color=color,lw=1.6,label=label)
    for ax in axes:
        ax.axhline(4,color='#333333',ls='--',lw=1,label='Acceptance: L2 < 4')
        ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.set_ylabel('Raw K+V L2 distance')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_xlabel('Logical candidate evaluations (thousands)')
    axes[1].set_xlabel('Optimization wall time (seconds)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Phone-native SPSA + Adam: layer-5 single-position recovery',fontsize=12)
    fig.savefig(p/'convergence.png',dpi=180)
    fig.savefig(p/'convergence.svg')
    plt.close(fig)

if __name__=='__main__':main()
