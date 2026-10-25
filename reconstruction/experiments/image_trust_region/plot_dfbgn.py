"""Plot measured search time; warm-start curves include their preceding cost."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
out=BASE/'dfbgn_comparison_20261021'
curves=json.loads((out/'curves.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
for ax,seed in zip(axes,(29,17)):
    for curve in curves:
        if curve['seed']!=seed:continue
        name=curve['name'];trace=curve['trace'];warm=curve['warm_seconds']
        result=json.loads((BASE/'results'/name/'result.json').read_text())
        label=('Joint ridge + SPSA' if 'joint300' in name else
            'DFBGN after joint plateau' if 'warm' in name else
            'DFBGN, larger initial radius' if 'radius' in name else 'DFBGN, default radius')
        times=[warm]+[warm+v['elapsed_seconds'] for v in trace]
        values=[result['initial_rmse']]+[v['best_rmse'] for v in trace]
        ax.plot(times,values,label=label)
        ax.scatter(times[-1],values[-1],s=18)
    ax.set_title(f'Initialization seed {seed}');ax.set_xlabel('Cumulative optimization time (seconds)')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[0].set_ylabel('Best measured LSB RMSE (lower is better)')
fig.suptitle('Same image slot; warm-start curves include preceding joint-search time')
fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);fig.savefig(out/'comparison.pdf')
