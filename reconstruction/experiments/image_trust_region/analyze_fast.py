"""Wall-time and query-count comparison of baseline, full-batch, and hybrid."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
OUT=BASE/'fast_comparison_20261020';OUT.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(12,4.4));summary=[];hits=[]
for ax,seed in zip(axes,(29,17)):
    ridge=BASE/f'results/0001_rounds300_seed{seed}_20261020/recovery/position_0005'
    rt=json.loads((ridge/'optimization.json').read_text());rr=json.loads((ridge/'result.json').read_text())
    for kind in ('baseline','fast','hybrid'):
        suffix='spsa1000' if kind=='baseline' else kind+'300'
        run=BASE/f'results/0001_{suffix}_seed{seed}_20261020'
        t=json.loads((run/'optimization.json').read_text());r=json.loads((run/'result.json').read_text())
        offset=rr['seconds'] if kind=='hybrid' else 0.
        qoffset=rr['queries'] if kind=='hybrid' else 0
        times=[offset]+[offset+x['elapsed_seconds'] for x in t]
        values=[r['initial_rmse']]+[x['best_rmse'] for x in t]
        queries=[qoffset+1]+[qoffset+x['queries'] for x in t]
        if kind=='hybrid':
            times=[x['elapsed_seconds'] for x in rt]+times
            values=[x['distance'] for x in rt]+values
            queries=[x['queries'] for x in rt]+queries
        ax.plot(times,values,label=kind)
        summary.append(dict(seed=seed,method=kind,rounds=r['rounds'],best_rmse=r['best_rmse'],
            stage_seconds=r['optimizer_seconds'],total_seconds=offset+r['optimizer_seconds'],
            stage_queries=r['queries'],total_queries=qoffset+r['queries'],
            repeat_exact=r['repeat_exact'],passes_query=r['passes_query']))
        for threshold in (40,35,30,25,22,20,15,10):
            idx=next((i for i,v in enumerate(values) if v<threshold),None)
            hits.append(dict(seed=seed,method=kind,target_rmse=threshold,
                seconds=times[idx] if idx is not None else None,
                queries=queries[idx] if idx is not None else None))
    ax.set(title=f'Seed {seed}',xlabel='Cumulative optimizer seconds (setup excluded)',ylabel='Best measured KV RMSE (LSB steps)')
    ax.grid(alpha=.25);ax.legend()
fig.suptitle('image_chat_100 / 0001, first image slot, Pixel NPU — unequal budgets shown explicitly')
fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'time_to_target.json').write_text(json.dumps(hits,indent=2)+'\n')
print(json.dumps(summary,indent=2))
