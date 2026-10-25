import json
from pathlib import Path
r=Path(__file__).resolve().parent
rows=[]
for seed in (29,17):
 for method in ('fast','hybrid','joint'):
  folder=r/'results'/f'0001_{method}300_seed{seed}_20261020'
  if not (folder/'result.json').exists():continue
  result=json.loads((folder/'result.json').read_text());trace=json.loads((folder/'optimization.json').read_text())
  warm=result.get('total_seconds_including_warmup',result['optimizer_seconds'])-result['optimizer_seconds']
  thresholds={str(t):next((p['elapsed_seconds']+warm for p in trace if p['best_rmse']<t),None) for t in (30,25,20,16,12,10)}
  if method=='hybrid':
   pre=json.loads((r/'results'/f'0001_rounds300_seed{seed}_20261020/recovery/position_0005/optimization.json').read_text())
   for t in thresholds:
    first=next((v['elapsed_seconds'] for v in pre if v['distance']<float(t)),None)
    if first is not None:thresholds[t]=first
  rows.append(dict(seed=seed,method=method,rmse=result['best_rmse'],seconds=result['total_seconds_including_warmup'],queries=result['total_queries_including_warmup'],repeat_exact=result['repeat_exact'],thresholds=thresholds))
print(json.dumps(rows,indent=2))
(r/'joint_comparison_20261020').mkdir(exist_ok=True)
(r/'joint_comparison_20261020/summary.json').write_text(json.dumps(rows,indent=2)+'\n')
