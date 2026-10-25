"""Compare official DFBGN with measured joint ridge/SPSA traces."""
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent
out=BASE/'dfbgn_comparison_20261021';out.mkdir(exist_ok=True)
rows=[];curves=[]
for seed in (29,17):
    names=[f'0001_joint300_seed{seed}_20261020']+[p.name for p in sorted((BASE/'results').glob(f'0001_dfbgn*seed{seed}_20261021*'))]
    for name in names:
        folder=BASE/'results'/name
        if not (folder/'result.json').exists():continue
        result=json.loads((folder/'result.json').read_text())
        if not result.get('queries') or 'best_rmse' not in result:continue
        if (folder/'optimization.jsonl').exists():
            trace=[json.loads(line) for line in (folder/'optimization.jsonl').read_text().splitlines()]
        else:trace=json.loads((folder/'optimization.json').read_text())
        elapsed=result['optimizer_seconds'];warm=result['total_seconds_including_warmup']-elapsed
        row=dict(name=name,seed=seed,solver_msg=result.get('solver_msg'),initial_rmse=result['initial_rmse'],best_rmse=result['best_rmse'],
            seconds=elapsed,total_seconds=result['total_seconds_including_warmup'],queries=result['queries'],
            total_queries=result['total_queries_including_warmup'],passes=result['passes_query'],
            repeat_exact=result['repeat_exact'],status=result['status'],
            seconds_to_threshold={str(t):next((v['elapsed_seconds'] for v in trace if v['best_rmse']<t),None) for t in (30,25,20,16,12,10)})
        rows.append(row);curves.append(dict(name=name,seed=seed,warm_seconds=warm,trace=trace))
(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
(out/'curves.json').write_text(json.dumps(curves,indent=2)+'\n')
print(json.dumps(rows,indent=2))
report=BASE/'REPORT.md'
text=report.read_text().split('\n## Official DFBGN comparison (2026-10-21)')[0]
text+='''
## Official DFBGN comparison (2026-10-21)

Published dfbgn 0.1, unchanged solver source; only process-local NumPy int/float
alias compatibility. Same observation, physical adapter and NPU, first image
position, and initial input replay verification. Residuals are integer-code
errors / sqrt(1024), so the solver minimizes MSE and acceptance remains strict
LSB RMSE < 0.125. Initial radius is recorded per run (package default or explicit 2.7712812921); block 64; single-candidate
callbacks. Each run has a 173-second optimization cap, checked at callbacks.
Startup and repeatability diagnostics are excluded, as in the prior comparison.
No clean image or hidden text is read. Solver exit success does not imply KV
acceptance. The locked-phone launch produced zero queries and is excluded.

| Run | Initial RMSE | Best RMSE | Stage seconds | Stage queries | Acceptance |
|---|---:|---:|---:|---:|---|
'''
for row in rows:
    text+=f"| {row['name']} | {row['initial_rmse']:.4f} | {row['best_rmse']:.4f} | {row['seconds']:.2f} | {row['queries']} | {row['passes']} |\n"
text+='''
The two default-radius runs terminated early because delta reached rhoend,
not because the KV gate passed. They used 114/116 calls and 18.16/22.56 seconds.
Their measured best states repeated exactly 32 times. The default initial
radius is roughly 0.3 in Euclidean norm (about 0.01 per-coordinate RMS in
768 dimensions). The larger-radius follow-up uses 0.1*sqrt(768)=2.7712812921, matching the
Euclidean length of a prior SPSA perturbation. Its disconnected first launch
had zero queries and is excluded; the retry is listed separately above.
These results reject the current default configuration as an improvement on
this sample, not every possible DFBGN configuration.

The larger-radius seed-17 run reduced RMSE from 65.0270 to 59.0338 in
25.77 seconds / 130 queries, then hit the radius limit. The seed-17 warm start
remained exactly 13.1808706678 after 118 queries / 23.11 additional seconds.
Both best states repeated exactly 32 times. Thus these tested DFBGN settings
neither improve cold-start recovery nor break the existing plateau; they do
not establish that all parameterizations fail or that a global minimum was found.

Warm-start runs begin from the corresponding joint method's saved best point;
these are plateau-refinement experiments, not equal-total-cost comparisons.
Total costs including that preceding search are in summary.json. Single sample
and sequential timings are exploratory, not a general speedup claim. Physical
query efficiency can differ from wall-time efficiency because each native call
uses the compiled execution shape even for a single valid candidate. No complete
image recovery or subsequent text recovery has been demonstrated by these runs.

Artifacts: `dfbgn_comparison_20261021/summary.json`, `curves.json`, and each run's
`optimization.jsonl`, `query_*.npz`, `best_input.npy`, `best_kv.npy`, `result.json`.
Entry point: `run_dfbgn.py`; rerun analysis using `analyze_dfbgn.py`.
'''
report.write_text(text)
