"""Measure fixed-shape replay costs without changing the running optimizer."""
import argparse
import json
from pathlib import Path
import statistics
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
rows=[json.loads(line) for line in (args.run/'device/rpc.jsonl').read_text().splitlines()]
# Continuation: two-row resume verification first, 32-row repeat check last.
rows=[row for row in rows if row['op']=='vision_query'][1:-1]
summary=[]
for size in (1,32):
    selected=[row for row in rows if row['payload_bytes']==size*1024]
    summary.append(dict(candidate_rows=size,calls=len(selected),
        mean_host_seconds=statistics.mean(row['host_seconds'] for row in selected),
        mean_decoder_seconds=statistics.mean(row['decoder_seconds'] for row in selected),
        mean_adapter_seconds=statistics.mean(row['profile_seconds'].get('model_vision_adapter_280',0) for row in selected)))
(args.run/'query_cost.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
