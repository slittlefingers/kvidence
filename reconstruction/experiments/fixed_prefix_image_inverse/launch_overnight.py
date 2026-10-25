"""Gate overnight acquisition on exact old/new collector equivalence."""
import json,statistics,subprocess,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
new=HERE/'data/socket_validation_v2_20261025';old=HERE/'data/full_fixed991'
deadline=time.time()+900
print('Waiting for 16-image socket validation.',flush=True)
while time.time()<deadline:
 p=new/'progress.json'
 if p.exists() and json.loads(p.read_text()).get('status')=='complete':break
 time.sleep(3)
else:raise RuntimeError('Validation did not complete; overnight collector not started.')
rows=json.loads((new/'index.json').read_text());assert len(rows)==16
checks=0
for row in rows:
 with np.load(new/row['file']) as actual,np.load(old/row['file']) as reference:
  for field in ('kv_codes','embedding','pre_adapter','pixels','positions','grid'):
   np.testing.assert_array_equal(actual[field],reference[field],err_msg=row['id']+': '+field);checks+=1
seconds=[row['seconds'] for row in rows];intervals=[b-a for a,b in zip(seconds,seconds[1:])]
report=dict(exact=True,samples=16,array_checks=checks,median_sample_interval_seconds=statistics.median(intervals),old_interval_reference_seconds=2.48,validation_directory=str(new))
(HERE/'socket_validation_result.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
subprocess.run([sys.executable,str(HERE/'overnight_collect.py')],cwd=ROOT,check=True)
