"""Collect at most 500 new images per phone session, unload, then cool for 5 minutes."""
import json,subprocess,time
from pathlib import Path
from thermal_control import CoolingRequired,thermal_snapshot
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];OUT=HERE/'data/full_fixed991';PY=HERE/'.venv/bin/python';MANIFEST=ROOT/'kvidence/datasets/fixed_prefix_mixed_10000/train_manifest.json'
def write(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def count():
 if not (OUT/'index.json').exists():return 1000
 return len(json.loads((OUT/'index.json').read_text()))
def status(stage,**kw):
 d=dict(stage=stage,planned=10000,completed=count(),batch_new_images=500,cooldown_seconds=300,target_separate=True,updated_unix=time.time(),**kw);write(HERE/'build_status.json',d);print(d,flush=True)
def cool(reason):
 deadline=time.time()+300;status('cooling',reason=reason,resume_not_before_unix=deadline)
 while time.time()<deadline:time.sleep(min(5,deadline-time.time()))
 while True:
  try:
   d=thermal_snapshot(OUT/'thermal')
   if d['thermal_status']<=1:return
   status('cooling_extended',thermal_status=d['thermal_status'])
  except (CoolingRequired,subprocess.TimeoutExpired) as e:status('cooling_extended',reason=str(e))
  time.sleep(30)
try:
 write(OUT/'cooling_policy.json',dict(new_images_per_batch=500,rest_seconds=300,worker_closed_during_rest=True,thermal_poll_seconds=30,early_pause_thermal_status=3,resume_max_thermal_status=1))
 while count()<10000:
  try:d=thermal_snapshot(OUT/'thermal')
  except (CoolingRequired,subprocess.TimeoutExpired):cool('thermal telemetry unavailable');continue
  if d['thermal_status']>=3:cool('severe thermal status before session');continue
  goal=min(10000,((count()//500)+1)*500);status('collecting_batch',batch_end=goal)
  with (HERE/'collection_10000_batched.log').open('a') as log:
   r=subprocess.run([str(PY),str(HERE/'collect_fixed_10k.py'),'--manifest',str(MANIFEST),'--limit',str(goal),'--out',str(OUT)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  if r.returncode==75:cool('severe thermal status during session');continue
  if r.returncode:raise RuntimeError(f'Collector exit {r.returncode}; see collection_10000_batched.log')
  assert count()==goal
  if count()<10000:cool('500 new images completed')
 status('validating_10000')
 with (HERE/'validation_10000.log').open('a') as log:
  subprocess.run([str(PY),str(HERE/'validate_fixed.py'),str(OUT)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 status('complete_collected_10000')
except BaseException as e:status('failed_10000',error=str(e));raise
