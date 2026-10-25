"""Durable two-stage phone acquisition; no target-driven training or public uploads."""
import json,subprocess,time,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];PY=HERE/'.venv/bin/python';state=HERE/'build_status.json'
def status(stage,**kw):
 d=dict(stage=stage,updated_unix=time.time(),**kw);tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(state);print(d,flush=True)
def run(args,log):
 with (HERE/log).open('a') as f:r=subprocess.run([str(PY),*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(f'{log}: exit {r.returncode}')
try:
 smoke=HERE/'data/smoke_fixed991/protocol.json';status('waiting_for_smoke')
 while not smoke.exists() or json.loads(smoke.read_text()).get('status')!='complete':
  if 'Traceback' in (HERE/'smoke.log').read_text():raise RuntimeError('Smoke failed; inspect smoke.log')
  time.sleep(5)
 pilot=HERE/'data/pilot_fixed991';full=HERE/'data/full_fixed991';mixed=ROOT/'kvidence/datasets/fixed_prefix_mixed_10000/manifest.json'
 status('collecting_pilot',planned=1200)
 run([HERE/'collect_fixed.py','--limit','1200','--out',pilot], 'pilot_collection.log')
 run([HERE/'validate_fixed.py',pilot],'pilot_validation.log')
 status('pilot_complete_waiting_full_sources',pilot_images=1200)
 while not mixed.exists():
  if 'Traceback' in (HERE/'prepare_mixed_sources.log').read_text():raise RuntimeError('Mixed source preparation failed')
  time.sleep(5)
 (full/'samples').mkdir(parents=True,exist_ok=True)
 for p in (pilot/'samples').iterdir():
  dest=full/'samples'/p.name
  if not dest.exists():os.link(p,dest)
 status('collecting_full',planned=10600,pilot_reused=True)
 run([HERE/'collect_fixed.py','--manifest',mixed,'--out',full],'full_collection.log')
 run([HERE/'validate_fixed.py',full],'full_validation.log')
 status('complete',train_images=10000,validation_images=300,test_images=300,target_separate=True,validation_report=str(full/'validation.json'))
except BaseException as e:
 status('failed',error=str(e));raise
