"""Expand the verified 1000-image release to exactly 10000 training images."""
import hashlib,json,os,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];PY=HERE/'.venv/bin/python'
OLD=HERE/'data/pilot_fixed991';OUT=HERE/'data/full_fixed991';SOURCES=ROOT/'kvidence/datasets/fixed_prefix_mixed_10000'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def status(stage,**kw):
 d=dict(stage=stage,planned=10000,reused_images=1000,target_separate=True,updated_unix=time.time(),**kw);write(HERE/'build_status.json',d);print(d,flush=True)
try:
 status('resuming_sources_for_10000')
 source_manifest=SOURCES/'manifest.json'
 while True:
  try:m=json.loads(source_manifest.read_text())
  except (FileNotFoundError,json.JSONDecodeError):m={}
  if m.get('status')=='complete':break
  log=HERE/'prepare_mixed_sources_resume.log'
  if log.exists() and 'Traceback' in log.read_text():raise RuntimeError('Source preparation failed; inspect resume source log')
  time.sleep(5)
 rows=[r for r in m['samples'] if r['split']=='train'];assert len(rows)==10000
 target_sha=json.loads((HERE/'experiment_plan.json').read_text())['target']['sha256']
 assert target_sha not in {r['sha256'] for r in rows}
 training=dict(m);training.update(samples=rows,counts=dict(train=10000),source_manifest_sha256=sha(source_manifest),auxiliary_captures=False)
 manifest=SOURCES/'train_manifest.json';write(manifest,training)
 protocol=json.loads((OLD/'protocol.json').read_text());assert protocol['status']=='complete' and protocol['samples']==1000
 previous=json.loads((OLD/'index.json').read_text());valid={(r['source_dataset'],r['image_id']):r for r in rows}
 (OUT/'samples').mkdir(parents=True,exist_ok=True)
 for r in previous:
  key=(r['source_dataset'].lower(),r['source_image_id']);assert key in valid and valid[key]['sha256']==r['source_sha256']
  assert r['prefix_sha256']==protocol['prefix_sha256']
  p=OLD/r['file'];assert sha(p)==r['sha256']
  for src in [p,p.with_suffix('.json')]:
   dst=OUT/'samples'/src.name
   if dst.exists():assert sha(src)==sha(dst)
   else:os.link(src,dst)
 write(OUT/'warm_start.json',dict(images=1000,origin='data/pilot_fixed991',origin_protocol_sha256=sha(OLD/'protocol.json'),origin_index_sha256=sha(OLD/'index.json'),prefix_sha256=protocol['prefix_sha256'],files_hard_linked=True))
 status('collecting_to_10000',completed=1000,remaining=9000)
 with (HERE/'collection_10000.log').open('a') as log:
  result=subprocess.run([str(PY),str(HERE/'collect_fixed_10k.py'),'--manifest',str(manifest),'--out',str(OUT)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0,'Collection failed; inspect collection_10000.log'
 status('validating_10000')
 with (HERE/'validation_10000.log').open('a') as log:
  result=subprocess.run([str(PY),str(HERE/'validate_fixed.py'),str(OUT)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0,'Validation failed'
 report=json.loads((OUT/'validation.json').read_text());assert report['images']==10000 and report['split_summary']['train']['images']==10000
 status('complete_collected_10000',completed=10000,validation_report=str(OUT/'validation.json'))
except BaseException as e:
 status('failed_10000',error=str(e));raise
