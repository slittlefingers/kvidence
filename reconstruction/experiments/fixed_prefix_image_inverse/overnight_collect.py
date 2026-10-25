"""Deadline-limited new-image collection, 500-image rests and verified 10k archives."""
import concurrent.futures
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,os,random,shutil,signal,subprocess,sys,tarfile,threading,time,urllib.request,zipfile
from pathlib import Path
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
BASE=HERE/'data/overnight_20261025';SOURCE=ROOT/'kvidence/datasets/overnight_20261025'
ARCHIVE=Path('/Volumes/TOSHIBA EXT/KVidence/20261025')
DEADLINE=datetime(2026,10,25,9,0,tzinfo=ZoneInfo('Australia/Melbourne')).timestamp()
PY=HERE/'.venv/bin/python';CONTROL=HERE/'job_control.json'
sys.path.insert(0,str(HERE));from thermal_control import thermal_snapshot,CoolingRequired

def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));tmp.replace(p)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def expired():return time.time()>=DEADLINE
def status(state,**kw):
 write(BASE/'status.json',dict(state=state,deadline=DEADLINE,updated_at=time.time(),**kw));print(state,kw,flush=True)
def available():
 if shutil.disk_usage(BASE).free<15*1024**3:raise RuntimeError('Less than 15 GiB local free space; stopped.')
def prepare():
 SOURCE.mkdir(parents=True,exist_ok=True)
 existing=json.loads((ROOT/'kvidence/datasets/fixed_prefix_mixed_10000/manifest.json').read_text())['samples']
 excluded_hashes={r['sha256'] for r in existing};excluded_ids={(r['source_dataset'].lower(),r['image_id']) for r in existing}
 excluded_hashes.add(json.loads((HERE/'experiment_plan.json').read_text())['target']['sha256'])
 saved=SOURCE/'manifest.json'
 if saved.exists():return json.loads(saved.read_text())['samples']
 status('preparing_new_images')
 records=[]
 # Use remaining official TRAIN images only; keep existing validation/test excluded.
 source=ROOT/'kvidence/datasets/hiertext/source'
 import gzip
 with gzip.open(source/'train.jsonl.gz','rt') as f:annotations=json.load(f)['annotations']
 wanted={a['image_id'] for a in annotations if ('hiertext',a['image_id']) not in excluded_ids}
 with tarfile.open(source/'train.tgz','r|gz') as archive:
  for member in archive:
   if expired():raise TimeoutError('Deadline reached during source preparation.')
   iid=Path(member.name).stem
   if not member.isfile() or iid not in wanted:continue
   blob=archive.extractfile(member).read();digest=hashlib.sha256(blob).hexdigest()
   if digest in excluded_hashes:continue
   dest=SOURCE/'images/hiertext'/f'{iid}.jpg';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(blob)
   with Image.open(dest) as im:im.verify()
   records.append(dict(image_id=iid,source_dataset='hiertext',split='train',pilot=False,file=str(dest.relative_to(SOURCE)),absolute_file=str(dest),sha256=digest,source_split='train',license='CC BY-SA 4.0'))
   excluded_hashes.add(digest)
 random.Random(20261025).shuffle(records)
 with zipfile.ZipFile(ROOT/'kvidence/datasets/shadow_chat/source/coco2017/annotations_trainval2017.zip') as archive:
  text=archive.read('annotations/instances_train2017.json').decode();decoder=json.JSONDecoder()
  images,_=decoder.raw_decode(text,text.index('[',text.index('"images"')))
  licenses,_=decoder.raw_decode(text,text.index('[',text.index('"licenses"')))
 license_by_id={x['id']:x for x in licenses};random.Random(20261025).shuffle(images)
 images=[x for x in images if ('coco2017',f"{x['id']:012d}") not in excluded_ids][:20000-len(records)]
 def download(x):
  if expired():raise TimeoutError('Deadline reached.')
  available();iid=f"{x['id']:012d}";dest=SOURCE/'images/coco2017'/x['file_name'];dest.parent.mkdir(parents=True,exist_ok=True)
  url='https://s3.amazonaws.com/images.cocodataset.org/train2017/'+x['file_name']
  for attempt in range(4):
   try:
    if not dest.exists():
     with urllib.request.urlopen(url,timeout=45) as response:blob=response.read()
     tmp=dest.with_suffix('.tmp');tmp.write_bytes(blob);tmp.replace(dest)
    with Image.open(dest) as im:im.verify()
    return dict(image_id=iid,source_dataset='coco2017',split='train',pilot=False,file=str(dest.relative_to(SOURCE)),absolute_file=str(dest),sha256=sha(dest),source_url=url,source_split='train2017',license=license_by_id[x['license']])
   except Exception:
    if attempt==3:raise
    time.sleep(1)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for i,row in enumerate(pool.map(download,images),1):
   if row['sha256'] not in excluded_hashes:records.append(row);excluded_hashes.add(row['sha256'])
   if i%500==0:status('preparing_new_images',prepared=len(records),planned=20000)
 write(saved,dict(samples=records,excluded_existing_10000=True,target_excluded=True,seed=20261025,source_scope='Additional training-only source preparation; separate HierText-only capture manifest preserves official splits.'))
 return records

def archive_shard(shard,partial=False):
 if not Path('/Volumes/TOSHIBA EXT').is_mount():raise RuntimeError('External disk is not mounted; local data retained.')
 rows=[]
 for batch_index in sorted(shard.glob('batch_*/index.json')):
  for row in json.loads(batch_index.read_text()):
   row=dict(row);row['file']=str(batch_index.parent.relative_to(shard)/row['file']);rows.append(row)
 write(shard/'index.json',rows)
 for name in ('prefix.json','prefix.txt','quantization.json'):
  original=next(shard.glob('batch_*/'+name),None)
  if original:shutil.copy2(original,shard/name)
 files=[p for p in shard.rglob('*') if p.is_file()];size=sum(p.stat().st_size for p in files)
 ARCHIVE.mkdir(parents=True,exist_ok=True)
 if shutil.disk_usage(ARCHIVE).free<size+5*1024**3:raise RuntimeError('Insufficient external disk space; local data retained.')
 dest=ARCHIVE/(shard.name+('_partial' if partial else ''));staging=dest.with_name(dest.name+'.copying');staging.mkdir(parents=True,exist_ok=True)
 checks={}
 for p in files:
  rel=p.relative_to(shard);target=staging/rel;target.parent.mkdir(parents=True,exist_ok=True)
  digest=sha(p)
  if not target.exists() or sha(target)!=digest:shutil.copy2(p,target)
  if sha(target)!=digest:raise RuntimeError('Archive checksum mismatch; source retained.')
  checks[str(rel)]=digest
 write(staging/'archive_checksums.json',checks)
 if dest.exists():raise RuntimeError('Archive destination exists; refusing overwrite.')
 staging.rename(dest)
 write(BASE/(shard.name+'_archive.json'),dict(directory=str(dest),files=len(files),bytes=size,verified=True,partial=partial))
 # Delete only this newly collected shard after every copied file is verified.
 if shard.parent.resolve()!=BASE.resolve() or not shard.name.startswith('shard_'):raise RuntimeError('Invalid cleanup source.')
 shutil.rmtree(shard)
 status('archived',directory=str(dest),bytes=size)

def main():
 BASE.mkdir(parents=True,exist_ok=True);write(CONTROL,dict(pause_requested=False))
 records=json.loads((SOURCE/'hiertext_only.json').read_text())['samples'] if (SOURCE/'hiertext_only.json').exists() else prepare()
 status('sources_ready',count=len(records))
 def watch():
  while not expired():time.sleep(min(1,max(0,DEADLINE-time.time())))
  write(CONTROL,dict(pause_requested=True));status('deadline_pause_requested')
 threading.Thread(target=watch,daemon=True).start()
 completed=0;shard=None
 start=0
 while not expired():
  if len(records)-start<500:
   full=SOURCE/'hiertext_only.json'
   if not full.exists():status('waiting_for_sources',completed=completed);time.sleep(5);continue
   seen={(r['source_dataset'],r['image_id']) for r in records}
   extra=[r for r in json.loads(full.read_text())['samples'] if (r['source_dataset'],r['image_id']) not in seen]
   if not extra and start>=len(records):break
   records.extend(extra)
  if expired():break
  if len(records)-start<500 and not (SOURCE/'hiertext_only.json').exists():
   full=SOURCE/'hiertext_only.json'
   if not full.exists():status('waiting_for_sources',completed=completed);time.sleep(5);continue
  available();shard_number=start//10000+1;shard=BASE/f'shard_{shard_number:04d}';shard.mkdir(exist_ok=True)
  batch=shard/f'batch_{start//500+1:04d}';batch.mkdir(exist_ok=True)
  manifest=SOURCE/f'batch_{start//500+1:04d}.json';write(manifest,dict(samples=records[start:start+500]))
  while not expired():
   try:
    if thermal_snapshot(BASE/'thermal')['thermal_status']<=1:break
   except (CoolingRequired,subprocess.TimeoutExpired):pass
   status('cooling_extended',completed=completed);time.sleep(5)
  if expired():break
  status('collecting',completed=completed,batch=str(batch),new_images=len(records[start:start+500]))
  shutil.copy2(manifest,batch/'source_manifest.json')
  with (batch/'collection.log').open('a') as log:
   code=subprocess.call([str(PY),str(HERE/'collect_socket.py'),'--manifest',str(manifest),'--out',str(batch),'--prefix',str(HERE/'protocol/prefix.json'),'--control',str(CONTROL)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  count=len(json.loads((batch/'index.json').read_text())) if (batch/'index.json').exists() else 0
  completed+=count
  if code==76 or expired():break
  if code!=0:raise RuntimeError(f'Collector exit {code}; stopped without replay. See {batch}/collection.log')
  if count!=len(records[start:start+500]):raise RuntimeError('Incomplete batch; stopped.')
  with (batch/'validation.log').open('w') as log:
   subprocess.run([str(PY),str(HERE/'validate_fixed.py'),str(batch)],stdout=log,stderr=subprocess.STDOUT,check=True)
  if (start+count)%10000==0:
   status('archiving',completed=completed);archive_shard(shard);shard=None
  start+=count
  if not expired():
   until=time.time()+300;status('cooling',completed=completed,resume_not_before_unix=until)
   while time.time()<until and not expired():time.sleep(1)
 if shard and shard.exists():archive_shard(shard,partial=True)
 status('stopped_at_deadline' if expired() else 'complete',completed=completed)

if __name__=='__main__':
 try:main()
 except BaseException as error:
  status('failed',error=str(error));raise
