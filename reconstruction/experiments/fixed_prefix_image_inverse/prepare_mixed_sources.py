"""Prepare 5000 COCO train2017 photographs and 2000 official CMA CC0 drawings/paintings."""
import concurrent.futures,hashlib,io,json,random,time,urllib.request,zipfile
from pathlib import Path
from PIL import Image
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3];OUT=ROOT/'kvidence/datasets/fixed_prefix_mixed_10000'
OUT.mkdir(parents=True,exist_ok=True)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def retrieve(r):
 dest=OUT/r['file'];dest.parent.mkdir(parents=True,exist_ok=True)
 for attempt in range(5):
  try:
   if not dest.exists():
    headers={'User-Agent':'Mozilla/5.0','Referer':'https://www.clevelandart.org/'} if r['source_dataset']=='clevelandart' else {}
    request=urllib.request.Request(r['source_url'],headers=headers)
    with urllib.request.urlopen(request,timeout=90) as response:blob=response.read()
    with Image.open(io.BytesIO(blob)) as im:im.verify()
    dest.with_suffix('.tmp').write_bytes(blob);dest.with_suffix('.tmp').replace(dest)
   with Image.open(dest) as im:width,height=im.size;im.verify()
   assert width>0 and height>0 and width*height<=100000000
   r.update(width=width,height=height,sha256=sha(dest),bytes=dest.stat().st_size,absolute_file=str(dest.resolve()))
   return r
  except Exception:
   if attempt==4:raise
   time.sleep(min(2**attempt,15))
def download(records,label):
 result=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for i,r in enumerate(pool.map(retrieve,records),1):
   result.append(r)
   if i%100==0:
    write(OUT/f'{label}_progress.json',dict(completed=i,planned=len(records)));print(label,i,'/',len(records),flush=True)
 write(OUT/f'{label}_manifest.json',dict(status='complete',samples=result))
 return result
archive=ROOT/'kvidence/datasets/shadow_chat/source/coco2017/annotations_trainval2017.zip'
with zipfile.ZipFile(archive) as z:
 text=z.read('annotations/instances_train2017.json').decode()
 dec=json.JSONDecoder();images,_=dec.raw_decode(text,text.index('[',text.index('"images"')));licenses,_=dec.raw_decode(text,text.index('[',text.index('"licenses"')))
lic={x['id']:x for x in licenses};random.Random(20261023).shuffle(images)
coco=[dict(image_id=f"{x['id']:012d}",source_dataset='coco2017',split='train',pilot=False,file=f"images/coco2017/{x['file_name']}",source_url='https://s3.amazonaws.com/images.cocodataset.org/train2017/'+x['file_name'],license=lic[x['license']],source_split='train2017') for x in images[:5000]]
del text,images
coco=download(coco,'coco')
artpath=HERE/'cleveland_source.json'
assert artpath.stat().st_size==344104666 and sha(artpath)=='1f94283644e170ac7b974568b82064d147362e7ff9db4018be9785631bd4e4a2'
raw=json.loads(artpath.read_text());items=raw if isinstance(raw,list) else raw['data']
arts=[]
for x in items:
 kind=x.get('type','');kind=' '.join(kind) if isinstance(kind,list) else str(kind)
 images=x.get('images') or {};web=images.get('web') or {}
 if kind.lower() not in ['painting','drawing'] or x.get('share_license_status')!='CC0' or not web.get('url'):continue
 arts.append(dict(image_id=str(x['id']),source_dataset='clevelandart',split='train',pilot=False,file=f"images/clevelandart/{x['id']}.jpg",source_url=web['url'],license='CC0',object_type=kind,object_title=x.get('title'),object_url=x.get('url'),source_split='deterministic experiment subset'))
assert len(arts)>=2000,('eligible artwork count',len(arts),list(items[0]))
arts.sort(key=lambda x:x['image_id']);random.Random(20261024).shuffle(arts);arts=download(arts[:2000],'art')
text_manifest=ROOT/'kvidence/datasets/hiertext/manifest.json';text=json.loads(text_manifest.read_text())
rows=[]
for r in text['samples']:
 r=r.copy();r['absolute_file']=str((text_manifest.parent/r['file']).resolve());r['source_dataset']='hiertext';rows.append(r)
rows+=coco+arts
ids=[(r['source_dataset'],r['image_id']) for r in rows];assert len(ids)==len(set(ids))
hashes=[r['sha256'] for r in rows];assert len(hashes)==len(set(hashes)),'Exact duplicate image source bytes'
assert sum(r['split']=='train' for r in rows)==10000
target_sha=json.loads((HERE/'experiment_plan.json').read_text())['target']['sha256'];assert target_sha not in hashes
write(OUT/'manifest.json',dict(status='complete',seed=20261023,samples=rows,counts=dict(train=10000,validation=300,test=300),training_composition=dict(hiertext=3000,coco2017=5000,clevelandart=2000),source_image_hash_overlap=0,source_urls=dict(HierText='https://github.com/google-research-datasets/hiertext',COCO='https://cocodataset.org/',CMA='https://github.com/ClevelandMuseumArt/openaccess'),art_metadata_sha256=sha(artpath),coco_annotation_archive_sha256=sha(archive),source_caveat='Validation and test are independent HierText images; primary endpoint is a separate private phone target. Do not claim domain-balanced test performance.'))
print('MIXED_SOURCES_COMPLETE',len(rows),flush=True)
