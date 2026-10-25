"""Select real text images using official splits; never use target pixels/text for selection."""
import argparse,gzip,hashlib,json,math,random,tarfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DATA=ROOT/'kvidence/datasets/hiertext'
SEED=20261023

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def inspect(a):
    w,h=a['image_width'],a['image_height']
    factor=math.sqrt(2520*256/(w*h))
    nw=max(48,math.floor(factor*w/48)*48);nh=max(48,math.floor(factor*h/48)*48)
    if nw*nh>2520*256:return None
    visible=[]
    for p in a['paragraphs']:
        for l in p['lines']:
            for word in l['words']:
                text=word.get('text','')
                if not word.get('legible') or len(text)<3 or not any(c.isalnum() for c in text):continue
                v=[(x*nw/w,y*nh/h) for x,y in word['vertices']]
                # Minimum cyclic polygon edge is a conservative height proxy for quadrilateral words.
                edges=[math.dist(v[i],v[(i+1)%len(v)]) for i in range(len(v))]
                height=min(edges)
                if height>=12:
                    visible.append(dict(text=text,vertices=word['vertices'],handwritten=word.get('handwritten',False),estimated_min_edge_after_resize=height))
    if not visible:return None
    return dict(image_id=a['image_id'],width=w,height=h,estimated_native_width=nw,estimated_native_height=nh,estimated_token_grid=[nh//48,nw//48],has_visible_handwriting=any(x['handwritten'] for x in visible),evaluation_words=visible)

def select():
    records=[];stats={}
    for si,(split,count) in enumerate([('train',3000),('validation',300),('test',300)]):
        with gzip.open(DATA/'source'/f'{split}.jsonl.gz','rt') as f:source=json.load(f)
        eligible=[r for a in source['annotations'] if (r:=inspect(a)) is not None]
        hand=sorted([r for r in eligible if r['has_visible_handwriting']],key=lambda r:r['image_id'])
        printed=sorted([r for r in eligible if not r['has_visible_handwriting']],key=lambda r:r['image_id'])
        rng=random.Random(SEED+si);rng.shuffle(hand);rng.shuffle(printed)
        nh=min(round(count*.3),len(hand));np=count-nh
        assert len(printed)>=np,(split,len(eligible))
        chosen=hand[:nh]+printed[:np];rng.shuffle(chosen)
        for i,r in enumerate(chosen):r.update(split=split,file=f'images/{split}/{r["image_id"]}.jpg',pilot=i<(1000 if split=='train' else 100))
        records.extend(chosen)
        stats[split]=dict(source_images=len(source['annotations']),eligible=len(eligible),eligible_handwritten=len(hand),selected=len(chosen),selected_handwritten=nh,pilot=sum(r['pilot'] for r in chosen))
        print(split,stats[split],flush=True)
        del source,eligible,hand,printed
    ids=[r['image_id'] for r in records];assert len(ids)==len(set(ids))
    write(DATA/'selection.json',dict(schema=1,status='selected_not_extracted',seed=SEED,source='https://github.com/google-research-datasets/hiertext',license='CC BY-SA 4.0 per official repository',annotation_sha256={s:sha(DATA/'source'/f'{s}.jsonl.gz') for s in stats},selection_rule='At least one legible alphanumeric word of length >=3 with minimum polygon edge estimated >=12 px after native-size resize. Up to 30% of selected images have visible handwritten words. Native preprocessing visibility is an estimate, not proven OCR legibility.',target_used_for_selection=False,source_splits_preserved=True,all_text_strings_disjoint=False,stats=stats,samples=records))

def extract():
    from PIL import Image
    path=DATA/'selection.json';d=json.loads(path.read_text());target=json.loads((HERE/'experiment_plan.json').read_text())['target']['sha256']
    archive_meta={}
    for split in ['train','validation','test']:
        selected={r['image_id']:r for r in d['samples'] if r['split']==split}
        archive=DATA/'source'/f'{split}.tgz';found=set()
        with tarfile.open(archive,'r|gz') as t:
            for m in t:
                iid=Path(m.name).stem
                if not m.isfile() or iid not in selected:continue
                r=selected[iid];dest=DATA/r['file'];dest.parent.mkdir(parents=True,exist_ok=True)
                content=t.extractfile(m).read();assert content[:2]==b'\xff\xd8'
                digest=hashlib.sha256(content).hexdigest();assert digest!=target
                dest.write_bytes(content)
                with Image.open(dest) as im:
                    assert im.size==(r['width'],r['height']),(iid,im.size,r['width'],r['height'])
                    im.verify()
                r.update(sha256=digest,bytes=len(content),source_archive_member=m.name)
                found.add(iid)
        assert found==set(selected),(split,len(found),len(selected))
        archive_meta[split]=dict(url=f'https://open-images-dataset.s3.amazonaws.com/ocr/{split}.tgz',bytes=archive.stat().st_size,sha256=sha(archive))
        print('EXTRACTED',split,len(found),flush=True)
    hashes=[r['sha256'] for r in d['samples']];assert len(hashes)==len(set(hashes)),'Duplicate source image bytes'
    d.update(status='complete',archives=archive_meta,checks=dict(jpeg_verified=True,dimensions_match_annotations=True,source_image_id_overlap=0,source_image_sha256_overlap=0,target_sha256_overlap=0))
    write(DATA/'manifest.json',d)
    for split in ['train','validation','test']:
        (DATA/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in d['samples'] if r['split']==split))
    write(DATA/'preparation_summary.json',{k:v for k,v in d.items() if k!='samples'})
    print('COMPLETE',len(hashes),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['select','extract']);args=parser.parse_args()
    (select if args.phase=='select' else extract)()
