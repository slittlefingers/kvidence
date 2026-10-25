"""Wait for exactly 1000 captures, stop at that boundary, and publish authorized artifacts."""
import hashlib,json,shutil,subprocess,tarfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
DATA=HERE/'data/pilot_fixed991';PUB=HERE/'publish/fixed991_1000';REPO=ROOT/'kvidence/publications/gemma4-image-prefix-dataset'
TAG='v2026.10.23-fixed-prefix-1000';REMOTE='slittlefingers/gemma4-image-prefix-dataset'
PREFIX='data/20261023_fixed991_1000';URL=f'https://github.com/{REMOTE}/releases/tag/{TAG}'
PUB.mkdir(parents=True,exist_ok=True)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def status(stage,**kw):write(HERE/'build_status.json',dict(stage=stage,planned=1000,full_expansion_cancelled=True,target_upload_authorized=True,updated_unix=time.time(),**kw));print(stage,kw,flush=True)
def run(args,cwd=None):return subprocess.check_output(list(map(str,args)),cwd=cwd,text=True,stderr=subprocess.STDOUT)
def archive(name,entries):
 dest=PUB/name
 with tarfile.open(dest,'w:gz',compresslevel=1) as t:
  for src,arc in entries:t.add(src,arcname=arc)
 assert dest.stat().st_size<2*1024**3,'Asset exceeds GitHub limit'
 with tarfile.open(dest,'r:gz') as t:
  for m in t:
   assert m.name.startswith(PREFIX+'/') and not m.issym()
 return dict(name=name,bytes=dest.stat().st_size,sha256=sha(dest))
try:
 status('collecting_1000_then_publish')
 while True:
  try:p=json.loads((DATA/'protocol.json').read_text())
  except (FileNotFoundError,json.JSONDecodeError):p={}
  if p.get('status')=='complete':
   assert p['samples']==1000 and p['planned_samples']==1000;break
  log=HERE/'collection_1000.log'
  if log.exists() and 'Traceback' in log.read_text():raise RuntimeError('Collection failed; inspect collection_1000.log')
  time.sleep(5)
 status('validating_1000')
 print(run([HERE/'.venv/bin/python',HERE/'validate_fixed.py',DATA]),flush=True)
 index=json.loads((DATA/'index.json').read_text());assert len(index)==1000 and all(r['split']=='train' for r in index)
 source=json.loads((ROOT/'kvidence/datasets/hiertext/manifest.json').read_text());wanted={r['source_image_id'] for r in index};sources=[r for r in source['samples'] if r['image_id'] in wanted];assert len(sources)==1000
 core=PUB/'core';core.mkdir(exist_ok=True)
 notes=f'''固定前缀真实 NPU 数据集：1000 张训练来源图片与一个独立重点测试对象。

- 图片来自 HierText 官方训练集，共 1000 张，其中 {sum(r['has_visible_handwriting'] for r in sources)} 张包含符合筛选条件的手写文字。此 Release 不包含原计划的其他 9000 张图片。
- 图片前固定 991 tokens，上一轮完整 user/assistant 对话为 511 tokens。前缀是构造的实验对话，完整文本和 token IDs 随包提供；不是提取的实际受害者会话。
- Gemma 4 E2B Google Tensor G5，真实手机 CPU 视觉编码器/adapter，真实手机 NPU 第 5 层（索引 4）原始 INT8 K/V。量化参数随包提供。
- 每张图保存整张空间网格、K/V、768 维 adapter 前 FP32 特征、1536 维 adapter 后 FP32 embedding，以及 native STB 预处理后的 RGB 像素标签。
- 固定前缀快照重放与重新计算完整前缀在控制图片上逐位一致，RMSE=0 LSB。这个结论不覆盖其他分块或候选查询路径。
- 图片后的统一问题为“这个图片的 password 是什么？”。训练采集止于图片 K/V，后缀记录在元数据中；重点测试图片实际执行了 EOI 和后续问题。

重点测试对象：用户明确授权公开的手机白板手写照片。原图、同一固定前缀下的观测 K/V、以及独立的真实像素/embedding 对照放在 `important_test/whiteboard/`，不出现在训练 index.json 中。不得用目标原图、真实 embedding 或文字来训练或选择 checkpoint。目标文字尚未独立核对标注，不把模型猜测作为文字真值。

1000 张采集图片均来自官方 train split；本批没有另外采集验证/测试图片。需要调参时，应从这 1000 张中按原始图片 ID 留出验证集，重点白板图片始终独立。尚未训练恢复网络，未报告目标恢复成功。

下载两个训练分包、元数据/重点测试对象包和 SHA256SUMS：

```bash
gh release download {TAG} --repo {REMOTE}
shasum -a 256 -c SHA256SUMS
for f in gemma4-fixed991-1000-*.tar.gz; do tar -xzf "$f"; done
```

三个包都解压到 `{PREFIX}/`。训练数组位于 samples/；重点测试对象位于 important_test/whiteboard/。无需把重点测试目录传入训练加载器。

训练图片衍生数据遵循 HierText 官方 CC BY-SA 4.0 来源说明；附官方 README 和 LICENSE，不把模型权重或运行时随包分发。用户提供的重点测试照片经授权公开，不自动适用 HierText 的数据许可证。
'''
 (PUB/'RELEASE_NOTES.md').write_text(notes);(core/'README.md').write_text(notes)
 write(core/'source_metadata.json',dict(dataset='HierText',official_url='https://github.com/google-research-datasets/hiertext',source_split='train',samples=sources))
 for n in ['protocol.json','prefix.json','prefix.txt','index.json','quantization.json','acquisition_control.json','validation.json']:shutil.copy2(DATA/n,core/n)
 identity_path=max((DATA/'device').glob('*/device.json'),key=lambda p:p.stat().st_mtime)
 identity=json.loads(identity_path.read_text());write(core/'runtime_identity.json',{k:identity[k] for k in ['model_sha256','runtime_sha256','test_apk_sha256','native_mode','batch_signature','context_capacity','vocab_size','embedding_dim']})
 shutil.copytree(DATA/'source',core/'source',dirs_exist_ok=True)
 shutil.copy2(HERE/'validate_fixed.py',core/'source/validate_fixed.py')
 shutil.copy2(ROOT/'kvidence/datasets/hiertext/source/README.md',core/'HIERText_README.md');shutil.copy2(ROOT/'kvidence/datasets/hiertext/source/LICENSE',core/'HIERText_LICENSE')
 target=core/'important_test/whiteboard';target.mkdir(parents=True,exist_ok=True)
 private=HERE/'private_target';capture=private/'capture_fixed991'
 shutil.copy2(private/'PXL_20261022_233047650.jpg',target/'target.jpg')
 for n in ['observed.npz','evaluation_only_truth.npz']:shutil.copy2(capture/n,target/n)
 tm=json.loads((capture/'metadata.json').read_text());tm.update(publication_authorized=True,role='important held-out test case',training_included=False,text_ground_truth=None,files_sha256={n:sha(target/n) for n in ['target.jpg','observed.npz','evaluation_only_truth.npz']})
 assert tm['source_sha256']==sha(target/'target.jpg') and tm['source_sha256'] not in {r['source_sha256'] for r in index};write(target/'metadata.json',tm)
 (target/'README.md').write_text('Important independent test object, explicitly authorized for public upload. target.jpg is the original phone image. observed.npz contains only image-position KV and public geometry. evaluation_only_truth.npz contains real embedding/pixel controls and must never be used as training input or for checkpoint selection. metadata.json records the frozen prefix and model identity. Text transcription has not been independently verified.\n')
 status('packaging_1000')
 assets=[]
 for k in range(2):
  entries=[]
  for r in index[k*500:(k+1)*500]:
   path=DATA/r['file'];entries.extend([(path,f'{PREFIX}/{r["file"]}'),(path.with_suffix('.json'),f'{PREFIX}/{Path(r["file"]).with_suffix(".json")}')])
  assets.append(archive(f'gemma4-fixed991-1000-train-{k+1:02d}.tar.gz',entries));print('PACKAGED',assets[-1],flush=True)
 assets.append(archive('gemma4-fixed991-1000-metadata-target.tar.gz',[(p,f'{PREFIX}/{p.name}') for p in sorted(core.iterdir())]));print('PACKAGED',assets[-1],flush=True)
 (PUB/'SHA256SUMS').write_text(''.join(f"{a['sha256']}  {a['name']}\n" for a in assets))
 write(PUB/'publication.json',dict(tag=TAG,url=URL,assets=assets,status='prepared'))
 status('publishing_1000')
 print(run(['git','pull','--ff-only'],REPO),flush=True)
 doc=REPO/'datasets/20261023_fixed991_1000';doc.mkdir(parents=True,exist_ok=True)
 for n in ['README.md','protocol.json','prefix.json','prefix.txt','index.json','quantization.json','acquisition_control.json','validation.json','source_metadata.json','runtime_identity.json']:shutil.copy2(core/n,doc/n)
 for n in ['SHA256SUMS','publication.json']:shutil.copy2(PUB/n,doc/n)
 shutil.copytree(core/'source',doc/'source',dirs_exist_ok=True)
 rt=REPO/'important_test/whiteboard';rt.mkdir(parents=True,exist_ok=True)
 for n in ['target.jpg','metadata.json','README.md']:shutil.copy2(target/n,rt/n)
 rp=REPO/'README.md';text=rp.read_text();heading='## 2026-10-23 固定前缀 1000 张与重点测试图片'
 if heading not in text:
  section=f'''{heading}

固定 **991-token 前缀**（上一轮对话 511 tokens）的真实手机 NPU 数据，包含 **1000 张 HierText 训练来源图片**。与此前变化前缀的数据集不同。本版尚未训练恢复网络。

[下载数据与重点测试观测]({URL}) · [数据说明与解压步骤](datasets/20261023_fixed991_1000/README.md)

重点测试对象是单独保留的白板手写照片，用户已授权公开。它及其真实特征不得用于训练或选择模型；[测试对象说明](important_test/whiteboard/README.md)。

![重点测试对象：白板手写照片](important_test/whiteboard/target.jpg)

'''
  marker='## 2026-10-23 数据扩充';text=text.replace(marker,section+marker,1) if marker in text else text+'\n'+section;rp.write_text(text)
 print(run(['git','add','README.md','datasets/20261023_fixed991_1000','important_test/whiteboard'],REPO),flush=True)
 print(run(['git','commit','-m','Publish fixed-prefix 1000-image dataset and held-out whiteboard target'],REPO),flush=True)
 commit=run(['git','rev-parse','HEAD'],REPO).strip();print(run(['git','push','origin','main'],REPO),flush=True)
 print(run(['gh','release','create',TAG,'--repo',REMOTE,'--target',commit,'--title','Gemma 4 Fixed Prefix: 1000 Images + Important Whiteboard Test','--notes-file',PUB/'RELEASE_NOTES.md','--draft','--prerelease']),flush=True)
 print(run(['gh','release','upload',TAG,'--repo',REMOTE,*[PUB/a['name'] for a in assets],PUB/'SHA256SUMS']),flush=True)
 remote=json.loads(run(['gh','release','view',TAG,'--repo',REMOTE,'--json','assets,isDraft,url']));got={a['name']:a for a in remote['assets']}
 for a in assets:
  r=got[a['name']];assert r['size']==a['bytes'] and r['state']=='uploaded' and r['digest']=='sha256:'+a['sha256']
 assert got['SHA256SUMS']['digest']=='sha256:'+sha(PUB/'SHA256SUMS')
 print(run(['gh','release','edit',TAG,'--repo',REMOTE,'--draft=false']),flush=True)
 remote=json.loads(run(['gh','release','view',TAG,'--repo',REMOTE,'--json','isDraft,url']));assert not remote['isDraft']
 write(PUB/'publication.json',dict(tag=TAG,url=URL,assets=assets,status='published',remote_sha256_verified=True,commit=commit,target_upload_authorized=True))
 status('complete_uploaded_1000',url=URL,train_images=1000,target_separate=True,remote_sha256_verified=True)
except BaseException as e:
 status('failed_1000_publish',error=str(e));raise
