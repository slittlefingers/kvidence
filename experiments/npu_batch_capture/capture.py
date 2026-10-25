"""Submit JSON prompts to Ghost's on-device NPU batch capture runner.

Each sample gets a fresh Session. Capture is AFTER response generation. The
host downloads and verifies one losslessly compressed raw export at a time.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[2]
APP=WORKSPACE/'annoymous'
PACKAGE='com.example.annoymous'
RUNNER=PACKAGE+'.test/androidx.test.runner.AndroidJUnitRunner'
CLASS=PACKAGE+'.BatchKvCaptureDeviceTest'
SAFE=re.compile(r'[A-Za-z0-9_-]{1,64}\Z')


def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def load_samples(path):
    text=path.read_text(encoding='utf-8')
    if path.suffix.lower()=='.jsonl':
        rows=[json.loads(x) for x in text.splitlines() if x.strip()]
    else:
        obj=json.loads(text)
        rows=obj if isinstance(obj,list) else obj.get('samples',[obj])
    if not isinstance(rows,list) or not rows:
        raise ValueError('Expected a non-empty JSON array, {samples:[...]}, or JSONL')
    seen=set();result=[]
    for i,item in enumerate(rows):
        if not isinstance(item,dict):raise ValueError(f'Sample {i} is not an object')
        sid=item.get('id',f'{i+1:04d}')
        if not isinstance(sid,str) or not SAFE.fullmatch(sid) or sid in seen:
            raise ValueError(f'Invalid or duplicate id at sample {i}')
        prompt=item.get('user_prompt')
        if not isinstance(prompt,str) or not prompt or '\0' in prompt:
            raise ValueError(f'Sample {sid} needs a nonempty user_prompt without NUL')
        prompt.encode('utf-8')  # reject unpaired Unicode surrogates before submission
        row=dict(id=sid,user_prompt=prompt)
        if 'media' in item:
            media=item['media']
            if not isinstance(media,dict) or media.get('type') not in ('image','audio'):
                raise ValueError(f'Sample {sid}: media must be one image or audio object')
            relative=media.get('path')
            if not isinstance(relative,str) or not relative or Path(relative).is_absolute():
                raise ValueError(f'Sample {sid}: media.path must be relative to the input file')
            local=(path.parent/relative).resolve()
            if not local.is_relative_to(path.parent.resolve()) or not local.is_file():
                raise ValueError(f'Sample {sid}: missing media or path outside dataset')
            if not 0<local.stat().st_size<=20*1024*1024:
                raise ValueError(f'Sample {sid}: media must be nonempty and at most 20 MiB')
            if media.get('sha256')!=sha(local):
                raise ValueError(f'Sample {sid}: media SHA-256 mismatch')
            if item.get('content_order')!=['media','text']:
                raise ValueError(f'Sample {sid}: content_order must be [media, text]')
            row.update(media=dict(type=media['type'],path=relative,sha256=media['sha256']),
                       content_order=['media','text'])
        seen.add(sid);result.append(row)
    return result


def validate_capture(path,meta):
    expected=meta['kv']
    if path.stat().st_size!=expected['compressed_bytes'] or sha(path)!=expected['gzip_sha256']:
        raise ValueError('Compressed KV size/SHA-256 mismatch')
    with gzip.open(path,'rb') as f:
        header=f.read(16)
        if len(header)!=16 or header[:8]!=b'GHOSTKV1':raise ValueError('Invalid GHOSTKV1 header')
        length=struct.unpack('<Q',header[8:])[0]
        if not 0<length<16*1024*1024:raise ValueError('Invalid layout length')
        raw_manifest=f.read(length)
        if len(raw_manifest)!=length:raise ValueError('Truncated manifest')
        layout=json.loads(raw_manifest)
        offset=0;names=set();layers=set()
        for b in layout['buffers']:
            if b['name'] in names:raise ValueError('Duplicate buffer')
            names.add(b['name']);layers.add(b['layer'])
            if b['payload_offset']!=offset or b['allocated_bytes']<=0:raise ValueError('Invalid buffer range')
            if len(b['shape'])!=len(b['strides']) or any(x<=0 for x in b['shape']):raise ValueError('Invalid tensor layout')
            last=sum((n-1)*s for n,s in zip(b['shape'],b['strides']))
            if any(s<0 for s in b['strides']) or (last+1)*b['element_bytes']>b['allocated_bytes']:
                raise ValueError('Tensor components exceed exported allocation')
            offset+=b['allocated_bytes']
        if not names:raise ValueError('No KV buffers exported')
        count=0
        while block:=f.read(1024*1024):count+=len(block)
        if count!=offset or count+16+length!=expected['uncompressed_bytes']:
            raise ValueError('Truncated or oversized raw KV payload')
    return layout,dict(passed=True,buffer_count=len(names),exported_layer_ids=sorted(layers),
        uncompressed_bytes=count+16+length,gzip_sha256=expected['gzip_sha256'],
        gzip_crc_checked=True,all_buffer_ranges_checked=True,
        note='All exported allocations verified; this does not prove all historical token positions remain after sliding-window eviction.')


class Adb:
    def __init__(self,path,serial):
        self.command=[path]+(['-s',serial] if serial else [])
    def run(self,*args,**kwargs):
        return subprocess.run(self.command+list(args),check=True,**kwargs)
    def shell(self,*args):
        return self.run('shell',shlex.join(args),stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout
    def app(self,*args):return self.shell('run-as',PACKAGE,*args)
    def put(self,path,data):
        command=shlex.join(['run-as',PACKAGE,'sh','-c','cat > '+shlex.quote(path)])
        self.run('exec-in',command,input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    def get(self,remote,local):
        temp=local.with_suffix(local.suffix+'.part')
        with temp.open('wb') as f:
            self.run('exec-out',shlex.join(['run-as',PACKAGE,'cat',remote]),stdout=f,stderr=subprocess.PIPE)
            f.flush();os.fsync(f.fileno())
        temp.replace(local)
    def json(self,path):
        command=shlex.join(['run-as',PACKAGE,'cat',path])
        # shell protocol carries the remote exit status; exec-out may return 0
        # for a missing remote file and put the error in the output stream.
        p=subprocess.run(self.command+['shell',command],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if p.returncode:return None
        return json.loads(p.stdout)


def run(args):
    samples=load_samples(args.input)
    if args.limit is not None:
        if args.limit<=0:raise ValueError('--limit must be positive')
        samples=samples[:args.limit]
    if args.model!='gemma4-e2b' and any('media' in s for s in samples):
        raise ValueError('This multimodal capture path requires gemma4-e2b')
    uploads=[]
    for sample in samples:
        if 'media' not in sample:continue
        media=sample['media'];local=(args.input.parent/media['path']).resolve()
        # Device paths are generated by the host, never taken from external data.
        remote_name=f"media/{sample['id']}{'.jpg' if media['type']=='image' else '.wav'}"
        uploads.append((local,remote_name))
        media['path']=remote_name
    job_id='kv-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
    job=dict(schema_version=1,job_id=job_id,model=args.model,capture_stage='after_response',
             token_id_scrubbing=True,samples=samples)
    if args.prepare_only:
        args.output.mkdir(parents=True,exist_ok=False)
        atomic_json(args.output/'input.json',job)
        print(f'Prepared {len(samples)} samples; no device used.');return
    adb_path=args.adb or shutil.which('adb')
    if not adb_path:
        candidate=Path.home()/'Library/Android/sdk/platform-tools/adb'
        if candidate.is_file():adb_path=str(candidate)
    if not adb_path:raise FileNotFoundError('adb not found; use --adb')
    adb=Adb(adb_path,args.serial)
    adb.run('get-state',stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    adb.app('pwd')  # fail before creating a job if target APK is not debuggable
    if not args.no_install_test:
        if not args.test_apk.is_file():raise FileNotFoundError('Build the companion test APK with ./gradlew :app:assembleDebugAndroidTest')
        adb.run('install','-r','-t',str(args.test_apk))
    args.output.mkdir(parents=True,exist_ok=False)
    atomic_json(args.output/'input.json',job)
    remote='files/kv_jobs/'+job_id
    adb.app('mkdir','-p',remote)
    if uploads:
        adb.app('mkdir','-p',remote+'/media')
        for local,remote_name in uploads:
            adb.put(remote+'/'+remote_name,local.read_bytes())
    adb.put(remote+'/input.json',(args.output/'input.json').read_bytes())
    atomic_json(args.output/'submission.json',dict(job_id=job_id,source_input=str(args.input.resolve()),
        source_sha256=sha(args.input),sample_count=len(samples),remote_directory=remote,
        capture_stage='after_response',max_output_tokens=512,
        test_apk_sha256=sha(args.test_apk) if args.test_apk.is_file() else None))
    instrument=['am','instrument','-w','-r','-e','class',CLASS,'-e','batch_job',job_id,RUNNER]
    completed=[];downloaded=set();last_status=None;start=time.monotonic()
    print(f'Job {job_id}: {len(samples)} independent prompts, NPU, capture after response.',flush=True)
    print('Instrumentation starts the Ghost app process; the existing interactive Session is not retained.',flush=True)
    with (args.output/'instrumentation.log').open('wb') as log:
        process=subprocess.Popen(adb.command+['shell',shlex.join(instrument)],stdout=log,stderr=subprocess.STDOUT)
        try:
            while True:
                status=adb.json(remote+'/status.json')
                if status:
                    atomic_json(args.output/'status.json',status)
                    state=(status['state'],status['completed'],status['sample_id'])
                    if state!=last_status:
                        print(f"{status['state']}: {status['completed']}/{len(samples)} {status['sample_id']}",flush=True)
                        last_status=state
                    if status['state']=='failed':raise RuntimeError(status.get('error','Device capture failed'))
                if not (args.output/'job.json').exists():
                    info=adb.json(remote+'/job.json')
                    if info:atomic_json(args.output/'job.json',info)
                ready=adb.json(remote+'/ready.json')
                if ready and ready['id'] not in downloaded:
                    sid=ready['id']
                    if sid!=samples[len(completed)]['id']:raise RuntimeError('Out-of-order or unexpected sample')
                    dest=args.output/sid;dest.mkdir()
                    for name in ('sample.json','response.txt','all_layers.gkv.gz'):
                        adb.get(remote+'/'+sid+'/'+name,dest/name)
                    meta=json.loads((dest/'sample.json').read_text())
                    if meta['id']!=sid or meta['capture_stage']!='after_response':raise ValueError('Wrong sample metadata')
                    prompt=samples[len(completed)]['user_prompt']
                    if meta['prompt_sha256']!=hashlib.sha256(prompt.encode('utf-8')).hexdigest():raise ValueError('Prompt hash mismatch')
                    if meta['response_sha256']!=sha(dest/'response.txt'):raise ValueError('Response hash mismatch')
                    expected_sample=samples[len(completed)]
                    if 'media' in expected_sample:
                        if meta.get('media')!=expected_sample['media'] or meta.get('content_order')!=['media','text']:
                            raise ValueError('Media identity/order mismatch in device capture')
                    layout,validation=validate_capture(dest/'all_layers.gkv.gz',meta)
                    atomic_json(dest/'layout.json',layout);atomic_json(dest/'validation.json',validation)
                    completed.append(dict(id=sid,**validation,inference_seconds=meta['inference_seconds']))
                    downloaded.add(sid)
                    atomic_json(args.output/'progress.json',dict(completed=completed))
                    # Acknowledgement is permission to delete this sample on the phone.
                    adb.put(remote+'/ack-'+sid,b'verified\n')
                    print(f"VERIFIED {sid}: {validation['buffer_count']} buffers, {len(validation['exported_layer_ids'])} layer IDs, {(dest/'all_layers.gkv.gz').stat().st_size/2**20:.1f} MiB gzip",flush=True)
                if status and status['state']=='done':break
                if process.poll() is not None:
                    final=adb.json(remote+'/status.json')
                    if final and final['state']=='done':break
                    raise RuntimeError('Instrumentation exited before completion; inspect instrumentation.log')
                if time.monotonic()-start>args.timeout:
                    raise TimeoutError('Host timeout; completed samples are retained')
                time.sleep(1)
            process.wait(timeout=60)
            if process.returncode!=0 or len(completed)!=len(samples):raise RuntimeError('Incomplete batch')
            atomic_json(args.output/'summary.json',dict(passed=True,job_id=job_id,sample_count=len(samples),
                completed_count=len(completed),capture_stage='after_response',backend='npu',
                wall_seconds=time.monotonic()-start,results=completed,
                independent_sessions=True,transport='gzip GHOSTKV1 original allocations plus JSON layout'))
            # Remove uploaded prompts and device metadata after a fully verified run.
            adb.app('rm','-r',remote)
            print(f'COMPLETE: {len(completed)} verified captures in {args.output}',flush=True)
        except BaseException:
            try:adb.put(remote+'/cancel',b'cancel\n')
            except Exception:pass
            if process.poll() is None:
                # Cancel is cooperative between native calls. Do not force-stop a
                # shared app behind the user's back; the device job will see it.
                process.terminate()
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'short100.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=['gemma4-e2b','gemma3-1b'],default='gemma4-e2b')
    p.add_argument('--limit',type=int)
    p.add_argument('--serial')
    p.add_argument('--adb')
    p.add_argument('--timeout',type=float,default=12*3600)
    p.add_argument('--test-apk',type=Path,default=APP/'app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk')
    p.add_argument('--no-install-test',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    run(p.parse_args())
