"""Run the standalone probe under Ghost's UID using its installed native libs."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parent
PKG='com.example.annoymous'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--adb',default=str(Path.home()/'Library/Android/sdk/platform-tools/adb'))
    p.add_argument('--test-apk',type=Path,default=ROOT.parents[2]/'annoymous/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk')
    p.add_argument('--input-dir',type=Path,default=ROOT/'inputs')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    def adb(*a,**kw):return subprocess.run([args.adb,*a],check=True,**kw)
    def shell(*a,**kw):return adb('shell',shlex.join(a),**kw)
    apk=shell('pm','path',PKG,stdout=subprocess.PIPE).stdout.decode().strip().removeprefix('package:')
    lib=apk.rsplit('/',1)[0]+'/lib/arm64'
    name='candidate-probe-'+time.strftime('%Y%m%d-%H%M%S')
    remote='files/'+name
    adb('install','-r','-t',str(args.test_apk))
    shell('run-as',PKG,'mkdir','-p',remote+'/inputs',remote+'/results')
    for filename in ('probe.json','kv_payload.bin'):
        with (args.input_dir/filename).open('rb') as f:
            command=shlex.join(['run-as',PKG,'sh','-c','cat > '+shlex.quote(remote+'/inputs/'+filename)])
            adb('exec-in',command,stdin=f)
    command=['am','instrument','-w','-r','-e','class',PKG+'.CandidateBatchProbeDeviceTest','-e','probe_job',name,PKG+'.test/androidx.test.runner.AndroidJUnitRunner']
    input_bytes=(args.input_dir/'probe.json').read_bytes()
    (args.output/'input.json').write_bytes(input_bytes)
    provenance=dict(command=command,remote_directory=remote,test_apk_sha256=hashlib.sha256(args.test_apk.read_bytes()).hexdigest(),input_sha256=hashlib.sha256(input_bytes).hexdigest(),source_sha256=hashlib.sha256((ROOT/'probe.cc').read_bytes()).hexdigest(),runtime_sha256=shell('run-as',PKG,'sha256sum',lib+'/liblitert-lm.so',stdout=subprocess.PIPE).stdout.decode().split()[0])
    (args.output/'submission.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Running isolated NPU probe:',name,flush=True)
    start=time.monotonic()
    with (args.output/'device.log').open('wb') as log:
        proc=subprocess.run([args.adb,'shell',shlex.join(command)],stdout=log,stderr=subprocess.STDOUT,timeout=600)
    print('Device exit:',proc.returncode,'wall seconds:',round(time.monotonic()-start,3),flush=True)
    with (args.output/'results.tar').open('wb') as out:
        adb('exec-out',shlex.join(['run-as',PKG,'tar','-cf','-','-C',remote+'/results','.']),stdout=out)
    import tarfile
    with tarfile.open(args.output/'results.tar') as archive:
        archive.extractall(args.output/'data',filter='data')
    if proc.returncode or 'OK (1 test)' not in (args.output/'device.log').read_text():raise RuntimeError('Probe failed; inspect device.log and data/native.log. Device artifacts retained.')
    result=json.loads((args.output/'data/report.json').read_text())
    if not result.get('completed'):raise RuntimeError('Incomplete report')
    print('Completed prefixes:',[x['prefix_length'] for x in result['cases']],flush=True)
    # Retain the isolated phone directory until results have been reviewed.

if __name__=='__main__':main()
