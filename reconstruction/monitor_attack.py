#!/usr/bin/env python3
"""Relay real per-batch events to Ghost; does not start/stop or score recovery.

python monitor_attack.py --run results/my_run_or_comparison --serial SERIAL
Start before or during a run. It also publishes completed runs; --once exits
once currently available events are copied. No network service is exposed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
from kvrecon.dashboard import DashboardState

PKG='com.example.annoymous'


def complete_lines(path,offset):
    with path.open('rb') as f:
        f.seek(offset);raw=f.read(4<<20)
    end=raw.rfind(b'\n')+1
    return raw[:end]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--adb',default='/path/to/user/Library/Android/sdk/platform-tools/adb')
    p.add_argument('--serial');p.add_argument('--interval',type=float,default=1.)
    p.add_argument('--once',action='store_true')
    args=p.parse_args()
    if args.interval<.25:raise ValueError('Refresh interval must be at least 0.25 seconds')
    adb=[args.adb]+(['-s',args.serial] if args.serial else [])
    def shell(command,data=b''):
        subprocess.run(adb+['shell','-T',shlex.join(['run-as',PKG,'sh','-c',command])],
            input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=15)
    source=None;offset=0;state=None
    print('Watching '+str(args.run.resolve()),flush=True)
    try:
        while True:
            sources=list(args.run.rglob('events.jsonl')) if args.run.exists() else []
            newest=max(sources,key=lambda x:x.stat().st_mtime_ns) if sources else None
            if newest is not None and newest!=source:
                source=newest;offset=0
                # New relay session gets its own archive: reconnect cannot duplicate events.
                run_id=hashlib.sha256((str(source.resolve())+str(time.time_ns())).encode()).hexdigest()[:20]
                decode=None
                protocol=source.parent/'protocol.json'
                if protocol.exists():
                    from tokenizers import Tokenizer
                    config=json.loads(protocol.read_text())
                    tokenizer=Tokenizer.from_file(config['model_config']['tokenizer'])
                    decode=lambda ids:tokenizer.decode(ids,skip_special_tokens=False)
                state=DashboardState(run_id,str(source.parent.relative_to(args.run.parent)),decode)
                archive=f'files/attack_runs/{run_id}'
                shell(f'mkdir -p {archive}; : > {archive}/events.jsonl')
                print('Publishing '+str(source),flush=True)
            if state:
                raw=complete_lines(source,offset)
                # Advance only after a successful device append. A failed ADB command
                # stops the relay (unknown append outcome); restarting uses a new archive.
                if raw:
                    events=[json.loads(line) for line in raw.splitlines()]
                    for event in events:
                        if event['type']=='batch':event['pieces']=[state.decode([t]) for t in event['token_ids']]
                    archived=''.join(json.dumps(e,ensure_ascii=False,separators=(',',':'))+'\n' for e in events).encode()
                    shell(f'cat >> {archive}/events.jsonl',archived)
                    for event in events:state.apply(event)
                    offset+=len(raw)
                evaluation=source.parent/'evaluation.json'
                if evaluation.exists() and state.data['status'] not in ('waiting','running'):
                    try:state.data['evaluation']=json.loads(evaluation.read_text())
                    except json.JSONDecodeError:pass
                snapshot=state.snapshot()
                snapshot['archive']=f'attack_runs/{state.data["run_id"]}/events.jsonl'
                encoded=json.dumps(snapshot,ensure_ascii=False,allow_nan=False).encode()
                shell('cat > files/attack_live.tmp && mv files/attack_live.tmp files/attack_live.json',encoded)
                if args.once and offset==source.stat().st_size:return
            elif args.once:raise FileNotFoundError('No events.jsonl yet; enable --live-events on run/compare')
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('Relay stopped; recovery is unaffected. Ghost will show a stale indicator.',flush=True)


if __name__=='__main__':main()
