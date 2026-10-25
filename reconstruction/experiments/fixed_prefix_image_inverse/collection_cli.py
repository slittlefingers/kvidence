"""Terminal client for the local acquisition controller."""
import argparse
import json
from pathlib import Path
import sys
import subprocess
import time
import urllib.request
import urllib.error
from collection_profiles import HERE

ap = argparse.ArgumentParser(description='NPU collection controls shared by the terminal and Ghost')
sub = ap.add_subparsers(dest='command', required=True)
sub.add_parser('serve', help='Keep the collection service running in this terminal')
sub.add_parser('status');sub.add_parser('pause');sub.add_parser('resume');sub.add_parser('profiles')
p = sub.add_parser('use');p.add_argument('name')
p = sub.add_parser('create');p.add_argument('name');p.add_argument('--messages', type=Path, required=True)
p.add_argument('--limit', type=int, default=10000);p.add_argument('--suffix', default='这个图片的 password 是什么？')
sub.add_parser('watch')
sub.add_parser('phone', help='Connect the Ghost collection screen over USB')
args = ap.parse_args()
if args.command == 'serve':
    sys.exit(subprocess.call(['/usr/bin/caffeinate','-i','-s',sys.executable,str(HERE / 'collection_panel.py')]))
try:
    connection=json.loads((HERE/'panel_connection.json').read_text())
    def call(endpoint, payload=None):
        req=urllib.request.Request(connection['url']+'/api/'+endpoint,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'X-Control-Token':connection['token'],'Content-Type':'application/json'})
        return json.load(urllib.request.urlopen(req,timeout=10))
    def show():
        s=call('status')
        print(f"{s['profile']}  {s['completed']}/{s['total']}  {s['mode']}  cooldown {s['cooldown']} s",flush=True)
        if s.get('error'):print(s['error'])
    if args.command=='phone':
        adb='/path/to/user/Library/Android/sdk/platform-tools/adb'
        cmd=[adb,'-s','YOUR_DEVICE_SERIAL']
        subprocess.run(cmd+['reverse','tcp:8765','tcp:8765'],check=True)
        subprocess.run(cmd+['shell','mkdir','-p','/sdcard/Android/data/com.example.annoymous/files'],check=True)
        subprocess.run(cmd+['push',str(HERE/'panel_connection.json'),'/sdcard/Android/data/com.example.annoymous/files/collection_connection.json'],check=True)
        subprocess.run(cmd+['shell','am','start','-n','com.example.annoymous/.CollectionActivity'],check=True)
    elif args.command=='status':show()
    elif args.command=='watch':
        while True:show();time.sleep(5)
    elif args.command=='profiles':
        for job in call('profiles'):print(json.dumps(job,ensure_ascii=False))
    elif args.command=='use':print(call('profile',dict(name=args.name)))
    elif args.command=='create':
        print(call('create',dict(name=args.name,messages=json.loads(args.messages.read_text()),limit=args.limit,suffix=args.suffix)))
    else:print(call(args.command,{}))
except KeyboardInterrupt:pass
except urllib.error.HTTPError as e:
    print(e.read().decode(),file=sys.stderr);sys.exit(1)
except (OSError,ValueError) as e:
    print(f'Collection service unavailable: {e}. Start ./collectctl serve first',file=sys.stderr);sys.exit(1)
