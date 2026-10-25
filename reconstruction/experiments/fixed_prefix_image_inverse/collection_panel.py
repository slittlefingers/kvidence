"""Loopback-only controls for the existing fixed-prefix acquisition job."""
import json
import os
from pathlib import Path
import secrets
import shlex
import signal
import subprocess
import threading
import time
from collection_profiles import load, listing, create
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / 'data/full_fixed991'
STATE = HERE / 'panel_state.json'
TOKEN = secrets.token_urlsafe(24)
LOCK = threading.Lock()
PORT = 8765


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def processes():
    rows = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,command='], text=True)
    found = {}
    for row in rows.splitlines():
        parts = row.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            args = shlex.split(parts[2])
        except ValueError:
            continue
        # Match Python's script argument, never a shell command mentioning it.
        if len(args) > 1 and Path(args[0]).name.lower().startswith('python'):
            name = Path(args[1]).name
            if name in ('collect_500_batches.py', 'collect_fixed_10k.py', 'run_profile.py', 'collect_configurable.py'):
                found[int(parts[0])] = (int(parts[1]), name)
    return found


control = read(STATE)
control.setdefault('profile', 'fixed991')
control.setdefault('mode', 'running')
control.setdefault('resume_after', 0)
control.pop('error', None)
def current_out():
    return OUT if control['profile']=='fixed991' else Path(load(control['profile'])['out'])
def current_build():
    return HERE/'build_status.json' if control['profile']=='fixed991' else current_out()/'build_status.json'
def total():
    return 10000 if control['profile']=='fixed991' else load(control['profile'])['limit']
def request_pause(value):
    p=HERE/'job_control.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(pause_requested=value)));t.replace(p)
children = []


def save():
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(control, indent=2) + '\n')
    temp.replace(STATE)


def send(pid, sig):
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def pause():
    with LOCK:
        control.update(mode='pausing', error=None)
        save()
        jobs = processes()
        parents = [p for p, (_, n) in jobs.items() if n in ('collect_500_batches.py','run_profile.py')]
        workers = [p for p, (_, n) in jobs.items() if n in ('collect_fixed_10k.py','collect_configurable.py')]
        # Prevent the batch supervisor from racing with a manual stop.
        cooperative=control['profile']!='fixed991'
        request_pause(True)
        if not cooperative:
            for p in parents:
                send(p, signal.SIGSTOP)
        try:
            if not cooperative:
                for p in workers:
                    send(p, signal.SIGINT)
            deadline = time.time() + (900 if cooperative else 120)
            while any(p in processes() for p in workers):
                if time.time() >= deadline:
                    raise RuntimeError('Worker is still closing. Retry pause; collection remains stopped.')
                time.sleep(.5)
        except Exception as e:
            control.update(mode='error', error=str(e))
            save()
            return
        for p in parents:
            send(p, signal.SIGTERM)
            send(p, signal.SIGCONT)
        while any(p in processes() for p in parents):
            time.sleep(.2)
        # Never shorten an existing thermal rest. Every manual stop also rests 5 min.
        control.update(mode='paused', resume_after=max(time.time() + 300,
                       read(current_build()).get('resume_not_before_unix', 0)))
        save()


def launch():
    request_pause(False)
    log = (HERE / 'collect_500_batches.log').open('a')
    try:
        child = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-s',
            str(HERE / '.venv/bin/python'),
            *([str(HERE/'collect_500_batches.py')] if control['profile']=='fixed991' else [str(HERE/'run_profile.py'),control['profile']])],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
    finally:
        log.close()


def monitor():
    while True:
        with LOCK:
            for child in children[:]:
                if child.poll() is not None:
                    children.remove(child)
            if control['mode'] == 'waiting' and time.time() >= control['resume_after']:
                if not processes():
                    launch()
                control.update(mode='running', error=None)
                save()
        time.sleep(1)


def status():
    progress = read(current_out() / 'progress.json')
    build = read(current_build())
    thermal = read(current_out() / 'thermal/thermal_latest.json')
    active = bool(processes())
    mode = control['mode']
    if mode == 'running' and not active:
        mode = 'complete' if progress.get('completed', 0) >= total() else 'stopped'
    return dict(control, mode=mode, active=active, completed=progress.get('completed', 0),
                total=total(), profiles=listing(), build=build, thermal=thermal,
                cooldown=max(0, int(max(control['resume_after'],
                    build.get('resume_not_before_unix', 0)) - time.time())))




class Handler(BaseHTTPRequestHandler):
    def reply(self, data, kind='application/json', code=200):
        body = data.encode()
        self.send_response(code)
        self.send_header('Content-Type', kind + '; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.headers.get('Host') not in ('127.0.0.1:8765','localhost:8765'):
            self.reply('{}',code=403);return
        if self.path == '/api/profiles':
            self.reply(json.dumps(listing()))
        elif self.path == '/api/status':
            self.reply(json.dumps(status()))
        else:
            self.reply('{}', code=404)

    def do_POST(self):
        if self.headers.get('Host') not in ('127.0.0.1:8765','localhost:8765'):
            self.reply('{}',code=403);return
        if self.headers.get('X-Control-Token') != TOKEN:
            self.reply('Invalid control token', 'text/plain', 403)
            return
        if self.path == '/api/profile':
            try:
                payload=json.loads(self.rfile.read(min(int(self.headers.get('Content-Length','0')),131072)))
                name=payload['name']
                if name!='fixed991':load(name)
                with LOCK:
                    if processes() or control['mode'] in ('pausing','waiting'):raise ValueError('Pause the current task and wait for the worker to close before switching profiles')
                    control.update(profile=name,mode='paused',error=None);save()
            except (ValueError,KeyError,OSError) as e:
                self.reply(str(e),'text/plain',409);return
        elif self.path == '/api/create':
            try:
                payload=json.loads(self.rfile.read(min(int(self.headers.get('Content-Length','0')),131072)))
                job=create(payload['name'],payload['messages'],payload.get('suffix','这个图片的 password 是什么？'),payload.get('limit',10000))
                self.reply(json.dumps(job));return
            except (ValueError,KeyError,OSError) as e:
                self.reply(str(e),'text/plain',400);return
        elif self.path == '/api/pause':
            if control['mode'] != 'pausing':
                threading.Thread(target=pause, daemon=True).start()
        elif self.path == '/api/resume':
            with LOCK:
                if control['mode'] == 'pausing':
                    self.reply('Wait for the worker to close', 'text/plain', 409)
                    return
                if not processes() and read(current_out() / 'progress.json').get('completed', 0) < total():
                    control.update(mode='waiting', error=None)
                    save()
        else:
            self.reply('{}', code=404)
            return
        self.reply('{"ok":true}')

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    (HERE/'panel_connection.json').write_text(json.dumps(dict(url=f'http://127.0.0.1:{PORT}',token=TOKEN)))
    (HERE/'panel_connection.json').chmod(0o600)
    threading.Thread(target=monitor, daemon=True).start()
    print(f'Collection panel: http://127.0.0.1:{PORT}', flush=True)
    server.serve_forever()
