"""Dependency-free interactive terminal for KVidence recovery and acquisition."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import curses
import json
import locale
import os
import re
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import unicodedata
import urllib.request
import urllib.error

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COLLECTION = HERE / 'experiments/fixed_prefix_image_inverse'
RECOVERY_PY = ROOT / 'kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python'

HELP = """/status                 Refresh collection status
/pause  /resume         Pause or resume image collection
/profiles  /use NAME    List or switch context profiles
/create NAME FILE       Create a context from a messages JSON file
/watch hiertext         Show overnight HierText collection progress
/watch live             Return to the original collection panel
/watch PATH             Follow a recovery directory or events.jsonl
/run ARGUMENTS          Start recovery with the existing run.py options
/stop                   Interrupt recovery started in this session
/clear  /help  /quit    Clear the feed, show help, or leave the workspace

Up/Down: history   PageUp/PageDown: scroll   Ctrl+U: clear input
Leaving this workspace keeps collection and recovery workers running.
Recovery interruption preserves written files; resumability is not guaranteed."""
HELP += '\n\nOr describe a case in plain language: create case1, provide a capture path, then start recovery.\nThe local agent asks for missing information and records case metadata.'
HELP += '\n/case help              Fixed metadata commands; no local model required'
HELP += '\n/experiment_demo [plaintext_chat|encrypted_chat] [ID]  Export and show dataset experiments\n/results [plaintext_chat|encrypted_chat] [ID]  Save measurements and candidate distances'


def pixel_rows(bitmap):
    """Pack two bitmap rows into each terminal row using half-block pixels."""
    width = max(map(len, bitmap))
    bitmap = [row.ljust(width, '0') for row in bitmap]
    if len(bitmap) % 2:bitmap.append('0' * width)
    blocks = {'00':' ', '10':'▀', '01':'▄', '11':'█'}
    return [''.join(blocks[a+b] for a,b in zip(bitmap[i],bitmap[i+1]))
            for i in range(0,len(bitmap),2)]


PIXEL_FONT = {
    'K': ('10001','10010','10100','11000','10100','10010','10001'),
    'V': ('10001','10001','10001','10001','10001','01010','00100'),
    'i': ('010','000','110','010','010','010','111'),
    'd': ('00001','00001','01101','10011','10001','10011','01101'),
    'e': ('00000','00000','01110','10001','11111','10000','01110'),
    'n': ('00000','00000','11110','10001','10001','10001','10001'),
    'c': ('00000','00000','01110','10001','10000','10001','01110'),
}
PIXEL_WORDMARK = pixel_rows(['0'.join(PIXEL_FONT[c][row] for c in 'KVidence') for row in range(7)])
PIXEL_GHOST = pixel_rows((
    '0000111110000',
    '0011111111100',
    '0111111111110',
    '0111111111110',
    '1110011100111',
    '1110011100111',
    '1111111111111',
    '1111111111111',
    '1111100011111',
    '1111111111111',
    '1110111011101',
    '0100010001000',
))
MINI_GHOST = pixel_rows(('01110','11111','10101','11111','10101'))


def cells(text):
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in text)


def clean(text):
    return ''.join(c if c in '\n\t' or ord(c) >= 32 and ord(c) != 127 else ' ' for c in str(text)).replace('\t', '    ')


def wrap(text, width):
    lines = []
    for part in clean(text).split('\n'):
        line = ''; used = 0
        for c in part:
            n = cells(c)
            if used + n > width:
                lines.append(line); line = ''; used = 0
            line += c; used += n
        lines.append(line)
    return lines


def api(endpoint, payload=None):
    connection = json.loads((COLLECTION / 'panel_connection.json').read_text())
    request = urllib.request.Request(connection['url'] + '/api/' + endpoint,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'X-Control-Token': connection['token'], 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        message = e.read().decode()
        # Existing services can remain running while this client is upgraded.
        legacy_errors = {
            '请先暂停当前任务，等待 worker 关闭后切换': 'Pause collection and wait for the worker to close before switching profiles.',
            '请等待 worker 关闭': 'Wait for the worker to close.',
        }
        raise ValueError(legacy_errors.get(message, message)) from e


class EventReader:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.source = None; self.offset = 0; self.state = None; self.next_scan = 0

    def poll(self):
        from kvrecon.dashboard import DashboardState
        if self.path.is_file():
            source = self.path
        elif time.monotonic() >= self.next_scan:
            self.next_scan = time.monotonic() + 2
            sources = list(self.path.rglob('events.jsonl')) if self.path.exists() else []
            source = max(sources, key=lambda p: p.stat().st_mtime_ns) if sources else None
        else:
            source = self.source
        if source is None:
            return []
        if source != self.source or source.stat().st_size < self.offset:
            self.source = source; self.offset = 0
            decode = None
            protocol = source.parent / 'protocol.json'
            if protocol.exists():
                cfg = json.loads(protocol.read_text())
                tokenizer_file = cfg.get('model_config', {}).get('tokenizer')
                if tokenizer_file and Path(tokenizer_file).suffix == '.json':
                    from tokenizers import Tokenizer
                    tokenizer = Tokenizer.from_file(tokenizer_file)
                    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)
            self.state = DashboardState(str(source), source.parent.name, decode)
        with source.open('rb') as stream:
            stream.seek(self.offset); raw = stream.read(2 << 20)
        end = raw.rfind(b'\n') + 1
        changes = []
        for row in raw[:end].splitlines():
            event = json.loads(row)
            self.state.apply(event)
            if event['type'] == 'position_end':
                changes.append(f"Position {event['position']}  token={event.get('accepted_token_id')}  committed={event.get('committed')}")
            elif event['type'] in ('finish', 'error'):
                changes.append('Recovery status: ' + self.state.data['status'])
        self.offset += end
        return changes


def overnight_status(directory=None):
    directory=Path(directory or COLLECTION/'data/overnight_20261025')
    raw=json.loads((directory/'status.json').read_text())
    completed=raw.get('completed',0);stage=raw['state'];batch=raw.get('batch')
    if stage=='collecting' and batch:
        progress=Path(batch)/'progress.json'
        if progress.exists():completed+=json.loads(progress.read_text()).get('completed',0)
    if stage=='archived':
        # Archive notifications are brief; derive totals from verified local receipts.
        completed=0
        for receipt in directory.glob('shard_*_archive.json'):
            info=json.loads(receipt.read_text())
            index=Path(info['directory'])/'index.json'
            if index.exists():completed+=len(json.loads(index.read_text()))
    total=8639
    mode='complete' if stage=='complete' else 'stopped' if stage in ('stopped_at_deadline','failed','deadline_pause_requested') else 'running'
    cooldown=max(0,int(raw.get('resume_not_before_unix',0)-time.time()))
    return dict(profile='HierText overnight / fixed991',completed=completed,total=total,mode=mode,
        active=mode=='running',cooldown=cooldown,error=raw.get('error'),build=dict(stage=stage),
        deadline=raw.get('deadline'))


class Terminal:
    def __init__(self, demo=False, watch=None):
        self.demo = demo
        self.messages = deque(maxlen=800)
        self.history = []; self.history_index = 0; self.draft = ''
        self.input = ''; self.cursor = 0; self.scroll = 0; self.running = True
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.pending = []; self.next_poll = 0; self.collect = None; self.connected = False
        self.reader = EventReader(watch) if watch else None
        self.process = None; self.output = None; self.log = None; self.log_offset = 0
        self.pasting = False
        self.last_collect_error = None
        self.agent = None
        self.case_name = None
        self.case_model = None
        self.case_commands = None
        self.help_view = None
        self.help_offset = 0
        self.experiment_demo = None
        self.collection_view = None
        # The empty feed displays the workspace welcome screen.

    def add(self, who, text, color=0):
        self.messages.append((who, clean(text), color))

    def submit(self, title, function, handler=None):
        self.pending.append((title, self.pool.submit(function), handler))

    def collection_action(self, endpoint, payload):
        api(endpoint, payload)
        return api('status')

    def set_status(self, state):
        self.collect = state; self.connected = True
        error = state.get('error') or state.get('build', {}).get('error')
        if error and error != self.last_collect_error:
            self.add('Collection error', error, 3)
        self.last_collect_error = error

    def execute(self, text):
        text = text.strip()
        if not text:
            return
        self.add('You', text, 2)
        self.history.append(text); self.history_index = len(self.history); self.draft = ''
        try:
            if not text.startswith('/'):
                if self.demo:raise ValueError('Demo mode is read-only. Live cases are unchanged.')
                if any(title in ('Agent','Case command') for title,_,_ in self.pending):raise ValueError('The previous case operation is still in progress.')
                if self.agent is None:
                    from case_agent import CaseAgent
                    self.agent=CaseAgent()
                    self.agent.current=self.case_name
                self.submit('Agent',lambda:self.agent.handle(text),self.agent_reply)
                return
            args = shlex.split(text); command = args[0]
            if self.demo and command in ('/pause','/resume','/use','/create','/run','/stop'):
                raise ValueError('Demo mode is read-only. Live tasks are unchanged.')
            if command in ('/quit', '/exit'):
                self.running = False
            elif command == '/help':
                from case_commands import HELP as CASE_HELP
                self.open_help('Command reference',HELP+'\n\nCASE METADATA\n'+CASE_HELP)
            elif command in ('/experiment_demo','/results'):
                from run_text_experiment import options
                try: selection=options(args[1:])
                except SystemExit:raise ValueError('Usage: /experiment_demo [dataset] [ID] --methods guided,collision_plus,exhaustive [--max-queries N]')
                if selection.methods:
                    if command=='/results':raise ValueError('Use /experiment_demo to launch recovery.')
                    if self.demo:raise ValueError('Demo mode is read-only; start ./reconstruct without --demo.')
                    self.submit('Start experiment',lambda:self.prepare_experiment(selection),self.start_experiment)
                else:
                    from shadow_experiments import ShadowCatalog,dataset_name
                    from experiment_results import save_scope,report
                    dataset=dataset_name(selection.dataset);sample=selection.sample
                    def export_results():
                        catalog=ShadowCatalog()
                        folder,rows=save_scope(catalog,dataset,sample)
                        return report(folder,rows)
                    self.submit('Experiment results',export_results,lambda text:self.open_help('Text experiment results',text))
            elif command == '/case':
                if len(args)==1 or args[1]=='help':
                    from case_commands import HELP as CASE_HELP
                    self.open_help('Case commands',CASE_HELP)
                elif self.demo:
                    raise ValueError('Demo mode is read-only.')
                else:
                    if any(title in ('Agent','Case command') for title,_,_ in self.pending):raise ValueError('The previous case operation is still in progress.')
                    if self.case_commands is None:
                        from case_commands import CaseCommands
                        self.case_commands=CaseCommands(self.agent.store if self.agent else None)
                    self.case_commands.current=self.case_name
                    self.submit('Case command',lambda:self.case_commands.execute(args[1:]),self.agent_reply)
            elif command == '/clear':
                self.messages.clear()
            elif command == '/status':
                self.submit('Collection status', lambda: overnight_status(self.collection_view) if self.collection_view else api('status'), self.set_status)
            elif command in ('/pause', '/resume'):
                self.submit('Collection control', lambda: self.collection_action(command[1:], {}), self.set_status)
            elif command == '/profiles':
                self.submit('Profiles', lambda: api('profiles'), lambda rows: self.add('Profiles', '\n'.join(
                    f"{r['name']}  ·  {r.get('prefix_tokens', 991)} tokens" for r in rows)))
            elif command == '/use' and len(args) == 2:
                self.submit('Switch profile', lambda: self.collection_action('profile', {'name': args[1]}), self.set_status)
            elif command == '/create' and len(args) == 3:
                self.submit('Create profile', lambda: api('create', {'name': args[1], 'messages': json.loads(Path(args[2]).expanduser().read_text())}),
                            lambda job: self.add('Profiles', f"Created {job['name']} · {job['prefix_tokens']} tokens. Pause collection, then /use to switch."))
            elif command == '/watch' and len(args) == 2:
                if args[1] in ('hiertext','overnight'):
                    self.collection_view=COLLECTION/'data/overnight_20261025'
                    self.set_status(overnight_status(self.collection_view));self.next_poll=0
                    self.add('Collection','Following HierText overnight collection. /watch live returns to the original panel.')
                elif args[1]=='live':
                    self.collection_view=None;self.next_poll=0
                    self.add('Collection','Following the original collection service.')
                else:
                    self.reader = EventReader(args[1]); self.add('Recovery', 'Following ' + str(self.reader.path))
            elif command == '/run':
                self.submit('Start recovery', lambda: self.prepare_run(args[1:]), self.start_run)
            elif command == '/stop':
                if self.process and self.process.poll() is None:
                    os.killpg(self.process.pid, signal.SIGINT)
                    self.add('Recovery', 'Interrupt requested. Waiting for worker cleanup; written files are preserved.')
                else:
                    self.add('Recovery', 'No recovery worker is running in this session.')
            else:
                self.add('Hint', 'Unknown command. Type /help to see available commands.', 3)
        except (ValueError, OSError) as e:
            self.add('Error', str(e), 3)

    def open_help(self,title,text):
        self.help_view=(title,text);self.help_offset=0

    def agent_reply(self,response):
        self.case_name=response.get('case') or self.case_name
        if 'model_name' in response:self.case_model=response['model_name']
        if self.agent:self.agent.current=self.case_name
        self.add('Case' if response.get('origin')=='command' else 'Local agent',response['reply'],6)
        if response.get('metadata'):
            path=Path(response['metadata'])
            try:label=str(path.relative_to(HERE))
            except ValueError:label=str(path)
            self.add('Case metadata',label,4)
        intent=response.get('intent')
        if intent in ('pause','resume'):
            self.submit('Collection control',lambda:self.collection_action(intent,{}),self.set_status)
        elif intent=='run' and response.get('ready'):
            name=response['case']
            self.submit('Start case',lambda:self.prepare_case(name),self.start_case)

    def prepare_case(self,name):
        self.prepare_run(['case'])
        from casework import CaseStore
        CaseStore().plan(name)
        return name

    def prepare_experiment(self,selection):
        from run_text_experiment import check_idle,create_experiment
        if self.process and self.process.poll() is None:raise ValueError('A recovery worker is already running.')
        check_idle()
        return create_experiment(selection)

    def start_experiment(self,folder):
        from run_text_experiment import check_idle
        if self.process and self.process.poll() is None:raise ValueError('A recovery worker is already running.')
        check_idle()
        self.output=folder
        self.log=folder/'experiment.log';self.log_offset=0
        with self.log.open('ab') as log:
            self.process=subprocess.Popen(['/usr/bin/caffeinate','-i','-s',str(RECOVERY_PY),'-u',str(HERE/'run_text_experiment.py'),'--experiment',str(folder)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        self.reader=EventReader(folder)
        self.add('Text experiment',f'Selected methods queued sequentially. PID {self.process.pid}\nResults: {folder}\n/stop interrupts; /quit leaves the worker running.',1)

    def start_case(self,name):
        from casework import CaseStore
        if self.process and self.process.poll() is None:raise ValueError('A recovery worker is already running.')
        directory=CaseStore().path(name)
        run_id='run_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+str(time.time_ns()%1000000)
        self.output=directory/'runs'/run_id
        logs=directory/'logs';logs.mkdir(exist_ok=True)
        self.log=logs/(run_id+'.log');self.log_offset=0
        with self.log.open('ab') as log:
            self.process=subprocess.Popen(['/usr/bin/caffeinate','-i','-s',str(RECOVERY_PY),'-u',str(HERE/'run_case.py'),name,'--run-dir',str(self.output)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        self.reader=EventReader(self.output)
        self.add('Case recovery',f'{name}: registered methods queued sequentially. PID {self.process.pid}\nRun: {self.output}\n/stop interrupts; leaving the workspace keeps the worker running.',1)

    def prepare_run(self, args):
        if self.demo:
            raise ValueError('Demo mode does not start collection or recovery.')
        if self.process and self.process.poll() is None:
            raise ValueError('A recovery worker is already running.')
        try:
            state = api('status')
        except (OSError, ValueError):
            raise ValueError('Cannot verify collection state. Start ./collectctl serve, then pause collection.')
        if state['active'] or state['mode'] in ('waiting', 'pausing'):
            raise ValueError('Use /pause and wait for paused status before starting recovery.')
        if state.get('cooldown',0)>0:
            raise ValueError(f"The phone is cooling down. Wait {state['cooldown']} seconds before recovery.")
        if not args:
            raise ValueError('/run --observation PATH --output NEW_DIRECTORY [recovery options]')
        return args

    def start_run(self, args):
        if self.process and self.process.poll() is None:
            raise ValueError('A recovery worker is already running.')
        if '--live-events' not in args:
            args = args + ['--live-events']
        output = None
        for i, arg in enumerate(args):
            if arg == '--output' and i + 1 < len(args): output = args[i + 1]
            elif arg.startswith('--output='): output = arg.split('=', 1)[1]
        if not output:
            raise ValueError('Specify --output NEW_DIRECTORY')
        self.output = Path(output).expanduser().resolve()
        if self.output.exists():
            raise ValueError('Choose a new output directory to preserve existing experiments.')
        # Paths do not pass through a shell, so expand the output explicitly.
        args = list(args)
        for i, arg in enumerate(args):
            if arg == '--output':args[i+1] = str(self.output)
            elif arg.startswith('--output='):args[i] = '--output=' + str(self.output)
        logs=self.output.parent/'.reconstruct-logs';logs.mkdir(parents=True,exist_ok=True)
        self.log = logs/(self.output.name+'_'+str(time.time_ns())+'.log'); self.log_offset = 0
        with self.log.open('ab') as log:
            self.process = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-s',str(RECOVERY_PY), '-u', str(HERE / 'run.py'), 'run', *args],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.reader = EventReader(self.output)
        self.add('Recovery', f'Worker started · PID {self.process.pid}\nOutput: {self.output}\n/stop interrupts recovery; /quit leaves it running.')

    def tick(self):
        if self.experiment_demo:self.experiment_demo.advance()
        if self.demo:
            self.collect = dict(profile='demo_context', completed=1855, total=10000, mode='demo', cooldown=0)
        elif time.monotonic() >= self.next_poll:
            self.next_poll = time.monotonic() + 3
            if not any(title == '_poll' for title, _, _ in self.pending):
                self.submit('_poll', lambda: overnight_status(self.collection_view) if self.collection_view else api('status'), self.set_status)
        for item in self.pending[:]:
            title, future, handler = item
            if future.done():
                self.pending.remove(item)
                try:
                    result = future.result()
                    if handler:handler(result)
                    if title not in ('_poll', 'Profiles', 'Create profile', 'Start recovery', 'Agent', 'Start case', 'Case command'):
                        self.add(title, 'Updated.')
                except Exception as e:
                    if title == '_poll':
                        if self.connected or self.collect is None:
                            self.add('Connection', 'Collection service is offline. Run ./collectctl serve in another terminal.', 3)
                        self.connected = False
                        self.collect = {}
                    else:self.add(title, str(e), 3)
        if self.reader:
            try:
                for line in self.reader.poll():self.add('Recovery', line)
            except (OSError, ValueError, KeyError) as e:
                self.add('Event reader', str(e), 3); self.reader = None
        if self.log and self.log.exists():
            with self.log.open('rb') as log:
                log.seek(self.log_offset); raw = log.read(65536); self.log_offset += len(raw)
            if raw:self.add('worker', raw.decode(errors='replace').strip()[-4000:])
        if self.process and self.process.poll() is not None:
            self.add('Recovery', f'Worker exited with code {self.process.returncode}. Results: {self.output}')
            self.process = None

    def paint(self, screen):
        h, w = screen.getmaxyx(); screen.erase()
        if h < 16 or w < 54:
            try:screen.addstr(0, 0, 'Please resize to at least 54 columns x 16 rows.')
            except curses.error:pass
            screen.refresh();return
        width = min(w - 6, 108)
        left = max(3, (w - width) // 2)
        def put(y, text, color=0, bold=False, x=0):
            if not 0 <= y < h:return
            try:
                value = wrap(text, max(1,width-x))[0]
                screen.addstr(y, left+x, value, curses.color_pair(color) | (curses.A_BOLD if bold else 0))
            except curses.error:pass
        def rule(y):put(y, '─' * width, 8)
        if self.help_view:
            title,text=self.help_view
            put(1,'KVidence  /  '+title,1,True);rule(3)
            lines=wrap(text,width-1);space=h-8
            last_page=((len(lines)-1)//space)*space
            self.help_offset=max(0,min(self.help_offset,last_page))
            for y,line in enumerate(lines[self.help_offset:self.help_offset+space],4):
                parts=re.split(r' {3,}',line,maxsplit=1) if line.startswith('/') else []
                if len(parts)==2:
                    put(y,parts[0],6,True);put(y,parts[1],7,x=cells(parts[0])+3)
                else:put(y,line,6 if line.startswith('/') else 7,line.startswith('/'))
            rule(h-4)
            page=self.help_offset//space+1;pages=(len(lines)+space-1)//space
            put(h-3,f'Page {page} / {pages}  ·  Lines {self.help_offset+1}–{min(len(lines),self.help_offset+space)} / {len(lines)}',4)
            put(h-2,'↑↓ scroll  ·  PgUp/PgDn page  ·  Esc or q back',4)
            screen.refresh();return
        if self.experiment_demo:
            self.experiment_demo.draw(put,rule,h,width,wrap)
            screen.refresh();return
        state = self.collect or {}
        build = state.get('build', {})
        mode = {'running':'Collecting','paused':'Paused','pausing':'Pausing',
                'waiting':'Waiting to resume','complete':'Complete','stopped':'Stopped',
                'demo':'Preview'}.get(state.get('mode'), state.get('mode','Disconnected'))
        if state.get('mode') == 'running' and build.get('stage','').startswith('cooling'):
            mode = 'Waiting for device' if 'telemetry unavailable' in build.get('reason','') else 'Cooling down'
        for i,row in enumerate(MINI_GHOST):put(i+1,row,5)
        put(1, 'KVidence', 1, True, x=8)
        right = 'PREVIEW' if self.demo else 'RECONSTRUCTION'
        if width >= 64:put(1, right, 4, x=width-len(right))
        model_label={'gemma4-e2b':'Gemma 4 E2B','gemma3-1b':'Gemma 3 1B'}.get(self.case_model,'Model not identified')
        put(2, (model_label if self.case_name else 'Gemma 4 E2B')+'  /  Phone NPU', 4, x=8)
        profile = state.get('profile','No context selected')
        if self.case_name:profile=self.case_name+' / '+profile
        put(4, profile, 0, True)
        status_color = 2 if mode in ('Collecting','Complete') else 1
        if width >= len(profile)+len(mode)+5:put(4, mode, status_color, x=width-len(mode))
        completed = state.get('completed',0); total = state.get('total',0)
        bar_width = 18 if width >= 70 else 10
        filled = min(bar_width, max(0, int(bar_width*completed/total))) if total else 0
        progress = '━'*filled + '─'*(bar_width-filled)
        detail = f'{completed:,} / {total:,} images' if total else 'Collection service offline'
        cooldown = state.get('cooldown',0)
        if cooldown:detail += f'  ·  rest {cooldown//60:02d}:{cooldown%60:02d}'
        put(5, progress, 2)
        put(5, detail, 7, x=bar_width+2)
        rule(7)
        body_top = 9; body_end = h-9
        if not self.messages and body_end-body_top >= 3:
            available = body_end-body_top+1
            large = width >= 63 and available >= len(PIXEL_GHOST)
            ghost = PIXEL_GHOST if large else MINI_GHOST
            for i,row in enumerate(ghost):put(body_top+i,row,5)
            if large:
                for i,row in enumerate(PIXEL_WORDMARK):put(body_top+1+i,row,1,x=17)
                put(body_top+5,'RECONSTRUCTION WORKSPACE',4,x=17)
            else:
                put(body_top,'KVidence',1,True,x=9)
                put(body_top+2,'Reconstruction workspace',4,x=9)
            tagline_y = body_top+(8 if large else 4)
            if tagline_y <= body_end:
                put(tagline_y,'Create a case. Recover context. Inspect evidence.',7)
            shortcuts = [('create case1', 'Start with a description'),('/watch PATH', 'Follow a recovery'),('/help', 'Explore commands')]
            for i,(command,description) in enumerate(shortcuts):
                y = tagline_y+3+i*2
                if y > body_end:break
                put(y,command,6,True)
                if width >= 66:put(y,description,7,x=23)
        else:
            rows = []
            for who,text,color in self.messages:
                if who == 'You':
                    rows.extend(('› '+line,6,True) for line in wrap(text,width-3))
                else:
                    rows.append((who.upper(),color or 1,True))
                    body_color = 3 if color == 3 else 4 if who == 'worker' else 7
                    rows.extend((line,body_color,False) for line in wrap(text,width-1))
                rows.append(('',0,False))
            space = max(1,body_end-body_top+1)
            self.scroll = max(0,min(self.scroll,max(0,len(rows)-space)))
            end = max(0,len(rows)-self.scroll)
            for y,(line,color,bold) in enumerate(rows[max(0,end-space):end],body_top):
                parts = re.split(r' {3,}',line,maxsplit=1) if line.startswith('/') else []
                if len(parts)==2:
                    command,description=parts
                    put(y,command,6,True)
                    put(y,description,7,x=cells(command)+3)
                else:put(y,line,color,bold)
        summary = 'Ready  ·  /help for commands'
        if self.reader and self.reader.state:
            d = self.reader.state.data
            summary = f"Recovery {d['status']}  ·  {d['accepted']}/{d.get('planned_positions','?')} tokens  ·  {d['total_queries']:,} candidates"
            if d['recovered_text']:
                put(h-8,'Recovered  '+d['recovered_text'].replace('\n',' '),2)
        put(h-6,summary,4)
        rule(h-5)
        start = 0
        while cells(self.input[start:self.cursor]) > width-5:start += 1
        put(h-3,'❯',6,True)
        if self.input:put(h-3,self.input[start:],6,True,x=3)
        else:put(h-3,'Describe a case, or type /help',4,x=3)
        rule(h-2)
        hint = '↑↓ history   / commands   Ctrl+U clear'
        if self.pasting:hint = 'Pasting  ·  press Enter when ready'
        elif self.scroll:hint = 'Scrollback  ·  PgDn to return to the latest updates'
        elif any(title=='Agent' for title,_,_ in self.pending):hint='Local agent  ·  thinking on this Mac'
        put(h-1,hint,4)
        connection = 'demo' if self.demo else 'connected' if self.connected else 'offline'
        if width >= 76:put(h-1,connection,2 if self.connected else 4,x=width-len(connection))
        try:screen.move(h-3,left+3+cells(self.input[start:self.cursor]))
        except curses.error:pass
        screen.refresh()

    def key(self, screen, key):
        if self.experiment_demo and not self.help_view:
            if not self.experiment_demo.key(key):self.experiment_demo=None
            return
        if self.help_view:
            page=max(1,screen.getmaxyx()[0]-8)
            if key==curses.KEY_UP:self.help_offset=max(0,self.help_offset-1)
            elif key==curses.KEY_DOWN:self.help_offset+=1
            elif key==curses.KEY_PPAGE:self.help_offset=max(0,self.help_offset-page)
            elif key==curses.KEY_NPAGE:self.help_offset+=page
            elif key==curses.KEY_HOME:self.help_offset=0
            elif key==curses.KEY_END:self.help_offset=10**9
            elif key in ('q','\x1b','\x03','\n','\r'):self.help_view=None
            return
        if key == '\x1b':
            sequence = ''
            for _ in range(5):
                try:sequence += screen.get_wch()
                except (curses.error, TypeError):break
            if sequence == '[200~':self.pasting = True
            elif sequence == '[201~':self.pasting = False
            return
        if self.pasting:
            if isinstance(key,str):
                value = ' ' if key in '\r\n\t' else clean(key)
                self.input = self.input[:self.cursor] + value + self.input[self.cursor:]; self.cursor += len(value)
            return
        if key in ('\n','\r',curses.KEY_ENTER):
            text = self.input; self.input = ''; self.cursor = 0; self.scroll = 0; self.execute(text)
        elif key in ('\x03','\x15'):
            self.input = ''; self.cursor = 0
        elif key in (curses.KEY_BACKSPACE,'\x7f','\b') and self.cursor:
            self.input = self.input[:self.cursor-1]+self.input[self.cursor:];self.cursor -= 1
        elif key == curses.KEY_DC:
            self.input=self.input[:self.cursor]+self.input[self.cursor+1:]
        elif key in (curses.KEY_LEFT,curses.KEY_RIGHT):
            self.cursor=max(0,min(len(self.input),self.cursor+(-1 if key==curses.KEY_LEFT else 1)))
        elif key in (curses.KEY_HOME,'\x01'):self.cursor=0
        elif key in (curses.KEY_END,'\x05'):self.cursor=len(self.input)
        elif key in (curses.KEY_UP,curses.KEY_DOWN):
            if self.history_index==len(self.history):self.draft=self.input
            self.history_index=max(0,min(len(self.history),self.history_index+(-1 if key==curses.KEY_UP else 1)))
            self.input=self.history[self.history_index] if self.history_index<len(self.history) else self.draft;self.cursor=len(self.input)
        elif key == curses.KEY_PPAGE:self.scroll+=8
        elif key == curses.KEY_NPAGE:self.scroll=max(0,self.scroll-8)
        elif isinstance(key,str) and key.isprintable():
            self.input=self.input[:self.cursor]+key+self.input[self.cursor:];self.cursor+=1

    def run(self, screen):
        curses.raw();screen.timeout(100)
        if curses.has_colors():
            curses.start_color();curses.use_default_colors()
            palette = ((1,173),(2,114),(3,203),(4,243),(5,211),(6,117),(7,250),(8,239)) if curses.COLORS >= 256 else ((1,curses.COLOR_YELLOW),(2,curses.COLOR_GREEN),(3,curses.COLOR_RED),(4,curses.COLOR_WHITE),(5,curses.COLOR_MAGENTA),(6,curses.COLOR_CYAN),(7,curses.COLOR_WHITE),(8,curses.COLOR_WHITE))
            for i,color in palette:
                curses.init_pair(i,color,-1)
        sys.stdout.write('\x1b[?2004h');sys.stdout.flush()
        try:
            while self.running:
                self.tick();self.paint(screen)
                try:self.key(screen,screen.get_wch())
                except curses.error:pass
        finally:
            sys.stdout.write('\x1b[?2004l');sys.stdout.flush()
            self.pool.shutdown(wait=False, cancel_futures=True)
            if self.agent:self.agent.model.close()


def main():
    parser=argparse.ArgumentParser(description='KVidence interactive reconstruction terminal')
    parser.add_argument('--demo',action='store_true',help='Display sample status without starting a worker')
    parser.add_argument('--watch',type=Path,help='Open an existing recovery directory or events.jsonl')
    args=parser.parse_args()
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.exit(2,'Run ./reconstruct in an interactive terminal. For a one-shot status, use ./collectctl status.\n')
    locale.setlocale(locale.LC_ALL,'')
    try:curses.wrapper(Terminal(args.demo,args.watch).run)
    except KeyboardInterrupt:pass
    print('Workspace closed. Collection and recovery workers remain running.')


if __name__ == '__main__':main()
