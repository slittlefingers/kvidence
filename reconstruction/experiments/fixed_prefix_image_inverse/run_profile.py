"""Resumable profile worker with cooperative pause and thermal rests."""
import json
from pathlib import Path
import subprocess
import sys
import time
import hashlib
from tokenizers import Tokenizer
from collection_profiles import HERE, ROOT, load
from thermal_control import thermal_snapshot, CoolingRequired

job = load(sys.argv[1])
prefix = json.loads(Path(job['prefix']).read_text())
tokenizer_file = ROOT / 'kvidence/experiments/kv_guided_search_gemma4_int8/model/tokenizer.json'
assert hashlib.sha256(prefix['prefix_text'].encode()).hexdigest() == prefix['prefix_sha256'], 'prefix text changed'
assert hashlib.sha256(tokenizer_file.read_bytes()).hexdigest() == prefix['tokenizer_sha256'], 'tokenizer changed'
assert Tokenizer.from_file(str(tokenizer_file)).encode(prefix['prefix_text'], add_special_tokens=False).ids == prefix['prefix_ids'], 'prefix token IDs changed'
out = Path(job['out']); out.mkdir(parents=True, exist_ok=True)
control = HERE / 'job_control.json'
def paused():
    return control.exists() and json.loads(control.read_text()).get('pause_requested')
def count():
    return len(json.loads((out / 'index.json').read_text())) if (out / 'index.json').exists() else 0
def status(stage, **fields):
    p = out / 'build_status.json'; temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(stage=stage, completed=count(), planned=job['limit'], **fields)))
    temp.replace(p)
while count() < job['limit'] and not paused():
    try:
        safe = thermal_snapshot(out / 'thermal')['thermal_status'] <= 1
    except (CoolingRequired, subprocess.TimeoutExpired):
        safe = False
    if not safe:
        status('cooling_extended'); time.sleep(5); continue
    goal = min(job['limit'], (count() // 500 + 1) * 500)
    status('collecting_batch', batch_end=goal)
    with (out / 'collection.log').open('a') as log:
        result = subprocess.run([str(HERE / '.venv/bin/python'), str(HERE / 'collect_configurable.py'),
            '--manifest', job['manifest'], '--limit', str(goal), '--out', str(out),
            '--prefix', job['prefix'], '--control', str(control)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode == 76 or paused():
        status('paused'); break
    if result.returncode not in (0, 75):
        status('failed', error=f'Collector exit {result.returncode}; see {out}/collection.log'); sys.exit(1)
    if count() < job['limit']:
        deadline = time.time() + 300
        status('cooling', resume_not_before_unix=deadline)
        while time.time() < deadline and not paused():
            time.sleep(min(1, max(0, deadline - time.time())))
if count() >= job['limit']:
    status('validating')
    with (out / 'validation.log').open('a') as log:
        subprocess.run([str(HERE / '.venv/bin/python'), str(HERE / 'validate_fixed.py'), str(out)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    status('complete_collected')
