"""Execute a case's registered methods sequentially, with a frozen run manifest."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from casework import CaseStore,HERE,write,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('case');parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();store=CaseStore();case,plans=store.plan(args.case)
    from terminal import api,RECOVERY_PY,ROOT
    state=api('status')
    if state['active'] or state['mode'] in ('waiting','pausing'):raise ValueError('Collection must be paused before case recovery.')
    if state.get('cooldown',0)>0:raise ValueError('Wait for phone cooling to finish before case recovery.')
    args.run_dir.mkdir(parents=True,exist_ok=False)
    # Record the evidence fingerprint and exact argv before any NPU use.
    snapshot=dict(case=case,plans=plans,created_at=time.time(),status='running',results=[])
    write(args.run_dir/'plan.json',snapshot)
    run=dict(directory=str(args.run_dir),started_at=time.time(),status='running')
    case=store.load(args.case);case['runs'].append(run);store.save(case)
    worker=None
    try:
        for plan in plans:
            evidence=case['evidence'];observation=Path(case['fields']['observation'])
            if sha(observation/'metadata.json')!=evidence['metadata_sha256'] or sha(observation/'observed.npz')!=evidence['observed_sha256']:
                raise ValueError('Observation changed after the case run manifest was frozen.')
            # A resumed collection must not silently collide with the next method.
            state=api('status')
            if state['active'] or state['mode'] in ('waiting','pausing'):raise ValueError('Collection resumed; case recovery stopped before the next method.')
            if state.get('cooldown',0)>0:raise ValueError('A new cooling period started; case recovery stopped before the next method.')
            output=args.run_dir/plan['method']
            argv=[str(RECOVERY_PY),'-u',str(HERE/'run.py'),'run',*plan['args'],'--output',str(output),'--live-events']
            print('Starting '+plan['label'],flush=True)
            with (args.run_dir/(plan['method']+'.log')).open('w') as log:
                worker=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                code=worker.wait()
            row=dict(method=plan['method'],argv=argv,exit_code=code,output=str(output))
            if (output/'result.json').exists():
                result=json.loads((output/'result.json').read_text())
                row.update(status=result['status'],result_sha256=sha(output/'result.json'),
                           recovered_tokens=len(result['recovered_token_ids']),online_seconds=result['online_seconds'])
                trace=output/'trace.jsonl'
                if trace.exists():row['total_queries']=sum(json.loads(line)['queries'] for line in trace.read_text().splitlines())
            snapshot['results'].append(row);write(args.run_dir/'plan.json',snapshot)
            if code:raise RuntimeError(f"{plan['method']} exited with code {code}; see {plan['method']}.log")
        snapshot['status']='complete'
    except BaseException as e:
        if worker and worker.poll() is None:
            worker.send_signal(signal.SIGINT)
            try:worker.wait(timeout=45)
            except subprocess.TimeoutExpired:worker.terminate()
        snapshot['status']='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed';snapshot['error']=str(e)
        raise
    finally:
        snapshot['finished_at']=time.time();write(args.run_dir/'plan.json',snapshot)
        current=store.load(args.case)
        for entry in current['runs']:
            if entry['directory']==str(args.run_dir):entry.update(status=snapshot['status'],finished_at=time.time())
        store.save(current)
        from experiment_results import summarize_run,export_candidates
        measured=[summarize_run(args.run_dir/entry['method']) for entry in snapshot['results'] if (args.run_dir/entry['method']/'result.json').exists()]
        for measurement in measured:
            output=Path(measurement['output'])
            measurement['candidates_csv']=str(output/'candidates.csv')
            export_candidates(output,output/'candidates.csv')
        write(args.run_dir/'summary.json',measured)
    print('Case recovery complete. Acceptance is not independent identity verification.',flush=True)


if __name__=='__main__':main()
