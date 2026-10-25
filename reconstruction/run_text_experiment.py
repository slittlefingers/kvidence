"""Sequential dataset text experiments with isolated evidence and evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
from casework import CaseStore,HERE,ROOT,sha,write
from shadow_experiments import ShadowCatalog,DEFAULT_METHODS
from experiment_results import RESULTS_ROOT,summarize_run,export_candidates


def options(arguments):
    parser=argparse.ArgumentParser(prog='/experiment_demo')
    parser.add_argument('dataset',nargs='?',default='plaintext_chat')
    parser.add_argument('sample',nargs='?')
    parser.add_argument('--methods',default=None)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--max-queries',type=int)
    args=parser.parse_args(arguments)
    args.methods=args.methods.split(',') if args.methods else (list(DEFAULT_METHODS) if args.run else None)
    if args.methods and (len(set(args.methods))!=len(args.methods) or any(m not in DEFAULT_METHODS for m in args.methods)):
        raise ValueError('Choose distinct methods: guided, collision_plus, exhaustive. Official CPA requires a separate calibrated case.')
    if args.max_queries is not None and args.max_queries<1:raise ValueError('--max-queries must be positive.')
    return args


def check_idle():
    from terminal import api
    state=api('status')
    if state.get('active') or state.get('mode') in ('waiting','pausing'):
        raise ValueError('Pause collection before starting an experiment: /pause')
    if state.get('cooldown',0)>0:raise ValueError('Wait for phone cooling to finish before starting an experiment.')


def create_experiment(args,catalog=None,root=RESULTS_ROOT):
    catalog=catalog or ShadowCatalog();spec=catalog.specification(args.dataset,args.sample)
    if not args.methods:raise ValueError('Select --methods or --run to start an experiment.')
    spec.update(methods=args.methods,max_queries=args.max_queries,state='queued',created_at=time.time())
    if spec.get('dataset_directory'):spec['inputs_sha256']=sha(Path(spec['dataset_directory'])/'inputs.jsonl')
    folder=Path(root)/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+spec['dataset']+'-'+uuid.uuid4().hex[:10])
    folder.mkdir(parents=True,exist_ok=False);write(folder/'experiment.json',spec)
    return folder


def prepare_sample(spec,sid,folder):
    # This preparation process can see benchmark inputs; recovery runs in a separate
    # process with only the exported observation/public prefix, never these labels.
    from tokenizers import Tokenizer
    from kvrecon.config import load_model_config,DEFAULT_MODEL_CONFIG
    from kvrecon.observation import import_snapshot
    source=Path(spec['dataset_directory']);manifest=source/'manifest.json'
    if sha(manifest)!=spec['manifest_sha256']:raise ValueError('Dataset manifest changed.')
    job=json.loads((source/'capture/job.json').read_text())
    if spec.get('inputs_sha256') and sha(source/'inputs.jsonl')!=spec['inputs_sha256']:raise ValueError('Benchmark input changed after planning.')
    profile=load_model_config(DEFAULT_MODEL_CONFIG)
    if sha(profile['model'])!=spec['model_sha256'] or job['model_sha256']!=spec['model_sha256']:
        raise ValueError('Model fingerprint mismatch.')
    if sha(profile['tokenizer'])!=json.loads(manifest.read_text()).get('tokenizer_sha256','cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f'):
        raise ValueError('Tokenizer fingerprint mismatch.')
    snapshot=source/'capture'/sid/'all_layers.gkv.gz'
    if sha(snapshot)!=spec['sample_fingerprints'][sid]:raise ValueError('Capture fingerprint mismatch.')
    item=next(json.loads(line) for line in (source/'inputs.jsonl').read_text().splitlines() if json.loads(line)['id']==sid)
    message=item['user_prompt'];before='<bos><|turn>user\n'
    import hashlib
    capture=json.loads((source/'capture'/sid/'sample.json').read_text())
    if hashlib.sha256(message.encode()).hexdigest()!=capture['prompt_sha256']:
        raise ValueError('Benchmark prompt does not match the captured prompt.')
    tokenizer=Tokenizer.from_file(str(profile['tokenizer']))
    prefix=tokenizer.encode(before,add_special_tokens=False).ids
    encoded=tokenizer.encode(before+message+'<turn|>\n<|turn>model\n',add_special_tokens=False)
    positions=[i for i,(a,b) in enumerate(encoded.offsets) if a<len(before)+len(message) and b>len(before)]
    if not positions or positions!=list(range(len(prefix),len(prefix)+len(positions))) or encoded.ids[:len(prefix)]!=prefix:
        raise ValueError('Noncontiguous or inconsistent benchmark target boundaries.')
    observation=folder/'observations'/sid
    sections=json.loads(profile['manifest'].read_text())['model_sections']
    import_snapshot(snapshot,observation,sections,spec['model_sha256'],prefix,positions[0],len(positions),spec['layer_index'])
    truth=folder/'private_eval'/sid/'truth.json'
    write(truth,dict(token_ids=[encoded.ids[i] for i in positions],positions=positions))
    return observation,truth


def execute(folder):
    folder=Path(folder).resolve();spec=json.loads((folder/'experiment.json').read_text());rows=[]
    store=CaseStore();worker=None
    try:
        check_idle();spec.update(state='running',started_at=time.time());write(folder/'experiment.json',spec)
        for sid in spec['sample_ids']:
            check_idle();observation,truth=prepare_sample(spec,sid,folder)
            name='exp_'+uuid.uuid4().hex[:20]+'_'+sid
            store.create(name)
            fields=dict(model_name=spec['model_name'],observation=str(observation),methods=spec['methods'])
            if spec.get('max_queries'):fields['max_queries']=spec['max_queries']
            store.update(name,fields,source='frozen text experiment '+str(folder))
            store.plan(name)
            output=folder/'samples'/sid
            print(f'Sample {sid}: '+', '.join(spec['methods']),flush=True)
            worker=subprocess.Popen([sys.executable,'-u',str(HERE/'run_case.py'),name,'--run-dir',str(output)],cwd=ROOT)
            code=worker.wait();worker=None
            for method in spec['methods']:
                run=output/method
                if not (run/'result.json').exists():continue
                # Ground truth is first provided after recovery has exited.
                evaluation_code=subprocess.call([sys.executable,str(HERE/'run.py'),'evaluate','--run',str(run),'--truth',str(truth)],cwd=ROOT)
                subprocess.call([sys.executable,str(HERE/'run.py'),'decode','--run',str(run),'--model-name',spec['model_name']],cwd=ROOT)
                row=summarize_run(run);row.update(sample_id=sid,dataset=spec['dataset'],evaluation_exit_code=evaluation_code)
                row['candidates_csv']=str(run/'candidates.csv');export_candidates(run,run/'candidates.csv');rows.append(row)
                write(folder/'summary.json',rows)
            if code:raise RuntimeError(f'Sample {sid} failed (exit {code}); partial outputs retained.')
        spec['state']='complete'
    except BaseException as error:
        if worker and worker.poll() is None:
            import signal
            worker.send_signal(signal.SIGINT)
            try:worker.wait(timeout=50)
            except subprocess.TimeoutExpired:worker.terminate();worker.wait()
        spec.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        raise
    finally:
        # Include partial method outputs even if interrupted before evaluation.
        known={row['output'] for row in rows}
        for result in (folder/'samples').glob('*/*/result.json'):
            if str(result.parent.resolve()) not in known:
                row=summarize_run(result.parent);row.update(sample_id=result.parent.parent.name,dataset=spec['dataset'])
                row['candidates_csv']=str(result.parent/'candidates.csv');export_candidates(result.parent,result.parent/'candidates.csv');rows.append(row)
        import csv
        write(folder/'summary.json',rows)
        if rows:
            with (folder/'summary.csv').open('w',newline='') as handle:
                fields=list(dict.fromkeys(key for row in rows for key in row))
                writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        spec['finished_at']=time.time();write(folder/'experiment.json',spec)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,required=True)
    execute(parser.parse_args().experiment)
