"""Paired runs of existing guided and Collision+ (L2-adapted) strategies.

Each method runs in a fresh subprocess/backend. No labels enter recovery.
The first-layer V-only shortcut is deliberately not substituted for either arm.
"""
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .common import ROOT, dump, digest, resolve_distance_space
from .observation import load_observation

METHODS=('collision_plus','guided')
SHARED=('model_config','model','manifest','device_model','tokenizer','batch_signature',
        'embeddings','vocabulary','vocab_size','adb','serial','transport','native_mode','epsilon','distance_metric','distance_space','batch_size',
        'initial_candidates','prediction_candidates','proposals','seed','max_queries','max_positions')


def compare(args, sections, model_hash):
    from .cli import private_guard
    private_guard()
    meta,codes,_,positions,prefix=load_observation(args.observation,distance_space="auto")
    args.distance_space=resolve_distance_space(getattr(args,'distance_space','auto'),codes.dtype)
    if meta['model_sha256']!=model_hash:
        raise ValueError('Observation and selected model do not match')
    if not prefix or not np.array_equal(positions,np.arange(len(prefix),len(prefix)+len(positions))):
        raise ValueError('Paired K+V search requires a known prefix and contiguous positions, including at layer 0')
    if args.repeats<=0:raise ValueError('repeats must be positive')
    baseline=getattr(args,'baseline','collision_plus_cpa')
    cpa=baseline=='collision_plus_cpa'
    if cpa and not getattr(args,'calibration',None):raise ValueError('CPA comparison requires independent --calibration')
    if not cpa and getattr(args,'calibration',None):raise ValueError('L2-adapted baseline does not use CPA calibration')
    methods=(baseline,'guided')
    args.output.mkdir(parents=True,exist_ok=False)
    common=['--observation',str(args.observation.resolve())]
    for key in SHARED:
        value=getattr(args,key,None)
        if value is not None:common.extend(['--'+key.replace('_','-'),str(value)])
    if args.no_install:common.append('--no-install')
    if getattr(args,'live_events',False):common.append('--live-events')
    plan=dict(schema=1,model_sha256=model_hash,observation_metadata_sha256=digest(args.observation/'metadata.json'),
        metric=('CPA: separate calibrated original-value K/V; guided: ' if cpa else '')+f"K+V {args.distance_metric.upper()} ({args.distance_space})",epsilon=args.epsilon,batch_size=args.batch_size,
        methods=list(methods),baseline_label='Collision+ (CPA NPU port)' if cpa else 'Collision+ (L2-adapted probability-order baseline)',
        calibration_sha256=digest(args.calibration) if cpa else None,
        first_layer_shortcut=False,truth_available_to_recovery=False,
        note='Fresh processes and prefixes; alternating order reduces but does not remove thermal/cache/order effects',runs=[])
    dump(args.output/'comparison.json',plan)
    for repetition in range(args.repeats):
        first=list(methods) if args.first=='collision_plus' else list(reversed(methods))
        order=first if repetition%2==0 else list(reversed(first))
        for method in order:
            name=f'{repetition+1:03d}_{method}'
            command=[sys.executable,str(ROOT/'run.py'),'run',*common,'--method',method,'--output',str(args.output/name)]
            if method=='collision_plus_cpa':
                command.extend(['--calibration',str(args.calibration.resolve()),'--stop-partition',str(args.stop_partition)])
                if args.allow_pilot_calibration:command.append('--allow-pilot-calibration')
            row=dict(repetition=repetition+1,method=method,directory=name,command=command,status='running')
            plan['runs'].append(row);dump(args.output/'comparison.json',plan)
            print(f'comparison repetition={repetition+1} method={method}',flush=True)
            started=time.perf_counter()
            with (args.output/(name+'.log')).open('wb') as log:
                process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            row.update(exit_code=process.returncode,wall_seconds=time.perf_counter()-started,
                       status='finished' if process.returncode==0 else 'failed')
            dump(args.output/'comparison.json',plan)
            if process.returncode:
                summarize(args.output)
                raise RuntimeError(f'{method} failed; inspect {name}.log. No performance conclusion is generated.')
    summarize(args.output)


def summarize(path):
    path=Path(path);plan=json.loads((path/'comparison.json').read_text());rows=[];reference=None
    for run in plan['runs']:
        folder=path/run['directory'];row={k:run.get(k) for k in ('repetition','method','directory','status','wall_seconds')}
        if (folder/'result.json').exists():
            r=json.loads((folder/'result.json').read_text());p=json.loads((folder/'protocol.json').read_text())
            signature=[p['model_sha256'],p['manifest_sha256'],p['model_config'],p['observation_metadata'],
                       p['candidate_vocabulary'],p['config']['epsilon'],p['config']['batch_size'],
                       p['config']['max_queries'],p['config'].get('distance_metric','l2'),p.get('max_positions'),p['source_hashes'],r['target_positions']]
            device=folder/'device/device.json'
            if device.exists():
                device=json.loads(device.read_text())
                signature.append({k:device.get(k) for k in ('model_sha256','runtime_sha256','test_apk_sha256','batch_signature','batch_capacity','transport','ranking_location','native_mode')})
            if reference is not None and signature!=reference:
                raise ValueError('Paired protocols differ in model/observation/pool/threshold/batch/budget')
            reference=signature
            if p['config']['method']!=run['method'] or r['config']['method']!=run['method']:
                raise ValueError('Method routing differs from comparison plan')
            if run['method']=='collision_plus_cpa' and p.get('calibration_sha256')!=plan.get('calibration_sha256'):
                raise ValueError('CPA calibration differs from comparison plan')
            trace=[json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()]
            row.update(status=r['status'] if run['status']=='finished' else run['status'],recovery_status=r['status'],accepted_tokens=len(r['recovered_token_ids']),
                metric=p.get('metric','original_value_kv_l2'),distance_space=p['config'].get('distance_space','original'),
                attempted_positions=r['attempted_positions'],total_queries=sum(x['queries'] for x in trace),
                online_seconds=r['online_seconds'],decoder_seconds=r['decoder_seconds'],
                public_prefix_seconds=r['public_prefix_seconds'],device_setup_seconds=r.get('device_setup_seconds'),
                surrogate_setup_seconds=r.get('surrogate_setup_seconds'),
                probability_seconds=sum(x['probability_seconds'] for x in trace),
                ranking_seconds=sum(x['ranking_seconds'] for x in trace),
                query_transport_seconds=sum(x['query_transport_seconds'] for x in trace))
            evaluation=folder/'evaluation.json'
            row['evaluation']=json.loads(evaluation.read_text()) if evaluation.exists() else None
        rows.append(row)
    dump(path/'summary.json',dict(rows=rows,metric=plan['metric'],epsilon=plan['epsilon'],
        baseline_label=plan['baseline_label'],accuracy_requires_independent_evaluation=True,
        note=('Compare accuracy/completeness before timing. CPA includes offline calibration; NPU port is not GPU reproduction.'
              if 'collision_plus_cpa' in plan['methods'] else
              'Shared configured K+V distance criterion (see each protocol for units); no CPA calibration. Probability-order adaptation, not the original CPA baseline.')))
    lines=['# Guided vs '+plan['baseline_label'],'',
           '相同观测、模型、词表和候选批量；每次独立重建前缀。判定口径：'+plan['metric']+'。', '',
           '| Repeat | Method | Status | Accepted | Correct | Queries | Online s | Decoder s | Process wall s |',
           '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        e=row.get('evaluation')
        def number(key):return f"{row[key]:.4f}" if isinstance(row.get(key),(int,float)) else '—'
        lines.append(f"| {row['repetition']} | {row['method']} | {row['status']} | {row.get('accepted_tokens','—')} | {e['correct_ids'] if e else '未评价'} | {row.get('total_queries','—')} | {number('online_seconds')} | {number('decoder_seconds')} | {number('wall_seconds')} |")
    lines.extend(['','Accepted 不等于正确；正确性来自单独评价。Online 不含初始化和公开前缀预填；Process wall 包含命令内加载、校验、计算、输出和退出。',
                  '这两个方法均使用 K+V；第一层 V-only 查表的成本和阈值含义不同，不混入本表。',
                  '单次或极短样本不能作为速度优势证据；选择规则也不同：基线接受概率序中首个通过者，guided 选择完整阶段中距离最小的通过者。'])
    (path/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print('Comparison summary:',path/'SUMMARY.md')


def evaluate_comparison(args):
    from .cli import evaluate
    from types import SimpleNamespace
    plan=json.loads((args.comparison/'comparison.json').read_text())
    for run in plan['runs']:
        if run['status']!='finished':raise ValueError('Cannot evaluate an incomplete comparison')
        evaluate(SimpleNamespace(run=args.comparison/run['directory'],truth=args.truth))
    summarize(args.comparison)
