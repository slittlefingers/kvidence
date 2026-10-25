"""Persist measured experiment summaries and lossless candidate distance exports."""
import csv
import json
from datetime import datetime,timezone
from pathlib import Path
from casework import HERE,write

RESULTS_ROOT=HERE/'results/text_experiments'


def read(path,default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def summarize_run(path):
    path=Path(path);result=read(path/'result.json',{});protocol=read(path/'protocol.json',{})
    evaluation=read(path/'evaluation.json',{})
    queries=0;positions=0
    trace=path/'trace.jsonl'
    if trace.exists():
        with trace.open() as handle:
            for line in handle:
                row=json.loads(line);queries+=row.get('queries',len(row.get('queried_token_ids',[])));positions+=1
    else:queries=evaluation.get('total_queries')
    seconds=result.get('online_seconds');accepted=len(result.get('recovered_token_ids',[]))
    return dict(output=str(path.resolve()),method=result.get('config',protocol.get('config',{})).get('method'),
        status=result.get('status','unrecorded'),accepted_tokens=accepted,attempted_positions=positions,
        target_positions=result.get('target_positions'),correct_tokens=evaluation.get('correct_ids'),
        truth_tokens=evaluation.get('truth_tokens'),total_queries=queries,online_seconds=seconds,
        accepted_tokens_per_second=accepted/seconds if seconds and seconds>0 else None,
        queries_per_second=queries/seconds if queries is not None and seconds and seconds>0 else None,
        device_setup_seconds=result.get('device_setup_seconds'),public_prefix_seconds=result.get('public_prefix_seconds'),
        surrogate_setup_seconds=result.get('surrogate_setup_seconds'),metric=protocol.get('metric'),
        distance_space=protocol.get('config',{}).get('space'),
        trace=str(trace.resolve()) if trace.exists() else None,
        events=str((path/'events.jsonl').resolve()) if (path/'events.jsonl').exists() else None)


def export_candidates(source,destination):
    """Stream every recorded candidate; keep missing distances blank, never invent them."""
    fields=['position','query_index','token_id','distance','stage','accepted','search_seconds','commit_seconds']
    count=0
    with Path(destination).open('w',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader()
        trace=Path(source)/'trace.jsonl'
        if not trace.exists():return 0
        with trace.open() as handle:
            for line in handle:
                row=json.loads(line);ids=row.get('queried_token_ids',[]);distances=row.get('distances',[])
                stages=[]
                for stage in row.get('stages',[]):stages.extend([stage.get('phase',stage.get('name',''))]*stage.get('queries',0))
                for index,token in enumerate(ids):
                    distance=distances[index] if index<len(distances) else None
                    if isinstance(distance,(dict,list)):distance=json.dumps(distance,separators=(',',':'))
                    writer.writerow(dict(position=row.get('position',row.get('index')),query_index=index,
                        token_id=token,distance=distance,stage=stages[index] if index<len(stages) else '',
                        accepted=token==row.get('accepted_token_id'),search_seconds=row.get('search_seconds'),commit_seconds=row.get('commit_seconds')))
                    count+=1
    return count


def save_scope(catalog,dataset,sample=None,root=RESULTS_ROOT):
    spec=catalog.specification(dataset,sample)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder=Path(root)/(stamp+'-'+spec['dataset']+'-'+(str(sample).zfill(4) if sample else 'all'))
    folder.mkdir(parents=True,exist_ok=False)
    rows=[]
    for sample_id in spec['sample_ids']:
        for record in catalog.records.get((spec['dataset'],sample_id),[]):
            row=summarize_run(record['path']);row.update(dataset=spec['dataset'],sample_id=sample_id)
            detail=folder/'details'/sample_id/(record['path'].name+'-'+str(len(rows)))
            detail.mkdir(parents=True)
            row['candidates_csv']=str(detail/'candidates.csv')
            row['exported_candidates']=export_candidates(record['path'],detail/'candidates.csv')
            write(detail/'summary.json',row);rows.append(row)
    spec.update(state='archived_results_export',created_at=datetime.now(timezone.utc).isoformat(),
        recorded_runs=len(rows),recorded_samples=len({r['sample_id'] for r in rows}),
        note='Archive export only; no NPU execution. Accepted tokens are not independently verified correct tokens. Per-position timing repeats on candidate rows and must not be summed.')
    write(folder/'experiment.json',spec);write(folder/'summary.json',rows)
    with (folder/'summary.csv').open('w',newline='') as handle:
        if rows:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return folder,rows


def report(folder,rows):
    lines=[f'Results: {folder}',f'Recorded runs: {len(rows)}','',
        'Sample  Method                 Accepted Correct    Queries  Online(s)  Token/s']
    for row in rows:
        correct='—' if row['correct_tokens'] is None else f"{row['correct_tokens']}/{row['truth_tokens']}"
        seconds=row['online_seconds'];speed=row['accepted_tokens_per_second']
        lines.append(f"{row['sample_id']}    {str(row['method']):22} {row['accepted_tokens']:>8} {correct:>7} {str(row['total_queries']):>10} {seconds if seconds is not None else '—':>10} {round(speed,3) if speed is not None else '—':>8}")
    return '\n'.join(lines+['','summary.json / summary.csv: measurements and source paths.',
        'details/<sample>/<run>/candidates.csv: every recorded candidate distance.',
        'Source trace/events retain full search stages and timing.',
        'Missing samples/methods are unrecorded. No new recovery was started.'])
