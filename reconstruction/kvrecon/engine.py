import json
from pathlib import Path
import time
from dataclasses import asdict
import numpy as np
from .common import dump
from .search import kv_distance, recover_position, acceptance, measured_values
from .media import MediaUnavailable


def _reconstruct(backend, targets, positions, prefix, config, output, surrogate=None, max_positions=None, candidate_ids=None, media_router=None, emit=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/"trace.jsonl").exists() or (output/"result.json").exists():
        raise FileExistsError("Refusing to overwrite an existing recovery")
    if media_router and any(media_router.kind(t) or media_router.kind(t,'soft') for t in prefix):
        raise ValueError('Media in public prefix requires representation replay; include its start marker in recovery targets instead')
    positions=np.asarray(positions,dtype=np.int64)
    if emit:
        emit(dict(type="start", config=asdict(config), target_positions=len(positions),
            planned_positions=min(len(positions), max_positions) if max_positions else len(positions)))
    started=time.perf_counter();backend.reset(prefix);prefix_seconds=time.perf_counter()-started
    recovered=[];recovered_positions=[];media_segments=[];status="complete";count=len(positions)
    expected_end=None;index=0;attempted=0;covered=0
    if max_positions is not None:
        if max_positions<=0:
            raise ValueError("max_positions must be positive")
        count=min(count,max_positions)
    started=time.perf_counter();decoder_start=backend.decoder_seconds
    with (output/"trace.jsonl").open("w") as f:
        while index<count:
            position=int(positions[index])
            attempted+=1;next_index=index+1
            if backend.position!=position:
                raise ValueError("Reconstructed prefix position disagrees with target")
            if emit:emit(dict(type="position_start", index=index, position=position))
            tick=time.perf_counter()
            order=np.arange(backend.vocab_size) if config.method=='exhaustive' else backend.probabilities()
            probability_seconds=time.perf_counter()-tick
            row=recover_position(backend,targets[index],order,index,config,surrogate,candidate_ids,on_batch=emit)
            row.update(position=position,probability_seconds=probability_seconds)
            if row["accepted_token_id"] is None:
                status="unresolved"
            elif media_router and media_router.kind(row['accepted_token_id'],'soft'):
                status='unresolved_media'
                row['media_error']='Soft media slots cannot be committed as text token IDs'
            elif expected_end is not None and row['accepted_token_id']!=expected_end:
                status='unresolved_media'
                row['media_error']='Recovered media end marker disagrees with the selected model rule'
            elif media_router and position in media_router.spans and media_router.kind(row['accepted_token_id'])!=media_router.spans[position]['modality']:
                status='unresolved_media'
                row['media_error']='Recovered marker disagrees with supplied media span'
            else:
                tick=time.perf_counter()
                commit=backend.accept_query if config.method=='collision_plus_cpa' else backend.commit
                codes=commit(row["accepted_token_id"])
                actual=measured_values(backend,codes,config)[0];distance=float(kv_distance(actual,targets[index],config.metric))
                row.update(commit_seconds=time.perf_counter()-tick,commit_distance=distance,commit_l2=float(np.linalg.norm(actual-targets[index])),
                    commit_matches_query=bool(np.array_equal(codes[0],np.asarray(row["accepted_codes"],dtype=codes.dtype))))
                # Don't advance to later positions if the single-token commit violates acceptance.
                valid,components=acceptance(actual,targets[index],config)
                if config.method=='collision_plus_cpa':row['accepted_kv_l2']=components.tolist()
                if not valid or (config.method=='collision_plus_cpa' and not row['commit_matches_query']):
                    status="commit_mismatch"
                else:
                    recovered.append(row["accepted_token_id"])
                    recovered_positions.append(position);covered+=1;expected_end=None
                    if media_router and media_router.kind(row['accepted_token_id']):
                        try:
                            span=media_router.spans.get(position)
                            if span and span['end']>int(positions[count-1]):
                                raise MediaUnavailable('media_position_limit: whole span and end marker must fit the position budget')
                            segment=media_router.recover(backend,row['accepted_token_id'],position,
                                targets,positions,config,output/'media'/str(position))
                            media_segments.append(segment);row['media']=segment
                            next_index=int(np.searchsorted(positions,segment['end']))
                            covered+=segment['end']-segment['start']
                            expected_end=media_router.rules[segment['modality']]['end']
                        except MediaUnavailable as error:
                            status='unresolved_media';row['media_error']=str(error)
            f.write(json.dumps(row,allow_nan=False)+"\n");f.flush()
            if emit:
                emit(dict(type="position_end", index=index, position=position,
                    accepted_token_id=row["accepted_token_id"], accepted_l2=row["accepted_l2"],
                    accepted_distance=row["accepted_distance"],distance_metric=config.metric,
                    commit_l2=row.get("commit_l2"),commit_distance=row.get("commit_distance"), committed=position in recovered_positions,
                    queries=row["queries"], stop_reason=row["stop_reason"], status=status,
                    search_seconds=row["search_seconds"], stages=row["stages"]))
            print(f"position={position} accepted={row['accepted_token_id']} queries={row['queries']} {config.metric.upper()}={row['accepted_distance']}",flush=True)
            if status!="complete":
                break
            index=next_index
    if status=="complete" and count<len(positions):
        status="position_limit"
    result=dict(status=status,recovered_token_ids=recovered,target_positions=len(positions),
        attempted_positions=attempted,covered_positions=covered,config=asdict(config),
        recovered_token_positions=recovered_positions,media_segments=media_segments,
        public_prefix_tokens=len(prefix),public_prefix_seconds=prefix_seconds,
        online_seconds=time.perf_counter()-started,decoder_seconds=backend.decoder_seconds-decoder_start,
        own_recovered_prefix=True,target_cache_used_as_prefix=False,
        correctness="Threshold acceptance only; exact identity requires separate evaluation",uniqueness_proven=False)
    dump(output/"result.json",result)
    if emit:emit(dict(type="finish", result=result))
    return result


def reconstruct(backend, targets, positions, prefix, config, output, surrogate=None,
                max_positions=None, candidate_ids=None, media_router=None, live_events=False):
    """Optional telemetry is observation-only; it never reads evaluation labels."""
    from contextlib import nullcontext
    from .events import EventLog
    context=EventLog(Path(output)/"events.jsonl") if live_events else nullcontext(None)
    with context as emit:
        return _reconstruct(backend, targets, positions, prefix, config, output, surrogate,
            max_positions, candidate_ids, media_router, emit)
