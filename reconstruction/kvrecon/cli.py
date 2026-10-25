import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from .common import ROOT, dump, digest, distance_values, resolve_distance_space, validate_quantization_grid
from .observation import load_observation
from .adapters import get_adapter
from .backend import CandidateBatchBackend, validate_contract
from .search import SearchConfig
from .engine import reconstruct
from .config import DEFAULT_MODEL_CONFIG, MODEL_CONFIGS, load_model_config, load_vocabulary


def private_guard():
    def audit(event,args):
        if event=="open" and isinstance(args[0],(str,bytes,Path)):
            parts=Path(os.fsdecode(args[0])).resolve().parts
            if "private_eval" in parts or "datasets" in parts:
                raise PermissionError("Recovery cannot read private evaluation labels or source datasets")
    sys.addaudithook(audit)


def backend(args, sections, layer, output, model_hash):
    options={key:getattr(args,key) for key in ('transport','native_mode') if getattr(args,key,None) is not None}
    return args.adapter.open(sections,args.resolved_model,layer,output,model_hash,
        adb=args.adb,serial=args.serial,install=not args.no_install,
        **options)


def prepare(args,sections,model_hash):
    prefix=json.loads(args.prefix_json.read_text())
    spans=None
    if args.media_spans_json:
        from .media import resolve_rules, MediaRouter
        if args.layer==0:raise ValueError('First-layer text V lookup does not support media spans')
        spans=json.loads(args.media_spans_json.read_text())
        if not isinstance(spans,list):raise ValueError('Media spans must be a JSON array')
        MediaRouter(resolve_rules(args.resolved_model,args.tokenizer),spans,np.arange(args.start,args.start+args.count))
    args.adapter.import_observation(args.snapshot,args.output,sections,model_hash,prefix,args.start,args.count,args.layer)
    if spans is not None:
        path=args.output/'metadata.json';meta=json.loads(path.read_text())
        meta.update(media_spans=spans,media_spans_sha256=digest(args.media_spans_json))
        dump(path,meta)
    print("Prepared observation:",args.output)


def fixture(args,sections,model_hash):
    args.output.mkdir(parents=True,exist_ok=False)
    prefix=json.loads(args.prefix_json.read_text())
    if not prefix:
        raise ValueError("Public prefix must include at least BOS")
    explicit=json.loads(args.token_ids_json.read_text()) if args.token_ids_json else None
    ids=[];vectors=[]
    with backend(args,sections,args.layer,args.output/"capture_device",model_hash) as b:
        b.reset(prefix)
        for i in range(len(explicit) if explicit is not None else args.greedy_tokens):
            token=int(explicit[i]) if explicit is not None else int(b.probabilities()[0])
            ids.append(token);vectors.append(b.commit(token)[0])
    private=args.output/"private_eval";private.mkdir()
    dump(private/"truth.json",dict(token_ids=ids,public_prefix_ids=prefix,
        note="Controlled same-device fixture; not a recovery benchmark or encrypted-file result"))
    args.adapter.save_fixture(args.output/"observation",vectors,np.arange(len(prefix),len(prefix)+len(ids)),prefix,
        args.layer,sections,model_hash,dict(kind="controlled backend single-token decode fixture",
        selection="explicit IDs" if explicit is not None else "greedy next-token; intentionally easy integration smoke"))
    print("Fixture saved:",args.output)


def first_v_context(args, sections, model_hash):
    from .first_v import table_identity
    factory=getattr(args.adapter,'first_v_spec',None)
    if not callable(factory):
        raise ValueError("Adapter does not declare first-layer V lookup support")
    spec=factory(sections,args.resolved_model)
    dims=args.adapter.dimensions(sections,args.resolved_model)
    ids,pool_meta=load_vocabulary(args.vocabulary,dims['vocab_size'])
    identity=table_identity(model_hash,digest(args.manifest),args.resolved_model['adapter'],args.batch_signature,ids)
    return spec,ids,pool_meta,identity


def build_v_table(args, sections, model_hash):
    from .first_v import build_table
    spec,ids,_,identity=first_v_context(args,sections,model_hash)
    if args.output.exists():raise FileExistsError("Preserve the existing table directory")
    with backend(args,sections,0,args.output.parent/(args.output.name+'_device'),model_hash) as b:
        build_table(args.output,b,spec,ids,identity)
    print("First V table saved:",args.output)


def cpa_identity(args, model_hash, meta):
    return dict(model_sha256=model_hash,manifest_sha256=digest(args.manifest),
                adapter=args.resolved_model['adapter'],batch_signature=args.batch_signature,
                layer_index=meta['layer_index'],quantization=meta['quantization'])


def runtime_identity(directory):
    path=Path(directory)/'device.json'
    if not path.exists():return None
    record=json.loads(path.read_text())
    return {k:record.get(k) for k in ('runtime_sha256','test_apk_sha256','batch_signature','native_mode')}


def calibrate_cpa(args, sections, model_hash):
    from .cpa import collect
    if args.rank<2 or not 1<=args.chunk_size<=4096:raise ValueError('rank >= 2 and chunk size in 1..4096 required')
    meta,_,targets,positions,prefix=load_observation(args.observation)
    if meta['model_sha256']!=model_hash:raise ValueError('Calibration observation model mismatch')
    tokens=json.loads(args.known_tokens_json.read_text())
    if isinstance(tokens,dict):tokens=tokens['token_ids']
    if not isinstance(tokens,list) or any(type(t) is not int for t in tokens):raise ValueError('Known tokens must be integer IDs')
    dims=args.adapter.dimensions(sections,args.resolved_model)
    if len(tokens)!=len(targets) or any(not 0<=t<dims['vocab_size'] for t in tokens):raise ValueError('Calibration labels/positions mismatch')
    ids=np.arange(dims['vocab_size'])
    if args.pilot_candidates is not None:
        count=args.pilot_candidates
        if count<max(2,len(set(tokens))) or count>len(ids):raise ValueError('Invalid pilot candidate count')
        fixed=np.unique(tokens);rest=np.setdiff1d(ids,fixed)
        ids=np.sort(np.concatenate([fixed,np.random.default_rng(args.seed).choice(rest,count-len(fixed),replace=False)]))
    identity=cpa_identity(args,model_hash,meta)
    identity.update(observation_sha256=digest(args.observation/'observed.npz'),
                    observation_metadata_sha256=digest(args.observation/'metadata.json'))
    # Each resumed backend gets separate logs; completed numeric checkpoints
    # are reused only when model/runtime/calibration identities all match.
    device_dir=args.output/('device_'+str(time.time_ns()))
    with backend(args,sections,meta['layer_index'],device_dir,model_hash) as b:
        identity['runtime']=runtime_identity(device_dir)
        asset=collect(b,targets,positions,prefix,tokens,ids,meta['quantization'][0]['width'],
                      args.output,identity,args.rank,args.chunk_size)
    print(json.dumps(dict(calibration=str(args.output/'calibration.json'),coverage=asset['coverage'],fit=asset['fit']),indent=2))


def run_first_v(args, sections, model_hash, meta, targets, positions, distance_space):
    import hashlib
    from .first_v import build_table,load_table,match_observation
    if meta['layer_index']!=0:raise ValueError("First V lookup requires layer index 0")
    if not np.isfinite(args.epsilon) or args.epsilon<=0 or (args.max_positions is not None and args.max_positions<=0):
        raise ValueError("First V lookup requires positive finite epsilon and a positive position limit")
    spec,ids,pool_meta,identity=first_v_context(args,sections,model_hash)
    if meta['quantization']!=spec['quantization']:
        # Older INT8 observations omitted dtype; normalize that compatible case.
        normalized=[dict(q,dtype=q.get('dtype','int8')) for q in meta['quantization']]
        if normalized!=spec['quantization']:raise ValueError("Observed first KV metadata differs from table")
    if args.output.exists():raise FileExistsError("Preserve the existing recovery output")
    key=hashlib.sha256(json.dumps(dict(identity=identity,spec=spec),sort_keys=True).encode()).hexdigest()[:24]
    table=args.v_table or (args.resolved_model.get('first_v_table') if pool_meta['kind']=='full_original' else None) or ROOT/'assets/first_v'/key
    if not table.exists():
        if args.v_table:raise FileNotFoundError("Explicit V table missing; create it with build-v-table")
        with backend(args,sections,0,table.parent/(table.name+'_device'),model_hash) as b:
            build_table(table,b,spec,ids,identity)
    table_meta,codes=load_table(table,identity,spec,ids)
    result=match_observation(args.output,codes,ids,table_meta,targets,positions,args.epsilon,args.max_positions,distance_space=distance_space,distance_metric=args.distance_metric)
    dump(args.output/'protocol.json',dict(schema=1,method='first_v',observation_metadata=meta,
        model_sha256=model_hash,manifest_sha256=digest(args.manifest),adapter=args.resolved_model['adapter'],
        model_config={k:str(v) if isinstance(v,Path) else v for k,v in args.resolved_model.items()},
        model_config_sha256=digest(args.model_config),tokenizer_sha256=digest(args.tokenizer),
        candidate_vocabulary=pool_meta,v_table=str(table.resolve()),v_table_metadata_sha256=digest(table/'metadata.json'),
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(ROOT.rglob('*')) if p.suffix in ('.py','.cc','.h','.sh')},
        metric=f'{distance_space}_v_{args.distance_metric}',distance_space=distance_space,epsilon=args.epsilon,
        private_labels_allowed=False,ambiguity_policy='Keep all passing IDs; accept only a single passing ID',
        note='V-only threshold is not numerically equivalent to the deep K+V threshold'))
    print(json.dumps(result,indent=2))


def run(args,sections,model_hash):
    private_guard()
    meta,codes,targets,positions,prefix=load_observation(args.observation,distance_space="auto")
    if meta["model_sha256"]!=model_hash:
        raise ValueError("Observation and selected model do not match")
    method=('first_v' if meta['layer_index']==0 else 'guided') if args.method=='auto' else args.method
    space="original" if method=='collision_plus_cpa' else resolve_distance_space(args.distance_space,codes.dtype)
    # Native integer targets remain in code units throughout guided recovery.
    if space != resolve_distance_space("auto",codes.dtype):
        targets=distance_values(codes,meta['quantization'],space)
    from .media import resolve_rules, MediaRouter
    media_rules=resolve_rules(args.resolved_model,args.tokenizer)
    media_router=MediaRouter(media_rules,meta.get('media_spans',[]),positions) if media_rules else None
    if meta.get('media_spans') and (not media_router or method=='first_v'):
        raise ValueError('Media spans require a media-capable deep-layer route')
    if method=='first_v':
        return run_first_v(args,sections,model_hash,meta,targets,positions,space)
    calibration=None
    if method=='collision_plus_cpa':
        from .cpa import load_calibration
        if args.calibration is None:raise ValueError('CPA requires --calibration from an independent known-text experiment')
        calibration=load_calibration(args.calibration,cpa_identity(args,model_hash,meta),
                                     digest(args.observation/'observed.npz'),args.allow_pilot_calibration)
    elif args.calibration is not None:
        raise ValueError('--calibration only applies to collision_plus_cpa')
    config=SearchConfig(method=method,distance_space=space,distance_metric=args.distance_metric,epsilon=args.epsilon,batch_size=args.batch_size,
        probability_seeds=args.prediction_candidates,random_seeds=args.initial_candidates-args.prediction_candidates,
        proposals=args.proposals,seed=args.seed,max_queries=args.max_queries,
        kv_thresholds=tuple(calibration['fit']['thresholds']) if calibration else None,
        k_width=calibration['k_width'] if calibration else None,stop_partition=args.stop_partition).validate()
    dims=args.adapter.dimensions(sections,args.resolved_model)
    if positions[-1]>=dims["context_capacity"]:
        raise ValueError("Observation exceeds selected model context capacity")
    pool,pool_metadata=load_vocabulary(args.vocabulary,dims["vocab_size"])
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();surrogate=None
    if config.method=="guided":
        from .surrogate import ExistingRidge
        surrogate=ExistingRidge(args.embeddings,dims["vocab_size"],dims["embedding_dim"])
    surrogate_seconds=time.perf_counter()-started
    dump(args.output/"protocol.json",dict(schema=2,observation_metadata=meta,config=asdict(config),
        manifest_sha256=digest(args.manifest),model_sha256=model_hash,
        model_config={k:str(v) if isinstance(v,Path) else v for k,v in args.resolved_model.items()},
        model_config_sha256=digest(args.model_config),model_dimensions=dims,
        adapter=args.resolved_model["adapter"],
        media_rules=media_rules,media_boundary_policy='Known spans; dispatch only after accepted start token',
        capabilities=asdict(args.adapter.capabilities(sections,args.resolved_model,meta["layer_index"])),
        tokenizer_sha256=digest(args.tokenizer),candidate_vocabulary=pool_metadata,
        original_vocabulary=dict(start=0,stop_exclusive=dims["vocab_size"],filtered=False),
        max_positions=args.max_positions,metric="separate_k_v_l2_float32" if calibration else f"{space}_kv_{config.metric}",distance_space=space,
        implementation=("collision_plus_cpa_npu_port" if calibration else "collision_plus_l2_adapted" if method=="collision_plus" else "token_id_exhaustive" if method=='exhaustive' else "kv_guided_ridge"),
        calibration=calibration,calibration_sha256=digest(args.calibration) if calibration else None,
        selection_policy=("first passing ID in model-probability order" if method.startswith('collision_plus') else "minimum distance in first passing token-ID batch" if method=='exhaustive' else "minimum distance across complete accepting stage"),
        initial_candidates=args.initial_candidates,prediction_candidates=args.prediction_candidates,
        random_candidates=config.random_seeds,probability_source="unused; ascending original token IDs" if method=='exhaustive' else "selected backend model's full probability ranking",
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(ROOT.rglob("*")) if p.suffix in (".py",".cc",".h",".sh")},
        surrogate=surrogate.metadata if surrogate else None,surrogate_setup_seconds=surrogate_seconds,
        candidate_quantization=("Native integer codes widened to float64; no dequantization, no rounding of surrogate predictions" if space=='lsb' else "Original-value KV; integer codes dequantized, floating KV used directly"),
        stopping=("K < tau_K + .01*tau_K AND V < tau_V + .01*tau_V; epsilon unused" if calibration else f"strict {space} K+V {config.metric.upper()} < epsilon; no default query cap; threshold is not proof of identity"),
        live_events=args.live_events,private_labels_allowed=False))
    with backend(args,sections,meta["layer_index"],args.output/"device",model_hash) as b:
        candidate_quant=getattr(b,'quant',None)
        if candidate_quant is not None:
            validate_quantization_grid(meta['quantization'],candidate_quant)
        elif space=='lsb':
            raise ValueError('LSB recovery requires backend.quant to verify the native quantization grid')
        if calibration and calibration['identity'].get('runtime')!=runtime_identity(args.output/'device'):
            raise ValueError('CPA calibration runtime differs from recovery runtime')
        result=reconstruct(CandidateBatchBackend(b,config.batch_size),targets,positions,prefix,config,args.output,surrogate,args.max_positions,pool,media_router,live_events=args.live_events)
        result.update(device_setup_seconds=b.setup_seconds,surrogate_setup_seconds=surrogate_seconds)
        if calibration:result['cache_acceptance_policy']=getattr(b,'cache_acceptance_policy','retained candidate cache')
        dump(args.output/"result.json",result)
    print(json.dumps(result,indent=2))


def check_adapter(args, sections, model_hash):
    args.output.mkdir(parents=True,exist_ok=False)
    prefix=json.loads(args.prefix_json.read_text())
    ids=json.loads(args.candidate_ids_json.read_text())
    caps=args.adapter.capabilities(sections,args.resolved_model,args.layer)
    if not isinstance(prefix,list) or not prefix or any(type(x) is not int or not 0<=x<caps.vocab_size for x in prefix):
        raise ValueError("Contract prefix must contain valid original IDs")
    if not isinstance(ids,list) or any(type(x) is not int for x in ids):
        raise ValueError("Contract candidates must be an original ID array")
    if len(prefix)+1>=caps.context_capacity:
        raise ValueError("Contract test needs room for a commit and next-position query")
    try:
        with backend(args,sections,args.layer,args.output/"device",model_hash) as b:
            result=validate_contract(b,prefix,ids,atol=args.atol,rtol=args.rtol)
        result.update(adapter=args.resolved_model["adapter"],model_sha256=model_hash,
                      layer=args.layer,capabilities=asdict(caps))
        dump(args.output/"validation.json",result)
        print(json.dumps(result,indent=2))
    except Exception as error:
        dump(args.output/"validation.json",dict(passed=False,error=str(error),adapter=args.resolved_model["adapter"]))
        raise


def evaluate(args):
    result=json.loads((args.run/"result.json").read_text())
    truth_record=json.loads(args.truth.read_text());truth=truth_record["token_ids"]
    rows=[json.loads(x) for x in (args.run/"trace.jsonl").read_text().splitlines()]
    recovered=result["recovered_token_ids"]
    if any(len(r.get("queried_token_ids",[]))!=len(set(r.get("queried_token_ids",[]))) for r in rows):
        raise ValueError("Repeated candidate in trace")
    passing=[r for r in rows if r["accepted_token_id"] is not None]
    if result['config'].get('method')=='collision_plus_cpa':
        tau=np.asarray(result['config']['kv_thresholds']);limit=tau+.01*tau
        if any(not np.all(np.asarray(r['accepted_kv_l2'])<limit) for r in passing):
            raise ValueError('Accepted CPA candidate violates component thresholds')
    elif any(not r.get("accepted_distance",r.get("accepted_l2"))<result["config"]["epsilon"] for r in passing):
        raise ValueError("Accepted candidate violates threshold")
    if result.get('media_segments'):
        truth_positions=truth_record.get('positions')
        if (not isinstance(truth_positions,list) or len(truth_positions)!=len(truth)
                or any(type(p) is not int for p in truth_positions) or len(set(truth_positions))!=len(truth_positions)):
            raise ValueError('Multimodal text evaluation requires explicit unique positions paired with truth token_ids')
        truth_map=dict(zip(truth_positions,truth))
        recovered_positions=result['recovered_token_positions']
        correct=sum(t==truth_map.get(p) for p,t in zip(recovered_positions,recovered))
        exact=recovered_positions==truth_positions and recovered==truth
    else:
        correct=sum(x is not None and x==y for x,y in zip(recovered,truth));exact=recovered==truth
    accepted=sum(x is not None for x in recovered)
    evaluation=dict(recovered=accepted,truth_tokens=len(truth),correct_ids=correct,
        exact_match=exact,false_acceptances=accepted-correct,
        total_queries=sum(r["queries"] for r in rows),
        note="Separate evaluation reads private labels; no unique-solution claim")
    if result.get('method')=='first_v':
        covered=sum(i<len(truth) and truth[i] in np.load(args.run/r['passing_ids_file'],allow_pickle=False)
                    for i,r in enumerate(rows))
        evaluation.update(positions_evaluated=len(rows),truth_in_threshold_candidates=int(covered),
            nearest_correct=sum(i<len(truth) and r['nearest_token_id']==truth[i] for i,r in enumerate(rows)),
            ambiguous_positions=sum(r['stop_reason']=='ambiguous' for r in rows),
            unresolved_positions=sum(r['stop_reason']=='unresolved' for r in rows))
    dump(args.run/"evaluation.json",evaluation);print(json.dumps(evaluation,indent=2))


def decode(args):
    result=json.loads((args.run/"result.json").read_text())
    if any(x is None for x in result["recovered_token_ids"]):
        raise ValueError("Unresolved/ambiguous positions remain; inspect trace and candidate files before decoding")
    if args.tokenizer.suffix==".model":
        import sentencepiece as spm
        tokenizer=spm.SentencePieceProcessor(model_file=str(args.tokenizer))
        text=tokenizer.decode(result["recovered_token_ids"])
        special_mode="SentencePiece native decoding; control pieces may be omitted; token IDs remain authoritative"
    else:
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file(str(args.tokenizer))
        text=tokenizer.decode(result["recovered_token_ids"],skip_special_tokens=False)
        special_mode="keep special tokens"
    target=args.run/"recovered.txt"
    if target.exists():
        raise FileExistsError("Preserve the existing decoded artifact")
    target.write_text(text)
    dump(args.run/"decoding.json",dict(tokenizer_sha256=digest(args.tokenizer),special_token_handling=special_mode,
        note="Token IDs are authoritative; text decoding is a separate presentation step"))
    print("Decoded:",target)


def build_parser():
    p=argparse.ArgumentParser(description="KVidence NPU reconstruction framework")
    sub=p.add_subparsers(dest="command",required=True)
    common=argparse.ArgumentParser(add_help=False)
    selection=common.add_mutually_exclusive_group()
    selection.add_argument("--model-name",choices=tuple(MODEL_CONFIGS),help="Named model selection; default gemma4-e2b")
    selection.add_argument("--model-config",type=Path)
    common.add_argument("--manifest",type=Path)
    common.add_argument("--model",type=Path)
    common.add_argument("--device-model",help="Absolute model path in the Android app's accessible storage")
    common.add_argument("--tokenizer",type=Path)
    common.add_argument("--batch-signature")
    common.add_argument("--vocab-size",type=int,help="Optional expected model vocabulary size; checked against logits, never used to truncate IDs")
    device=argparse.ArgumentParser(add_help=False)
    device.add_argument("--adb");device.add_argument("--serial");device.add_argument("--no-install",action="store_true")
    device.add_argument("--transport",choices=("socket","file"),help="Native NPU RPC transport; default persistent socket; file preserves legacy transport")
    device.add_argument("--native-mode",choices=("reference","optimized","optimized_v2"),help="Native execution: default optimized_v2; optimized preserves the prior version; reference disables copy/context optimizations")
    q=sub.add_parser("prepare",parents=[common])
    q.add_argument("--snapshot",type=Path,required=True);q.add_argument("--prefix-json",type=Path,required=True)
    q.add_argument("--start",type=int,required=True);q.add_argument("--count",type=int,required=True)
    q.add_argument("--layer",type=int);q.add_argument("--output",type=Path,required=True)
    q.add_argument('--media-spans-json',type=Path,help='Known media intervals only; no source media or true features')
    q=sub.add_parser("check-adapter",parents=[common,device],help="Explicit finite backend contract validation")
    q.add_argument("--prefix-json",type=Path,required=True)
    q.add_argument("--candidate-ids-json",type=Path,required=True)
    q.add_argument("--layer",type=int);q.add_argument("--output",type=Path,required=True)
    q.add_argument("--atol",type=float,default=0.);q.add_argument("--rtol",type=float,default=0.)
    q=sub.add_parser("fixture",parents=[common,device])
    q.add_argument("--prefix-json",type=Path,required=True)
    group=q.add_mutually_exclusive_group();group.add_argument("--token-ids-json",type=Path);group.add_argument("--greedy-tokens",type=int,default=3)
    q.add_argument("--layer",type=int);q.add_argument("--output",type=Path,required=True)
    q=sub.add_parser("build-v-table",parents=[common,device],help="Precompute native first-layer V dictionary")
    q.add_argument("--vocabulary",type=Path)
    q.add_argument("--output",type=Path,required=True)
    q=sub.add_parser('calibrate-cpa',parents=[common,device],help='Known-plaintext Collision+ calibration; full vocabulary unless explicitly pilot')
    q.add_argument('--observation',type=Path,required=True);q.add_argument('--known-tokens-json',type=Path,required=True)
    q.add_argument('--output',type=Path,required=True);q.add_argument('--rank',type=int,default=8)
    q.add_argument('--chunk-size',type=int,default=4096);q.add_argument('--pilot-candidates',type=int)
    q.add_argument('--seed',type=int,default=20261019)
    recovery=argparse.ArgumentParser(add_help=False)
    q=recovery
    q.add_argument("--observation",type=Path,required=True);q.add_argument("--output",type=Path,required=True)
    q.add_argument("--embeddings",type=Path)
    q.add_argument("--vocabulary",type=Path,help="Optional JSON array or {token_ids:[...]} of original IDs; default full vocabulary")
    q.add_argument("--initial-candidates",type=int,default=200,help="Initial guided seed total, not a query budget (default 200)")
    q.add_argument("--prediction-candidates",type=int,default=50,help="Seeds selected by model probability; remainder sampled randomly (default 50)")
    q.add_argument("--epsilon",type=float,default=.125,help='Strict distance threshold; default RMSE 0.125 equals L2 4 for 1024 coordinates. CPA ignores this')
    q.add_argument("--distance-metric",choices=("rmse","l2"),default="rmse",help="Dimension-normalised RMSE (default), or legacy L2; official CPA retains calibrated K/V L2")
    q.add_argument("--distance-space",choices=('auto','lsb','original'),default='auto',help='Default: LSB codes for integer KV, original units for floating KV. CPA always retains original units. Old raw thresholds are not equivalent.')
    q.add_argument("--batch-size",type=int,help='Default 256 for CPA, 128 for legacy/guided')
    q.add_argument("--proposals",type=int,default=20);q.add_argument("--seed",type=int,default=20261019)
    q.add_argument("--live-events",action="store_true",help="Write per-batch events for the Ghost Attack dashboard")
    q.add_argument("--max-queries",type=int);q.add_argument("--max-positions",type=int)
    q.add_argument('--calibration',type=Path);q.add_argument('--allow-pilot-calibration',action='store_true')
    q.add_argument('--stop-partition',type=int,default=1)
    q=sub.add_parser("run",parents=[common,device,recovery])
    q.add_argument("--method",choices=("auto","first_v","guided","collision_plus","collision_plus_cpa","exhaustive"),default="auto")
    q.add_argument("--v-table",type=Path,help="Existing matching first V table; omitted: cache/build automatically")
    q=sub.add_parser("compare",parents=[common,device,recovery],help="Paired guided and official-rule Collision+ CPA NPU baseline")
    q.add_argument('--baseline',choices=('collision_plus_cpa','collision_plus'),default='collision_plus_cpa',
                   help='Default CPA requires independent calibration; collision_plus selects the optional shared-L2 adaptation')
    q.add_argument("--repeats",type=int,default=1)
    q.add_argument("--first",choices=("collision_plus","guided"),default="collision_plus")
    q=sub.add_parser("evaluate-comparison",help="Independent truth evaluation after both recovery methods finish")
    q.add_argument("--comparison",type=Path,required=True);q.add_argument("--truth",type=Path,required=True)
    q=sub.add_parser("summarize-comparison")
    q.add_argument("--comparison",type=Path,required=True)
    q=sub.add_parser("evaluate")
    q.add_argument("--run",type=Path,required=True);q.add_argument("--truth",type=Path,required=True)
    q=sub.add_parser("decode")
    q.add_argument("--run",type=Path,required=True);q.add_argument("--tokenizer",type=Path)
    selection=q.add_mutually_exclusive_group()
    selection.add_argument("--model-name",choices=tuple(MODEL_CONFIGS))
    selection.add_argument("--model-config",type=Path)
    sub.add_parser("models",help="List registered models and local asset availability")
    return p


def resolve_search_defaults(args):
    if hasattr(args,'batch_size') and args.batch_size is None:
        args.batch_size=256 if getattr(args,'method',None)=='collision_plus_cpa' or getattr(args,'baseline',None)=='collision_plus_cpa' else 128


def main():
    args=build_parser().parse_args()
    resolve_search_defaults(args)
    if args.command=="evaluate-comparison":
        from .comparison import evaluate_comparison
        return evaluate_comparison(args)
    if args.command=="summarize-comparison":
        from .comparison import summarize
        return summarize(args.comparison)
    if args.command=="models":
        rows=[]
        for name,path in MODEL_CONFIGS.items():
            profile=load_model_config(path)
            missing=[key for key in ("model","manifest","tokenizer","embeddings") if not profile[key].exists()]
            rows.append(dict(id=name,name=profile["name"],default=name=="gemma4-e2b",config=str(path),
                local_assets_ready=not missing,missing=missing,default_layer_1based=profile["default_layer"]+1))
        print(json.dumps(rows,indent=2));return
    explicit_selection=getattr(args,"model_config",None) is not None or getattr(args,"model_name",None) is not None
    args.model_config=getattr(args,"model_config",None) or MODEL_CONFIGS.get(getattr(args,"model_name",None),DEFAULT_MODEL_CONFIG)
    if args.command in ("evaluate","decode"):
        if args.command=="decode" and args.tokenizer is None:
            protocol=args.run/"protocol.json"
            saved=json.loads(protocol.read_text()).get("model_config",{}) if protocol.exists() and not explicit_selection else {}
            args.tokenizer=Path(saved["tokenizer"]) if saved.get("tokenizer") else load_model_config(args.model_config)["tokenizer"]
        return {"evaluate":evaluate,"decode":decode}[args.command](args)
    profile=load_model_config(args.model_config)
    for key in ("model","device_model","manifest","tokenizer","embeddings","vocabulary","batch_signature"):
        override=getattr(args,key,None)
        if override is not None:
            profile[key]=override.resolve() if isinstance(override,Path) else override
        setattr(args,key,profile[key])
    args.mask_token_id=profile["mask_token_id"]
    if hasattr(args,"layer") and args.layer is None:
        args.layer=profile["default_layer"]
    args.resolved_model=profile
    args.adapter=get_adapter(profile["adapter"])
    sections=args.adapter.read_manifest(args.manifest)
    dims=args.adapter.dimensions(sections,args.resolved_model)
    if args.vocab_size is not None and args.vocab_size!=dims["vocab_size"]:
        raise ValueError("vocab-size does not match the selected model logits; use --vocabulary to select original IDs")
    model_hash=digest(args.model)
    if args.command=="compare":
        from .comparison import compare
        return compare(args,sections,model_hash)
    return {"prepare":prepare,"fixture":fixture,"run":run,"check-adapter":check_adapter,"build-v-table":build_v_table,"calibrate-cpa":calibrate_cpa}[args.command](args,sections,model_hash)
