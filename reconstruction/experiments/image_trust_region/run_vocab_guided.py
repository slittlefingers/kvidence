"""Three-round text-style vocabulary search at a media position; no adapter/refinement."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from kvrecon.common import EXPERIMENTS,DEFAULT_EMBEDDINGS,code_values,dump,digest,validate_quantization_grid
from kvrecon.npu import NpuBackend
from kvrecon.cli import private_guard
from kvrecon.surrogate import ExistingRidge
from kvrecon.search import lowest_scored_ids

def main():
    p=argparse.ArgumentParser();p.add_argument('--observation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--seed',type=int,default=217)
    a=p.parse_args();private_guard()
    if not 1<=a.rounds<=3:raise ValueError('Pilot allows 1..3 guided rounds')
    meta=json.loads((a.observation/'metadata.json').read_text())
    if digest(a.observation/'observed.npz')!=meta['observation_sha256']:raise ValueError('Observation hash mismatch')
    with np.load(a.observation/'observed.npz') as f:target=code_values(f['codes'][0])
    a.output.mkdir(parents=True,exist_ok=False)
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    surrogate=ExistingRidge(DEFAULT_EMBEDDINGS,262144,1536)
    ids=[];measurements=[];distances=[];trace=[];rng=np.random.default_rng(a.seed)
    with NpuBackend(sections,meta['layer_index'],a.output/'device',meta['model_sha256'],install=False,serial='YOUR_DEVICE_SERIAL') as b:
        validate_quantization_grid(meta['quantization'],b.quant);b.reset(meta['public_prefix'])
        dump(a.output/'protocol.json',dict(observation=meta,seed=a.seed,probability_candidates=50,random_candidates=150,proposals=20,rounds=a.rounds,
            method='existing_text_ridge_at_media_position',surrogate=surrogate.metadata,source_sha256=digest(Path(__file__)),
            actual_embedding_source='deployed runtime embedding RPC',media_ple_id=0,adapter_called=False,
            distance_metric='LSB code-space RMSE',epsilon=.1,original_media_accessed=False))
        started=time.perf_counter();order=b.probabilities();top=order[:50]
        random=rng.choice(np.setdiff1d(np.arange(b.vocab_size),top),150,replace=False)
        def measure(chosen,label,ranking_seconds=0.):
            tick=time.perf_counter();vectors=np.array([np.frombuffer(b.rpc('embedding',id=int(t)),dtype=np.float32).copy() for t in chosen])
            q=np.concatenate([code_values(b.query_embeddings(vectors[s:s+128])) for s in range(0,len(chosen),128)])
            if b.position!=meta['media_start'] or q.shape!=(len(chosen),len(target)):raise ValueError('Replay mismatch')
            d=np.sqrt(np.mean((q-target)**2,axis=1));ids.extend(map(int,chosen));measurements.extend(q);distances.extend(d)
            np.savez_compressed(a.output/f'{label}.npz',ids=chosen,embeddings=vectors,kv=q,rmse=d)
            best=int(np.argmin(distances));record=dict(stage=label,queries=len(ids),best_id=ids[best],best_rmse=float(distances[best]),
                stage_best_rmse=float(d.min()),ranking_seconds=ranking_seconds,measurement_seconds=time.perf_counter()-tick,elapsed_seconds=time.perf_counter()-started)
            trace.append(record);dump(a.output/'trace.json',trace);print(json.dumps(record),flush=True)
        measure(top,'probability');measure(random,'random')
        for stage in range(1,a.rounds+1):
            if min(distances)<=.1:break
            tick=time.perf_counter();scores=surrogate.scores(ids,np.asarray(measurements),target,np.asarray(distances))
            available=np.setdiff1d(np.arange(b.vocab_size),ids)
            chosen=lowest_scored_ids(available,scores,20)
            measure(chosen,f'guided_{stage}',time.perf_counter()-tick)
        dump(a.output/'result.json',dict(trace=trace,best_rmse=float(min(distances)),queries=len(ids),passes=bool(min(distances)<=.1),
            whole_image_recovered=False,continuous_refinement_performed=False))

if __name__=='__main__':main()
