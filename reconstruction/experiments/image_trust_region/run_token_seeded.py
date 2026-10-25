"""Direct first-block input search: deployed vocabulary seeds, media PLE, no adapter."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from kvrecon.common import EXPERIMENTS,code_values,dump,digest,validate_quantization_grid
from kvrecon.npu import NpuBackend
from kvrecon.cli import private_guard
from run_multiscale import proposals,SCALES

def main():
    p=argparse.ArgumentParser();p.add_argument('--observation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed-candidates',type=int,default=1024);p.add_argument('--seed',type=int,default=217)
    p.add_argument('--seconds',type=float,default=120);p.add_argument('--rounds',type=int,default=300);p.add_argument('--epsilon',type=float,default=.1)
    args=p.parse_args();private_guard()
    meta=json.loads((args.observation/'metadata.json').read_text())
    if digest(args.observation/'observed.npz')!=meta['observation_sha256']:raise ValueError('Observation mismatch')
    with np.load(args.observation/'observed.npz') as f:target=code_values(f['codes'][0])
    args.output.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(args.seed)
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    best=float('inf');best_x=None;best_q=None;queries=0;trace=[]
    with NpuBackend(sections,meta['layer_index'],args.output/'device',meta['model_sha256'],install=False,serial='YOUR_DEVICE_SERIAL') as b:
        validate_quantization_grid(meta['quantization'],b.quant);b.reset(meta['public_prefix'])
        ids=rng.choice(b.vocab_size,size=args.seed_candidates,replace=False)
        dump(args.output/'protocol.json',dict(optimizer='token_seeded_direct_embedding_multiscale',observation=meta,
            seed=args.seed,seed_ids=ids.tolist(),seed_domain='uniform subset of full original vocabulary, no language filter',
            embedding_source='deployed native embedding signature via RPC, including runtime scaling',
            search_space='1536 float32 input coordinates of first Transformer block, after adapter',
            target_layer_1based=meta['layer_index']+1,adapter_called=False,media_ple_id=0,
            epsilon=args.epsilon,distance_space='lsb',distance_metric='rmse',seconds=args.seconds,rounds=args.rounds,
            source_sha256=digest(Path(__file__)),proposal_source_sha256=digest(Path(__file__).with_name('run_multiscale.py')),
            original_media_accessed=False,original_text_accessed=False,global_nearest_token_claim=False))
        started=time.perf_counter()
        embeddings=[]
        for n,tid in enumerate(ids):
            row=np.frombuffer(b.rpc('embedding',id=int(tid)),dtype=np.float32).copy()
            if row.shape!=(1536,) or not np.isfinite(row).all():raise ValueError('Invalid deployed embedding')
            embeddings.append(row)
            if (n+1)%256==0:print(f'Deployed embeddings {n+1}/{len(ids)}',flush=True)
        embeddings=np.asarray(embeddings);np.savez_compressed(args.output/'vocabulary_seeds.npz',ids=ids,embeddings=embeddings)
        extraction_seconds=time.perf_counter()-started
        def measure(rows,label):
            nonlocal best,best_x,best_q,queries
            rows=np.asarray(rows,dtype=np.float32);q=code_values(b.query_embeddings(rows))
            if b.position!=meta['media_start'] or q.shape!=(len(rows),len(target)) or not np.isfinite(q).all():raise ValueError('Invalid replay')
            d=np.sqrt(np.mean((q-target)**2,axis=1));j=int(d.argmin());queries+=len(rows)
            if d[j]<best:
                best=float(d[j]);best_x=rows[j].copy();best_q=q[j].copy()
                np.save(args.output/'best_input.npy',best_x);np.save(args.output/'best_kv.npy',best_q)
            np.savez_compressed(args.output/f'{label}.npz',inputs=rows,kv=q)
            return q,d
        all_q=[];all_d=[]
        for start in range(0,len(ids),128):
            q,d=measure(embeddings[start:start+128],f'seed_{start:05d}');all_q.extend(q);all_d.extend(d)
        all_q=np.array(all_q);all_d=np.array(all_d);order=np.argsort(all_d);top=order[:4]
        dump(args.output/'seed_results.json',dict(best_id=int(ids[order[0]]),best_rmse=float(all_d[order[0]]),
            top_ids=ids[top].tolist(),top_distances=all_d[top].tolist(),candidates=len(ids),
            extraction_seconds=extraction_seconds,seed_seconds=time.perf_counter()-started))
        # Normalize optimization coordinates by one fixed, target-independent scale.
        unit=max(float(np.sqrt(np.mean(embeddings.astype(float)**2))),1e-8)
        centers=embeddings[top].copy();states=all_q[top].copy();distances=all_d[top].copy()
        seed_best=best;refine_start=time.perf_counter()
        for step in range(1,args.rounds+1):
            if best<=args.epsilon or time.perf_counter()-refine_start>=args.seconds:break
            branch=(step-1)%len(top);u=centers[branch]/unit;uq=states[branch]
            delta=(2*rng.integers(0,2,(64,1536))-1).astype(np.float32);scales=np.repeat(SCALES,16)[:,None]
            probes=np.concatenate([u+scales*delta,u-scales*delta]).astype(np.float32)
            pq,pd=measure(probes*unit,f'round_{step:04d}_probes')
            trial=proposals(u,uq,delta,pq,target)
            # Four independent seed branches; no branch receives a different branch's incumbent.
            trial=np.concatenate([trial,u[None]]).astype(np.float32)
            tq,td=measure(trial*unit,f'round_{step:04d}_trials')
            j=int(td.argmin());k=int(pd.argmin())
            if pd[k]<td[j]:centers[branch]=probes[k]*unit;states[branch]=pq[k];distances[branch]=pd[k]
            else:centers[branch]=trial[j]*unit;states[branch]=tq[j];distances[branch]=td[j]
            trace.append(dict(round=step,branch=branch,queries=queries,elapsed_seconds=time.perf_counter()-started,
                refinement_seconds=time.perf_counter()-refine_start,best_rmse=best,branch_rmse=distances.tolist()))
            dump(args.output/'optimization.json',trace)
            if step%20==0:print(f'Direct round={step} best_RMSE={best:.6f} sec={trace[-1]["refinement_seconds"]:.1f}',flush=True)
        elapsed=time.perf_counter()-started
        repeats=code_values(b.query_embeddings(np.repeat(best_x[None],32,axis=0)))
        report=dict(status='threshold' if best<=args.epsilon else 'budget',seed_best_rmse=seed_best,best_rmse=best,
            queries=queries,rounds=len(trace),optimizer_seconds=elapsed,embedding_extraction_seconds=extraction_seconds,
            refinement_seconds=time.perf_counter()-refine_start,coordinate_scale=unit,epsilon=args.epsilon,
            passes_query=best<=args.epsilon,repeat_exact=bool(np.all(repeats==best_q)),diagnostic_candidates=32,
            adapter_called=False,whole_image_recovered=False,subsequent_text_recovered=False)
        if best<=args.epsilon:
            actual=code_values(b.commit_embeddings(best_x[None]))[0];error=float(np.sqrt(np.mean((actual-target)**2)))
            report.update(commit_rmse=error,passes_commit=error<=args.epsilon)
        dump(args.output/'result.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
