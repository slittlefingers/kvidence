"""Sequential phone-only post-adapter pilot; never jointly optimize positions."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import numpy as np
from spsa_adam import Config,unit
from direct_spsa import fit_direct

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from kvrecon.common import dump,digest,EXPERIMENTS
from kvrecon.npu import NpuBackend
from kvrecon.cli import private_guard


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--positions',type=int,default=5)
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--output',type=Path,default=HERE/'direct_spsa_l2_5')
    args=parser.parse_args()
    if not 1<=args.positions<=260:raise ValueError('Invalid position count')
    private_guard()
    meta=json.loads((HERE/'input/metadata.json').read_text())
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    with np.load(HERE/'input/observed.npz') as d:targets=d['media_kv'].astype(np.float64)
    cfg=Config(dimension=1536,steps=args.steps,seed=29,acceptance_l2=5.)
    out=args.output;out.mkdir(exist_ok=False)
    dump(out/'protocol.json',dict(config=asdict(cfg),planned_positions=args.positions,
        target_layer_1based=5,sample='0001',mode='strictly sequential single-position optimization',
        batch_meaning='128 alternative vectors at ONE position, not 128 causal positions',
        initialization='same 128 random pre-adapter vectors as baseline seed 29, projected once by phone adapter',
        iteration='free 1536-dimensional post-adapter vectors; no adapter calls or sphere projection',
        parameter_units='z = scalar_RMS(initial_features) * x; scalar independent of target',
        acceptance='each position raw K+V L2 < 5 in query and commit; stop at first failure',
        comparison='historical real_spsa_l2_5; first position has identical prefix and candidate pool; later prefixes differ',
        original_media_accessed=False,original_text_accessed=False,target_cache_used_as_prefix=False,
        whole_image_content_evaluated=False,model_sha256=meta['model_sha256'],
        source_sha256={p.name:digest(p) for p in (Path(__file__),HERE/'direct_spsa.py',HERE/'spsa_adam.py')}))
    runs=[];accepted=[]
    with NpuBackend(sections,4,out/'device',meta['model_sha256']) as b:
        b.reset(meta['public_prefix'])
        latent=unit(np.random.default_rng(29).standard_normal((128,768)))
        tick=time.perf_counter();initial=b.project_vision(latent);initialization_seconds=time.perf_counter()-tick
        np.save(out/'initial_features.npy',initial)
        # Interface controls before the real optimization, no private latent.
        via_adapter=b.query_vision(latent[:4]);direct=b.query_embeddings(initial[:4])
        permutation=np.array([2,0,3,1]);permuted=b.query_embeddings(initial[permutation])
        controls=dict(adapter_vs_direct_exact=bool(np.array_equal(via_adapter,direct)),
            candidate_order_exact=bool(np.array_equal(permuted,direct[permutation])),
            nonadvancing=b.position==5)
        committed=b.commit_embeddings(initial[:1])
        controls['query_vs_commit_exact']=bool(np.array_equal(direct[:1],committed))
        dump(out/'controls.json',controls)
        if not all(controls.values()):raise RuntimeError('Direct candidate interface control failed')
        b.reset(meta['public_prefix'])
        for i in range(args.positions):
            position=meta['media_start']+i;assert b.position==position
            folder=out/f'position_{position:04d}'
            z,result=fit_direct(lambda rows:b.values(b.query_embeddings(rows)),targets[i],initial,cfg,folder)
            assert b.position==position
            kv=b.values(b.commit_embeddings(z[None]))[0]
            distance=float(np.linalg.norm(kv.astype(np.float64)-targets[i]))
            result.update(position=position,commit_l2=distance,passes_commit=distance<5)
            np.save(folder/'commit_kv.npy',kv);dump(folder/'result.json',result)
            runs.append(result);dump(out/'runs.json',runs)
            print(f'Direct position={position} query={result["best_l2"]:.6f} commit={distance:.6f}',flush=True)
            if not(result['passes_query'] and result['passes_commit']):break
            accepted.append(z);np.save(out/'accepted_features.npy',np.array(accepted))
        dump(out/'result.json',dict(status='pilot_complete' if len(accepted)==args.positions else 'stopped_at_failed_position',
            accepted_positions=len(accepted),attempted_positions=len(runs),planned_positions=args.positions,
            entire_image_positions=260,initialization_seconds=initialization_seconds,
            device_setup_seconds=b.setup_seconds,whole_image_content_evaluated=False,runs=runs))


if __name__=='__main__':main()
