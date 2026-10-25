"""Phone adapter + NPU only; synthetic gate before the real first image slot."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import sys
import numpy as np
from spsa_adam import Config,fit,unit
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from kvrecon.common import dump,digest,EXPERIMENTS
from kvrecon.npu import NpuBackend
from kvrecon.cli import private_guard


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=HERE/'spsa_adam_device')
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--lr',type=float,default=.05)
    parser.add_argument('--perturbation',type=float,default=.1)
    args=parser.parse_args()
    private_guard()
    meta=json.loads((HERE/'input/metadata.json').read_text())
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    cfg=Config(steps=args.steps,lr=args.lr,perturbation=args.perturbation)
    out=args.output;out.mkdir(exist_ok=False)
    dump(out/'protocol.json',dict(config=asdict(cfg),seeds=[17,29],synthetic_target_seed=999,
        target_layer_1based=5,position=5,public_prefix=meta['public_prefix'],
        adapter='phone model vision_adapter_280 / CPU',decoder='phone model prefill_128 and decode / NPU',
        original_media_accessed=False,original_text_accessed=False,desktop_model_loaded=False,
        acceptance='per-position raw concatenated K+V L2 < 4, verify single-position commit',
        plan='two independent initializations for one reachable synthetic target; real first image slot only after both pass',
        synthetic_truth_supplied_to_optimizer=False,
        source_sha256={p.name:digest(p) for p in (Path(__file__),HERE/'spsa_adam.py')},
        model_sha256=meta['model_sha256']))
    results=[]
    with NpuBackend(sections,4,out/'device',meta['model_sha256']) as b:
        b.reset(meta['public_prefix'])
        secret=unit(np.random.default_rng(999).standard_normal((1,768)))
        target=b.values(b.query_vision(secret))[0];del secret
        np.save(out/'synthetic_target_kv.npy',target)
        def run(name,target,config):
            b.reset(meta['public_prefix'])
            u,result=fit(lambda rows:b.values(b.query_vision(rows)),target,config,out/name)
            assert b.position==5
            features=b.project_vision(u[None]);committed=b.values(b.commit_embeddings(features))[0]
            result.update(commit_l2=float(np.linalg.norm(committed-target)),
                passes_commit=bool(np.linalg.norm(committed-target)<4),kind=name)
            np.save(out/name/'features.npy',features);dump(out/name/'result.json',result)
            results.append(result);dump(out/'runs.json',results)
            print(f"{name}: query L2={result['best_l2']}, commit L2={result['commit_l2']}",flush=True)
            return result['passes_query'] and result['passes_commit']
        gates=[run(f'synthetic_seed_{seed}',target,replace(cfg,seed=seed)) for seed in (17,29)]
        real_attempted=all(gates)
        if real_attempted:
            with np.load(HERE/'input/observed.npz') as d:real=d['media_kv'][0].copy()
            run('real_image_position_5',real,cfg)
        dump(out/'result.json',dict(synthetic_passes=sum(gates),synthetic_runs=len(gates),
            real_first_position_attempted=real_attempted,whole_image_recovery_attempted=False,
            subsequent_text_recovery_attempted=False,runs=results,
            limitation='two starts on one synthetic target; not general recovery accuracy'))

if __name__=='__main__':main()
