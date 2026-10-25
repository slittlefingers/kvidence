"""Replay the selected input unchanged; separate numerical drift from optimization."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from kvrecon.common import EXPERIMENTS,dump,code_values
from kvrecon.npu import NpuBackend
from kvrecon.cli import private_guard
from kvrecon.search import kv_distance


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--observation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();private_guard()
    meta=json.loads((args.observation/'metadata.json').read_text())
    with np.load(args.observation/'observed.npz') as data:target=code_values(data['codes'][0])
    folder=args.run/'recovery/position_0005'
    z=np.load(folder/'best_input.npy');reference=np.load(folder/'best_kv.npy')
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    with NpuBackend(sections,meta['layer_index'],args.output/'device',meta['model_sha256'],install=False,serial='YOUR_DEVICE_SERIAL') as b:
        b.reset(meta['public_prefix'])
        for size in (1,32,128,1):
            values=code_values(b.query_vision(np.repeat(z[None],size,axis=0)))
            if b.position!=meta['media_start']:raise ValueError('Query advanced prefix')
            np.save(args.output/f'query_{len(rows)}.npy',values)
            rows.append(dict(batch_size=size,rmse_to_target=kv_distance(values,target).tolist(),
                rmse_to_recorded_best=kv_distance(values,reference).tolist(),
                all_rows_equal=bool(np.all(values==values[:1])),
                exact_recorded_best=bool(np.all(values==reference))))
        features=b.project_vision(z[None]);actual=code_values(b.commit_embeddings(features))[0]
        np.save(args.output/'commit.npy',actual)
        report=dict(query_replays=rows,commit_rmse_to_target=float(kv_distance(actual,target)),
            commit_rmse_to_recorded_best=float(kv_distance(actual,reference)),
            commit_exact_recorded_best=bool(np.array_equal(actual,reference)),
            input_unchanged=True,source_run=str(args.run))
        dump(args.output/'result.json',report)
        print(json.dumps(dict(commit_rmse_to_target=report['commit_rmse_to_target'],
            all_query_replays_exact=all(r['exact_recorded_best'] for r in rows),
            commit_exact_recorded_best=report['commit_exact_recorded_best'])))

if __name__=='__main__':main()
