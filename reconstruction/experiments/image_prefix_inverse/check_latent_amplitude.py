"""Check whether scalar rescaling escapes a finite-precision plateau."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from kvrecon.common import EXPERIMENTS,dump,digest
from kvrecon.npu import NpuBackend


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    with np.load(args.warm) as f:u=f['latent'].copy();expected=f['kv'].copy()
    factors=np.concatenate([np.geomspace(.125,8,49),1+2.**(-np.arange(8,24)),1-2.**(-np.arange(8,24))]).astype(np.float32)
    factors=np.unique(factors)
    inputs=u[None]*factors[:,None]
    data=HERE/'data/20261022_pilot_prefill';p=1976
    protocol=json.loads((data/'protocol.json').read_text())
    row=next(x for x in json.loads((data/'index.json').read_text()) if x['split']=='test')
    with np.load(data/row['file']) as f:prefix=f['embeddings'][:p].copy();target=f['kv_codes'][p].astype(float)
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    dump(args.output/'protocol.json',dict(warm_sha256=digest(args.warm),source_sha256=digest(Path(__file__)),
        position=p,factors=factors.tolist(),current_truth_used=False,prefix='true canonical prefix'))
    with NpuBackend(sections,4,args.output/'device',protocol['model_sha256'],serial='YOUR_DEVICE_SERIAL') as b:
        b.reset(protocol['initial_prefix_ids'])
        for image in row['images']:
            start=image['start'];end=min(image['start']+image['rows'],p)
            for a in range(start,end,128):
                chunk=np.asarray(prefix[a:min(a+128,end)],dtype='<f4')
                b.rpc('collect_embeddings',rows=len(chunk),position=a,features_hex=chunk.tobytes().hex())
            if b.position==p:break
            b.commit_embeddings(prefix[end:end+1]);b.commit(255999)
        before=b.query_vision(u[None])[0]
        np.testing.assert_array_equal(before,expected)
        codes=b.query_vision(inputs).astype(float)
        dist=np.sqrt(np.mean((codes-target)**2,axis=1));j=int(dist.argmin())
        repeated=b.query_vision(inputs[j:j+1])[0]
        np.testing.assert_array_equal(repeated,codes[j])
        np.savez_compressed(args.output/'best.npz',latent=inputs[j],kv=codes[j])
        np.savez_compressed(args.output/'candidates.npz',latent=inputs,kv=codes,factors=factors)
        initial=float(np.sqrt(np.mean((before.astype(float)-target)**2)))
        result=dict(initial_rmse=initial,best_rmse=float(dist[j]),improved=bool(dist[j]<initial),
            best_factor=float(factors[j]),candidates=len(factors),single_replay_exact=True,
            rows=[dict(factor=float(f),rmse=float(d)) for f,d in zip(factors,dist)])
        dump(args.output/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
