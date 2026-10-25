"""Isolate exact-input batch composition and tiny projection perturbations."""
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from kvrecon.common import EXPERIMENTS,dump
from kvrecon.npu import NpuBackend


def main():
    root=HERE/'results/20261022_inverse_npu'
    data=HERE/'data/20261022_pilot_prefill'
    protocol=json.loads((data/'protocol.json').read_text())
    row=next(x for x in json.loads((data/'index.json').read_text()) if x['split']=='test')
    with np.load(data/row['file']) as f:z=f['embeddings'].copy();target=f['kv_codes'].copy()
    p=1976
    with np.load(root/f'point_{p}.npz') as f:
        predictions=f['predicted_embeddings'].copy();projected=f['projected_embeddings'].copy();u=f['projected_latents'].copy()
    image=next(x for x in row['images'] if x['start']<=p<x['start']+x['rows'])
    with np.load(data/'images'/f"{image['image_id']}.npz") as f:original_u=f['pre_adapter'][p-image['start']].copy()
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    baseline=np.load(root/f'search_{p}_learned.npz')['latent'].copy()
    results=[]
    with NpuBackend(sections,4,root/'audit_device',protocol['model_sha256'],serial='YOUR_DEVICE_SERIAL') as b:
        b.reset(protocol['initial_prefix_ids'])
        for item in row['images']:
            start=item['start'];end=min(item['start']+item['rows'],p)
            for a in range(start,end,128):
                v=np.asarray(z[a:min(a+128,end)],dtype='<f4')
                b.rpc('collect_embeddings',rows=len(v),position=a,features_hex=v.tobytes().hex())
            if b.position==p:break
            b.commit_embeddings(z[end:end+1]);b.commit(255999)
        assert b.position==p
        original_z=b.project_vision(original_u[None])[0]
        dump(root/'projection_audit.json',dict(original_adapter_exact=bool(np.array_equal(original_z,z[p])),
            original_max_abs=float(np.max(np.abs(original_z-z[p]))),
            projected_max_abs=float(np.max(np.abs(projected[0]-z[p])))))
        variants={
            'truth_single':z[p:p+1],
            'truth_repeated3':np.repeat(z[p:p+1],3,axis=0),
            'truth_plus_direct_predictions':np.concatenate([z[p:p+1],predictions]),
            'truth_plus_projected_predictions':np.concatenate([z[p:p+1],projected[1:]]),
            'projected_truth_single':projected[:1],
            'projected_truth_plus_projected_predictions':projected,
            'original_adapter_truth_single':original_z[None],
        }
        for repeat in range(2):
            for name,values in variants.items():
                got=b.query_embeddings(values)[0];delta=got.astype(float)-target[p]
                results.append(dict(name=name,repeat=repeat,rmse_lsb=float(np.sqrt(np.mean(delta**2))),
                    first_embedding_exact=bool(np.array_equal(values[0],z[p]))))
            for name,values in [('original_u_single',original_u[None]),('projected_u_single',u[:1]),('learned_best_single',baseline[None])]:
                got=b.query_vision(values)[0];delta=got.astype(float)-target[p]
                results.append(dict(name=name,repeat=repeat,rmse_lsb=float(np.sqrt(np.mean(delta**2)))))
        dump(root/'batch_projection_audit.json',results)
        print(json.dumps(results,indent=2))


if __name__=='__main__':main()
