"""Validate the phone adapter and attempt the first image slot on its real graph.

This is a separate finite-difference feasibility test, not the MLX algorithm.
Only target layer-5 KV and public model/template enter it.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from kvrecon.common import dump,EXPERIMENTS,digest
from kvrecon.npu import NpuBackend
from kvrecon.cli import private_guard

def unit(x):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-8)

def fit_slot(b,target,rng,out,steps=30):
    queries=0;trace=[];tick=time.perf_counter()
    def score(u):
        nonlocal queries
        values=[]
        for start in range(0,len(u),128):
            part=u[start:start+128]
            values.append(b.values(b.query_vision(part)));queries+=len(part)
        return np.concatenate(values)
    u0=unit(rng.standard_normal((128,768)).astype(np.float32))
    v0=score(u0);d0=np.linalg.norm(v0-target,axis=-1);index=int(d0.argmin())
    u=u0[index].copy();v=v0[index];best=float(d0[index]);damping=.1;eps=.5
    for step in range(steps+1):
        trace.append(dict(step=step,l2=best,queries=queries,seconds=time.perf_counter()-tick,
            finite_difference_step=eps,damping=damping))
        dump(out/'trace.json',trace)
        np.save(out/'best_latent.npy',u)
        print(f'device adapter position={b.position} step={step} L2={best:.6f} candidates={queries}',flush=True)
        if best<4 or step==steps:break
        perturb=np.broadcast_to(u,(768,768)).copy()+np.eye(768,dtype=np.float32)*eps
        jac=(score(perturb)-v)/eps
        residual=v-target
        h=jac@jac.T
        ridge=max(float(np.trace(h))/768,1e-6)*damping
        delta=np.linalg.solve(h+np.eye(768)*ridge,-jac@residual)
        # A trust radius in latent RMS; measure all accepted updates on NPU.
        rms=float(np.sqrt(np.mean(delta**2)))
        delta*=min(1.,.5/max(rms,1e-12))
        proposals=unit(u[None]+np.array([1.,.5,.25,.125,.0625])[:,None]*delta)
        values=score(proposals);dist=np.linalg.norm(values-target,axis=-1);i=int(dist.argmin())
        if dist[i]<best:
            u=proposals[i].astype(np.float32);v=values[i];best=float(dist[i]);damping=max(.001,damping*.7)
        else:
            damping=min(100.,damping*4);eps=min(2.,eps*1.3)
    result=dict(query_l2=best,passes_query=best<4,queries=queries,seconds=time.perf_counter()-tick,
        iterations=step,position=b.position,original_media_accessed=False,original_text_accessed=False)
    np.save(out/'best_kv.npy',v);dump(out/'result.json',result)
    return u,result

def main():
    private_guard()
    meta=json.loads((HERE/'input/metadata.json').read_text())
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    with np.load(HERE/'input/observed.npz') as d:target=d['media_kv'][0].copy()
    out=HERE/'device_adapter';out.mkdir(exist_ok=False)
    dump(out/'protocol.json',dict(adapter='phone tf_lite_vision_adapter/vision_adapter_280, CPU',
        decoder='phone tf_lite_prefill_decode/prefill_128 and decode, NPU',
        model_sha256=meta['model_sha256'],target_sha256=digest(HERE/'input/observed.npz'),
        method='separate native finite-difference damped Gauss-Newton feasibility probe; existing algorithm unchanged',
        scope='first image position, public prefix only; expand only after validating this path',
        epsilon=4.,seed=17,steps=30,finite_difference_step=.5,
        original_image_accessed=False,original_text_accessed=False))
    rng=np.random.default_rng(17)
    with NpuBackend(sections,4,out/'device',meta['model_sha256']) as b:
        b.reset(meta['public_prefix'])
        u=unit(rng.standard_normal((4,768)).astype(np.float32))
        z=b.project_vision(u);single=b.project_vision(u[:1])
        np.testing.assert_allclose(z[:1],single,atol=1e-6,rtol=1e-6)
        q=b.query_vision(u);permuted=b.query_vision(u[::-1])
        np.testing.assert_array_equal(q,permuted[::-1]);assert b.position==5
        committed=b.commit_embeddings(z[:1])
        control_l2=float(np.linalg.norm(b.values(q[:1])-b.values(committed)))
        assert control_l2<4
        np.save(out/'control_latent.npy',u);np.save(out/'control_features.npy',z)
        # Public embedding comparison diagnoses coordinates without using media.
        native=np.stack([np.frombuffer(b.rpc('embedding',id=t),dtype='<f4') for t in meta['public_prefix']])
        mlx=np.load(HERE/'public_prefix_comparison/mlx_embeddings.npy')[0]
        cos=np.sum(native*mlx,axis=-1)/(np.linalg.norm(native,axis=-1)*np.linalg.norm(mlx,axis=-1))
        controls=dict(adapter_batch_independence=True,candidate_permutation_exact=True,
            query_does_not_advance=True,query_vs_commit_l2=control_l2,
            public_embedding_cosines=cos.tolist(),
            public_embedding_l2=np.linalg.norm(native-mlx,axis=-1).tolist())
        dump(out/'controls.json',controls)
        print('Phone adapter/query controls:',json.dumps(controls),flush=True)
        b.reset(meta['public_prefix'])
        slot=out/'position_5';slot.mkdir()
        fitted,result=fit_slot(b,target,rng,slot)
        features=b.project_vision(fitted[None]);actual=b.values(b.commit_embeddings(features))[0]
        result['commit_l2']=float(np.linalg.norm(actual-target))
        result['passes_commit']=result['commit_l2']<4
        np.save(slot/'features.npy',features);dump(slot/'result.json',result)
        print('Native first-position result:',json.dumps(result),flush=True)

if __name__=='__main__':main()
