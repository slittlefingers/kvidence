"""Numerical preflight diagnostic only; no target labels or candidate exclusions."""
import json
import numpy as np
import torch
import experiment as e
from adaptive import FastSurrogate as ExpandedQuadraticSurrogate

e.guard()
rt,prior,tokenizer,runtime=e.exp.load_runtime()
obs,ids,pool=e.inputs(runtime)
emb=runtime.embeddings(ids.tolist())
cache,logits=runtime.prepare(obs['known_prefix_ids'].tolist(),'native')
order=e.exp.ordered_probabilities(rt,logits,ids)
rest=np.setdiff1d(np.arange(len(ids)),order[:50])
initial=np.r_[order[:50],np.random.default_rng(20261019).choice(rest,150,replace=False)]
values=rt.Oracle(runtime,cache,ids,5,64).query(initial)
target=torch.from_numpy(obs['kv'][0])
d=np.linalg.norm(values.numpy().astype(np.float64)-target.numpy().astype(np.float64),axis=-1)
engine=ExpandedQuadraticSurrogate(rt,emb)
fit=engine.fit(initial.tolist(),values,target,d)
mean,scale,xm,xc,dual,ymean,ys,ym=fit['reference_factors']
subset=np.unique(np.r_[np.random.default_rng(187).choice(len(ids),1024,replace=False),
    np.argsort(fit['scores'])[:20],pool['special_token_ids'],0,len(ids)-1])
x=(emb[torch.tensor(subset)].double()-mean.double())/scale.double()-xm.double()
g=x@xc.double().T
b=dual.double()*ys.double();t=target.double()-(ym.double()+ymean.double()*ys.double())
reference=(g@b-t).square().sum(-1).numpy()
mx=rt.mx;gpu=lambda a:mx.array(a.contiguous().float().numpy())
g_full=engine.e@gpu((xc/scale).T)-gpu((mean/scale+xm)@xc.T)
g_part=g_full[mx.array(subset)]
direct_gpu=mx.sum((g_part@gpu(b)-gpu(t)).square(),axis=-1)
mx.eval(g_part,direct_gpu)
g_cpu=torch.from_numpy(np.array(g_part)).double()
cpu_from_gpu_g=(g_cpu@b-t).square().sum(-1).numpy()
error=np.abs(fit['scores'][subset]-reference)
worst=np.argsort(error)[-12:][::-1]
print(json.dumps(dict(
    maximum_embedding_abs=float(emb.abs().max()),sample_scale_min=float(scale.min()),
    maximum_g_reference_abs=float(g.abs().max()),maximum_g_error=float((g_cpu-g).abs().max()),
    max_factorized_error=float(error.max()),
    max_direct_gpu_error=float(np.max(np.abs(np.array(direct_gpu)-reference))),
    max_cpu_from_gpu_g_error=float(np.max(np.abs(cpu_from_gpu_g-reference))),
    worst=[dict(token_id=int(subset[j]),reference=float(reference[j]),
        factorized=float(fit['scores'][subset[j]]),direct_gpu=float(np.array(direct_gpu)[j]),
        cpu_from_gpu_g=float(cpu_from_gpu_g[j]),embedding_norm=float(emb[int(subset[j])].norm())) for j in worst]),indent=2))
