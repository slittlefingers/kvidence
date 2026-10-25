"""Short diagnostic of the archived shortlist policy; no target token labels."""
import time
import torch
import numpy as np
import recovery as r
from scorer import IndexedSurrogate

r.OUT=r.ROOT/'archive_v1/results'
r.guard()
rt,prior,tokenizer,runtime=r.exp.load_runtime()
obs,ids,pool=r.inputs()
emb=torch.from_numpy(np.load(r.ASSETS/'embeddings.npy'))
engine=IndexedSurrogate(rt,emb)
base_fit=engine.fit


def logged_fit(*args,**kw):
    tick=time.perf_counter()
    result=base_fit(*args,**kw)
    if len(args[0])<400 or len(args[0])%200==0:
        print('FIT queried',len(args[0]),'best real L2',min(args[3]),
              'seconds',time.perf_counter()-tick,'detail',engine.records[-1],flush=True)
    return result


engine.fit=logged_fit
cache,logits=runtime.prepare(obs['known_prefix_ids'].tolist(),'native')
for j in range(5):
    print('POSITION',j,flush=True)
    order=r.exp.ordered_probabilities(rt,logits,ids)
    row=r.full.fast_search(rt,runtime,cache,ids,emb,torch.from_numpy(obs['kv'][j]),order,j,engine,False)
    print('DONE',j,row['queries'],row['accepted_raw_l2'],flush=True)
    runtime.base.layers=runtime.full_layers
    output=runtime.model(rt.mx.array([[row['accepted_token_id']]]),cache=cache)
    rt.mx.eval(output);logits=output[0,-1]
