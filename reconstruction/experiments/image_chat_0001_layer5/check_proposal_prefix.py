"""Diagnostic only: compare public prefix across the two model graphs."""
import os
os.environ['HF_HUB_OFFLINE']='1'
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from kvrecon.common import dump,EXPERIMENTS
import mlx.core as mx
from mlx_vlm import load
model,_=load(str(EXPERIMENTS/'kv_guided_search_gemma4_int8/model'));model.eval()
from kvrecon.cli import private_guard
private_guard()
with np.load(HERE/'input/observed.npz') as d:
    ids=d['input_ids'][:,:5].copy();target=d['prefix_kv'].copy()
cache=model.make_cache()
output=model.language_model(mx.array(ids),cache=cache,logits_to_keep=1)
parts=[x[:,:,:5].transpose(0,2,1,3).reshape(5,-1) for x in (cache[4].keys,cache[4].values)]
actual=np.array(mx.concatenate(parts,-1).astype(mx.float32))
distance=np.linalg.norm(actual-target,axis=-1)
out=HERE/'public_prefix_comparison';out.mkdir(exist_ok=False)
np.save(out/'mlx_kv.npy',actual)
np.save(out/'mlx_embeddings.npy',np.array((model.language_model.model.embed_tokens(mx.array(ids))*model.language_model.model.embed_scale).astype(mx.float32)))
result=dict(public_ids=ids.tolist(),l2=distance.tolist(),
    k_l2=np.linalg.norm(actual[:,:512]-target[:,:512],axis=-1).tolist(),
    v_l2=np.linalg.norm(actual[:,512:]-target[:,512:],axis=-1).tolist(),
    mlx_norm=np.linalg.norm(actual,axis=-1).tolist(),npu_norm=np.linalg.norm(target,axis=-1).tolist())
dump(out/'result.json',result)
print(result,flush=True)
