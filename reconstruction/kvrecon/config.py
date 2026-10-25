"""Model assets and original-ID candidate pools; no token ID remapping."""
import json
from pathlib import Path
import numpy as np
from .common import ROOT, digest

DEFAULT_MODEL_CONFIG = ROOT / "configs/gemma4_e2b.json"
MODEL_CONFIGS={"gemma4-e2b":DEFAULT_MODEL_CONFIG,"gemma3-1b":ROOT/"configs/gemma3_1b.json"}


def load_model_config(path):
    path=Path(path).resolve();data=json.loads(path.read_text())
    required={"name","model","manifest","tokenizer","embeddings","vocabulary"}
    optional={"default_layer","adapter","adapter_options","device_model","batch_signature","mask_token_id","first_v_table","media"}
    if not required.issubset(data) or set(data)-required-optional:
        raise ValueError(f"Model config keys must be {sorted(required)}")
    data.setdefault('first_v_table',None)
    data.setdefault('media',{})
    if not isinstance(data['media'],dict):
        raise ValueError('media must be a model-specific rule object')
    for key in ("model","manifest","tokenizer","embeddings","vocabulary","first_v_table"):
        if data[key] is not None:
            data[key]=(path.parent/Path(data[key])).resolve()
    data.setdefault("mask_token_id",1437)
    data.setdefault("batch_signature","prefill_128")
    data.setdefault("device_model",None)
    data.setdefault("adapter_options",{})
    if not isinstance(data["adapter_options"],dict):
        raise ValueError("adapter_options must be an object")
    if not isinstance(data["mask_token_id"],int) or isinstance(data["mask_token_id"],bool):
        raise ValueError("mask_token_id must be an integer")
    if data["device_model"] is not None and (not isinstance(data["device_model"],str) or not data["device_model"].startswith("/")):
        raise ValueError("device_model must be an absolute Android path")
    data.setdefault("adapter","litert_tensor_gemma")
    if not isinstance(data["adapter"],str) or not data["adapter"]:
        raise ValueError("adapter must be a nonempty registry name or module:factory")
    data.setdefault("default_layer",4)
    if type(data["default_layer"]) is not int or data["default_layer"]<0:
        raise ValueError("default_layer must be a nonnegative producer index")
    return data


def model_dimensions(sections,signature="prefill_128"):
    sig=sections["tf_lite_prefill_decode"]["signatures"]
    inputs={x["name"]:x for x in sig["decode"]["inputs"]}
    outputs={x["name"]:x for x in sig["decode"]["outputs"]}
    batched={x["name"]:x for x in sig[signature]["inputs"]}
    vocab=outputs["logits"]["shape"][-1]
    embedding=inputs["embeddings"]["shape"][-1]
    capacity=batched["embeddings"]["shape"][1]
    cache=next(x for n,x in inputs.items() if n.startswith("kv_cache_k_"))
    context=cache["shape"][2]
    if min(vocab,embedding,capacity,context)<=0:
        raise ValueError("Dynamic dimensions are not supported by this worker")
    return dict(vocab_size=vocab,embedding_dim=embedding,batch_capacity=capacity,context_capacity=context)


def load_vocabulary(path,model_vocab_size):
    if path is None:
        return np.arange(model_vocab_size,dtype=np.int64),dict(kind="full_original",count=model_vocab_size,filtered=False)
    data=json.loads(Path(path).read_text())
    ids=data.get("token_ids") if isinstance(data,dict) else data
    if not isinstance(ids,list) or not ids or any(type(x) is not int for x in ids):
        raise ValueError("Vocabulary must contain a nonempty integer token_ids array")
    if len(set(ids))!=len(ids) or min(ids)<0 or max(ids)>=model_vocab_size:
        raise ValueError("Vocabulary IDs must be unique and inside the original model vocabulary")
    pool=np.asarray(sorted(ids),dtype=np.int64)
    return pool,dict(kind="explicit_original_ids",count=len(pool),filtered=len(pool)!=model_vocab_size,
        path=str(Path(path).resolve()),sha256=digest(path),ids_remapped=False)
