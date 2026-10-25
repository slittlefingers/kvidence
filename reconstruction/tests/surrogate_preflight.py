"""Numerical adapter check with saved NPU KV; not a recovery success claim."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.common import EXPERIMENTS, DEFAULT_EMBEDDINGS, quantization, dequantize, dump
from kvrecon.surrogate import ExistingRidge


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    source=EXPERIMENTS/"npu_candidate_batch/results_prefill128_retry"
    job=json.loads((source/"input.json").read_text());ids=[c["id"] for c in job["candidates"]]
    keys=np.fromfile(source/"data/p32-batch-kv_slice_k_4.bin",dtype=np.int8).reshape(128,512)
    values=np.fromfile(source/"data/p32-batch-kv_slice_v_4.bin",dtype=np.int8).reshape(512,128).T
    observed=dequantize(np.concatenate([keys,values],axis=1),quantization(job["model_sections"],4,"prefill_128"))
    engine=ExistingRidge(DEFAULT_EMBEDDINGS,262144);target=observed[100];q=ids[:96]
    torch=engine.torch;tensor=lambda v:torch.from_numpy(np.asarray(v,dtype=np.float32))
    started=time.perf_counter()
    fit=engine.engine.fit(q,tensor(observed[:96]),tensor(target),np.linalg.norm(observed[:96]-target,axis=1))
    elapsed=time.perf_counter()-started
    scores=fit["scores"];assert scores.shape==(262144,) and np.isfinite(scores).all()
    sample=np.unique(np.r_[np.argsort(scores)[:20],np.random.default_rng(19).choice(262144,64,replace=False)])
    mean,scale,xm,xc,dual,ymean,ys,ym=fit["reference_factors"]
    x=(engine.engine.embeddings[sample].double()-mean.double())/scale.double()-xm.double()
    direct=(x@xc.double().T@dual.double()+ymean.double())*ys.double()+ym.double()
    reference=(direct-tensor(target).double()).square().sum(1).numpy()
    np.testing.assert_allclose(scores[sample],reference,rtol=3e-4,atol=.05)
    result=dict(passed=True,source=str(source),anchors=96,original_vocabulary_scores=len(scores),
        finite_full_vocabulary=True,target_held_out_of_fit=True,checked_direct_float64=len(sample),
        maximum_absolute_squared_l2_error=float(np.max(np.abs(scores[sample]-reference))),fit_and_score_seconds=elapsed,
        note="Numerical compatibility check only; does not measure search success or prove embedding quantization equivalence")
    dump(a.output/"validation.json",result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
