"""Text replay and ordinary-LM ranking control; not an inverse-model test."""
import json
import sys
import time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from kvrecon.common import WORKSPACE,EXPERIMENTS,dump,digest
from kvrecon.npu import NpuBackend


def main():
    out=HERE/'results/20261023_text_pipeline_control';out.mkdir(parents=True,exist_ok=False)
    root=WORKSPACE/'kvidence/datasets/shadow_chat/plaintext_chat_100'
    obs=root/'observations/0076/layer_04'
    meta=json.loads((obs/'metadata.json').read_text())
    truth=json.loads((root/'private_eval/0076/truth.json').read_text())
    with np.load(obs/'observed.npz') as f:
        codes=f['codes'].copy();prefix=f['public_prefix_ids'].tolist();positions=f['positions'].copy()
    assert digest(obs/'observed.npz')==meta['observed_sha256']
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    results=[]
    with NpuBackend(sections,4,out/'device',meta['model_sha256'],serial='YOUR_DEVICE_SERIAL') as b:
        b.reset(prefix)
        for i,token in enumerate(truth['token_ids']):
            assert b.position==int(positions[i])
            tick=time.perf_counter();order=b.probabilities();rank=int(np.flatnonzero(order==token)[0])+1
            ranking=time.perf_counter()-tick
            proposed=order[:128];tick=time.perf_counter();q=b.query(proposed)
            distances=np.sqrt(np.mean((q.astype(float)-codes[i].astype(float))**2,axis=1))
            query_seconds=time.perf_counter()-tick
            control=b.query([token])[0]
            actual=float(np.sqrt(np.mean((control.astype(float)-codes[i].astype(float))**2)))
            accepted=np.flatnonzero(distances<.125)
            results.append(dict(position=b.position,lm_rank=rank,truth_control_rmse=actual,
                top128_best_rmse=float(distances.min()),top128_accepted_correct=bool(any(proposed[j]==token for j in accepted)),
                ranking_seconds=ranking,query128_seconds=query_seconds))
            b.commit(token)
    report=dict(sample='0076',purpose='pipeline calibration only; not held-out inverse-model evaluation',
        tokens=len(results),true_kv_all_exact=all(x['truth_control_rmse']==0 for x in results),
        ordinary_lm_topk={str(k):sum(x['lm_rank']<=k for x in results)/len(results) for k in [1,8,32,128]},rows=results)
    dump(out/'result.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
