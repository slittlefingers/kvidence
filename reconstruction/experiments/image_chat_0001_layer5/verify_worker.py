"""Known-token equivalence control for the continuous-input worker."""
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT))
from kvrecon.common import dump, EXPERIMENTS
from kvrecon.npu import NpuBackend
from kvrecon.cli import private_guard

def main():
    private_guard()
    meta=json.loads((HERE/'input/metadata.json').read_text())
    sections=json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    with np.load(HERE/'input/observed.npz') as d:
        target=d['prefix_kv'].copy()
    out=HERE/'worker_control';out.mkdir(exist_ok=False)
    with NpuBackend(sections,4,out,meta['model_sha256']) as b:
        b.reset([])
        prefix=np.concatenate([b.values(b.commit(t)) for t in meta['public_prefix']])
        token=42
        embedding=np.frombuffer(b.rpc('embedding',id=token),dtype='<f4').copy()[None]
        ordinary=b.commit(token); ordinary_logits=b.rpc('logits')
        b.reset(meta['public_prefix'])
        continuous=b.commit_embeddings(embedding,ple_id=token)
        continuous_logits=b.rpc('logits')
        np.testing.assert_array_equal(ordinary,continuous)
        assert ordinary_logits==continuous_logits
        assert b.position==6
        rejected=[]
        for bad in (np.zeros((1,1)),np.full((1,1536),np.nan),np.empty((0,1536))):
            try: b.commit_embeddings(bad)
            except ValueError: rejected.append(True)
            else: raise AssertionError('Malformed embeddings accepted')
        assert b.position==6
        result=dict(known_token_id=token,exact_kv_equal=True,exact_logits_equal=True,
            invalid_inputs_rejected=len(rejected),position_unchanged_after_rejections=True,
            public_prefix_l2=np.linalg.norm(prefix.astype(np.float64)-target,axis=-1).tolist(),
            scope='ordinary-token equivalence; not a proof of vision prefill equivalence')
        dump(out/'result.json',result)
        print(json.dumps(result),flush=True)

if __name__=='__main__':main()
