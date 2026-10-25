"""Validate periodic measured acceptance and accounting without a phone."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'experiments/image_trust_region'))

class TokenSeededTests(unittest.TestCase):
    def test_periodic_centers_budget_and_no_unmeasured_acceptance(self):
        spec=importlib.util.spec_from_file_location('fast',ROOT/'experiments/image_trust_region/run_token_seeded.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from kvrecon.common import digest
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);obs=d/'obs';obs.mkdir();ref=d/'ref';(ref/'recovery/position_0005').mkdir(parents=True)
            np.savez(obs/'observed.npz',codes=np.ones((1,1024),dtype=np.int8))
            quant=[dict(width=512,scale=1.,zero_point=0,dtype='int8')]*2
            meta=dict(observation_sha256=digest(obs/'observed.npz'),quantization=quant,
                model_sha256='fake',layer_index=4,public_prefix=[2]*5,media_start=5)
            (obs/'metadata.json').write_text(json.dumps(meta))
            (ref/'protocol.json').write_text(json.dumps(dict(observation=meta,distance_space='lsb',optimizer=dict(metric='rmse',seed=29,epsilon=.125))))
            np.save(ref/'best_input.npy',np.zeros(768,dtype=np.float32))
            np.save(ref/'best_kv.npy',np.zeros(1024))
            (ref/'result.json').write_text(json.dumps(dict(total_seconds_including_warmup=0.,total_queries_including_warmup=1)))
            calls=[]
            class Backend:
                def __init__(self,*a,**kw):self.quant=quant;self.position=5;self.vocab_size=64
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def reset(self,prefix):self.position=len(prefix)
                def rpc(self,op,**kwargs):
                    assert op=='embedding'
                    return np.ones(1536,dtype=np.float32).tobytes()
                def project_vision(self,*args):raise AssertionError('Adapter must not be called')
                def query_embeddings(self,rows):calls.append(len(rows));return np.zeros((len(rows),1024),dtype=np.int8)
                def commit_embeddings(self,*args):raise AssertionError('Failed candidate must not be committed')
            args=['fast','--observation',str(obs),'--output',str(d/'out'),
                '--rounds','2','--seconds','10','--seed-candidates','4']
            with patch.object(module,'NpuBackend',Backend),patch.object(module,'private_guard'),patch.object(sys,'argv',args):module.main()
            result=json.loads((d/'out/result.json').read_text());trace=json.loads((d/'out/optimization.json').read_text())
            self.assertEqual(result['queries'],422)
            self.assertFalse(result['passes_query'])
            self.assertEqual(calls,[4,128,81,128,81,32])
            self.assertEqual(len(trace),2)
            self.assertFalse(result['adapter_called'])
            self.assertEqual(result['best_rmse'],1.0)

if __name__=='__main__':unittest.main()
