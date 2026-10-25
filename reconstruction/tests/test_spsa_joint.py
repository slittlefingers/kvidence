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

class JointSpsaTests(unittest.TestCase):
    def test_secant_proposal_reduces_linear_residual(self):
        spec=importlib.util.spec_from_file_location('joint',ROOT/'experiments/image_trust_region/run_spsa_joint.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rng=np.random.default_rng(3)
        delta=rng.normal(size=(16,24));a=rng.normal(size=(24,32));u=rng.normal(size=24);c=.1
        forward=lambda x:np.einsum('...i,ij->...j',x,a,optimize=False)
        q=forward(np.concatenate([u+c*delta,u-c*delta]))
        steps=module.ridge_steps(delta,q,np.zeros(32),c)
        self.assertTrue(np.isfinite(steps).all())
        self.assertLess(min(np.linalg.norm(forward(u+x)) for x in steps),np.linalg.norm(forward(u)))

    def test_periodic_centers_budget_and_no_unmeasured_acceptance(self):
        spec=importlib.util.spec_from_file_location('fast',ROOT/'experiments/image_trust_region/run_spsa_joint.py')
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
            np.savez(ref/'recovery/position_0005/initial.npz',inputs=np.zeros((1,768),dtype=np.float32),kv=np.zeros((1,1024)))
            calls=[]
            class Backend:
                def __init__(self,*a,**kw):self.quant=quant;self.position=5
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def reset(self,prefix):self.position=len(prefix)
                def query_vision(self,rows):calls.append(len(rows));return np.zeros((len(rows),1024),dtype=np.int8)
                def commit_embeddings(self,*args):raise AssertionError('Failed candidate must not be committed')
            args=['fast','--reference-run',str(ref),'--observation',str(obs),'--output',str(d/'out'),
                '--steps','7','--directions','2','--center-every','5']
            with patch.object(module,'NpuBackend',Backend),patch.object(module,'private_guard'),patch.object(sys,'argv',args):module.main()
            result=json.loads((d/'out/result.json').read_text());trace=json.loads((d/'out/optimization.json').read_text())
            self.assertEqual(result['queries'],183)
            self.assertFalse(result['passes_query'])
            self.assertEqual(calls,[1]+[4,22]*7+[32])
            self.assertEqual([r['round'] for r in trace if r['current_rmse'] is not None],list(range(1,8)))
            self.assertEqual(result['total_queries_including_warmup'],183)
            calls.clear()
            args[args.index('--output')+1]=str(d/'geometry')
            args.append('--geometry')
            with patch.object(module,'NpuBackend',Backend),patch.object(module,'private_guard'),patch.object(sys,'argv',args):module.main()
            geometry=json.loads((d/'geometry/result.json').read_text())
            protocol=json.loads((d/'geometry/protocol.json').read_text())
            self.assertEqual(geometry['queries'],183)
            self.assertFalse(geometry['passes_query'])
            self.assertEqual(calls,[1]+[4,22]*7+[32])
            self.assertEqual(protocol['ridge_residual'],'measured center')
            trace=json.loads((d/'geometry/optimization.json').read_text())
            self.assertTrue(all(r['changed_code_fraction']==0 for r in trace))
            calls.clear()
            args[args.index('--output')+1]=str(d/'surrogate')
            args.append('--surrogate')
            with patch.object(module,'NpuBackend',Backend),patch.object(module,'private_guard'),patch.object(sys,'argv',args):module.main()
            surrogate=json.loads((d/'surrogate/result.json').read_text())
            self.assertEqual(surrogate['queries'],183)
            self.assertFalse(surrogate['passes_query'])
            self.assertEqual(calls,[1]+[4,22]*7+[32])
            trace=json.loads((d/'surrogate/optimization.json').read_text())
            self.assertEqual([r['surrogate_train_rows'] for r in trace],[4]+[8]*6)
            calls.clear()
            args.remove('--surrogate')
            args[args.index('--output')+1]=str(d/'full768')
            args[args.index('--directions')+1]='768'
            args.extend(['--max-queries','1605'])
            with patch.object(module,'NpuBackend',Backend),patch.object(module,'private_guard'),patch.object(sys,'argv',args):module.main()
            full=json.loads((d/'full768/result.json').read_text())
            self.assertEqual(full['queries'],1605)
            self.assertFalse(full['passes_query'])
            self.assertEqual(calls,[1]+[128]*12+[22,24,22,32])
            trace=json.loads((d/'full768/optimization.json').read_text())
            self.assertEqual([r['directions'] for r in trace],[768,12])
            calls.clear()
            pos=args.index('--max-queries');del args[pos:pos+2]
            args[args.index('--directions')+1]='2'
            args[args.index('--output')+1]=str(d/'plateau')
            args.extend(['--plateau-rounds','2'])
            with patch.object(module,'NpuBackend',Backend),patch.object(module,'private_guard'),patch.object(sys,'argv',args):module.main()
            plateau=json.loads((d/'plateau/result.json').read_text())
            self.assertEqual(plateau['status'],'plateau')
            self.assertEqual(plateau['rounds'],2)
            self.assertEqual(plateau['queries'],53)
            self.assertEqual(calls,[1,4,22,4,22,32])
            args[args.index('--output')+1]=str(d/'continued')
            args.extend(['--resume-run',str(d/'plateau'),'--continue-tail'])
            with patch.object(module,'NpuBackend',Backend),patch.object(module,'private_guard'),patch.object(sys,'argv',args):module.main()
            continued=json.loads((d/'continued/result.json').read_text())
            protocol=json.loads((d/'continued/protocol.json').read_text())
            self.assertEqual(continued['status'],'plateau')
            self.assertEqual(continued['total_queries_including_warmup'],106)
            self.assertEqual(protocol['adam_offset'],2)
            self.assertEqual(protocol['restored_state'],'checkpoint')
            prior_trace=json.loads((d/'plateau/optimization.json').read_text())
            new_trace=json.loads((d/'continued/optimization.json').read_text())
            self.assertEqual(new_trace[0]['lr'],prior_trace[-1]['lr'])
            self.assertEqual(new_trace[-1]['adam_step'],4)

if __name__=='__main__':unittest.main()
