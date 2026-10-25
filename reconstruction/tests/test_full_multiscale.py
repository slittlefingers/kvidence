import importlib.util,json,sys,tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kvrecon.common import digest
from kvrecon.media.image_checkpoint import save_state

class FullMultiscaleTests(unittest.TestCase):
    def test_budget_batch_limit_and_failed_candidate_not_committed(self):
        spec=importlib.util.spec_from_file_location('fullmultiscale',ROOT/'experiments/image_trust_region/run_full_multiscale.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);obs=p/'obs';obs.mkdir();warm=p/'warm';warm.mkdir()
            np.savez(obs/'observed.npz',codes=np.ones((1,1024),dtype=np.int8))
            quant=[dict(width=512,scale=1.,zero_point=0,dtype='int8')]*2
            meta=dict(observation_sha256=digest(obs/'observed.npz'),quantization=quant,model_sha256='fake',layer_index=4,public_prefix=[2]*5,media_start=5)
            (obs/'metadata.json').write_text(json.dumps(meta))
            (warm/'protocol.json').write_text(json.dumps(dict(observation=meta,seed=17,geometry=True,distance_space='lsb',distance_metric='rmse',epsilon=.125)))
            (warm/'result.json').write_text(json.dumps(dict(total_queries_including_warmup=50,total_seconds_including_warmup=2.)))
            (warm/'optimization.json').write_text(json.dumps([dict(lr=.005,perturbation=.08)]))
            np.save(warm/'best_input.npy',np.zeros(768,dtype=np.float32));np.save(warm/'best_kv.npy',np.zeros(1024))
            save_state(warm/'optimizer_state.npz',np.random.default_rng(17),np.zeros(768),np.zeros(768),3)
            calls=[]
            class Backend:
                batch_capacity=128
                def __init__(self,*a,**kw):self.position=5;self.quant=quant
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def reset(self,prefix):self.position=len(prefix)
                def query_vision(self,rows):
                    self.assert_valid(rows);calls.append(len(rows));return np.zeros((len(rows),1024),dtype=np.int8)
                def assert_valid(self,rows):
                    if not 1<=len(rows)<=128 or not np.isfinite(rows).all():raise AssertionError('Invalid batch')
                def commit_embeddings(self,*args):raise AssertionError('Failed candidate must not commit')
            args=['test','--warm-run',str(warm),'--observation',str(obs),'--output',str(p/'out'),'--rounds','1']
            with patch.object(module,'NpuBackend',Backend),patch.object(module,'private_guard'),patch.object(sys,'argv',args):module.main()
            result=json.loads((p/'out/result.json').read_text());trace=json.loads((p/'out/optimization.json').read_text())
            self.assertEqual(result['queries'],4730);self.assertEqual(result['total_queries_including_warmup'],4780)
            self.assertFalse(result['passes_query']);self.assertEqual(calls,[1]+[128]*36+[121,32])
            self.assertEqual(result['best_rmse'],1.);self.assertEqual(trace[0]['selected']['kind'],'incumbent')
            proposals=json.loads((p/'out/round_0001_updates.json').read_text())
            self.assertEqual(len(proposals),121)
            self.assertTrue(any(row.get('factor')==1/64 for row in proposals))
            self.assertTrue(any(row.get('factor')==4. for row in proposals))
            calls.clear()
            args[args.index('--output')+1]=str(p/'adaptive')
            args[args.index('--rounds')+1]='10'
            args.extend(['--adaptive','--plateau-rounds','2','--expansion-trial-rounds','1'])
            with patch.object(module,'NpuBackend',Backend),patch.object(module,'private_guard'),patch.object(sys,'argv',args):module.main()
            result=json.loads((p/'adaptive/result.json').read_text())
            trace=json.loads((p/'adaptive/optimization.json').read_text())
            self.assertEqual(result['status'],'expansion_exhausted')
            self.assertEqual(result['rounds'],4);self.assertEqual(result['queries'],18917)
            self.assertEqual([r['expansion_event']['action'] for r in trace],['continue','expand','reject_and_expand','reject_and_stop'])
            self.assertTrue(all(r['scale_probes'][0]['probe_rms']==.02 for r in trace))
            self.assertEqual(result['best_rmse'],1.)
            calls.clear()
            mlp_args=['test','--warm-run',str(warm),'--observation',str(obs),'--output',str(p/'mlp'),'--rounds','1','--mlp-test']
            with patch.object(module,'NpuBackend',Backend),patch.object(module,'private_guard'),patch.object(sys,'argv',mlp_args):module.main()
            mlp_result=json.loads((p/'mlp/result.json').read_text())
            mlp_trace=json.loads((p/'mlp/optimization.json').read_text())
            self.assertEqual(mlp_result['queries'],4762)
            self.assertFalse(mlp_result['passes_query'])
            self.assertEqual(calls,[1]+[128]*36+[121,32,32])
            self.assertEqual(mlp_trace[0]['mlp']['mlp_parameters'],172224)
            self.assertEqual(mlp_trace[0]['mlp']['standard_best_rmse'],1.)
            from kvrecon.media.image_mlp import TinyMLPSurrogate
            class FailFirst(TinyMLPSurrogate):
                def fit(self,*args,**kwargs):raise ValueError('Nonfinite injected training failure')
            constructions=[]
            def create_model(*args,**kwargs):
                constructions.append(kwargs.get('lr',.003))
                return (FailFirst if len(constructions)==1 else TinyMLPSurrogate)(*args,**kwargs)
            mlp_args[mlp_args.index('--output')+1]=str(p/'mlp_retry')
            with patch.object(module,'NpuBackend',Backend),patch.object(module,'private_guard'),patch.object(sys,'argv',mlp_args),patch('kvrecon.media.image_mlp.TinyMLPSurrogate',side_effect=create_model):module.main()
            retry=json.loads((p/'mlp_retry/optimization.json').read_text())[0]['mlp']
            self.assertEqual(constructions,[.003,.0003])
            self.assertEqual(retry['mlp_lr'],.0003)
            self.assertIn('injected',retry['mlp_recovery'])

if __name__=='__main__':unittest.main()
