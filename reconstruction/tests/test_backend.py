"""Behavioral contract tests with a prefix-dependent CPU backend, no LiteRT."""
import sys
import types
import os
import json
import subprocess
import tempfile
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.backend import CandidateBatchBackend, validate_contract
from kvrecon.adapters import get_adapter
from kvrecon.search import recover_position, SearchConfig


class CpuBackend:
    vocab_size=8
    batch_capacity=1
    decoder_seconds=0.
    setup_seconds=0.

    def reset(self,prefix):
        self.prefix=list(prefix);self.position=len(prefix)

    def probabilities(self):
        return np.arange(self.vocab_size)

    def query(self,ids):
        assert len(ids)<=self.batch_capacity
        return np.array([[sum(self.prefix)+i, self.position+i*2] for i in ids],dtype=np.float32)

    def commit(self,token):
        value=self.query([token]);self.prefix.append(token);self.position+=1
        return value

    def values(self,codes):
        return np.asarray(codes,dtype=np.float64)


class BackendTests(unittest.TestCase):
    def test_uncertain_publish_requires_ack_and_never_resends(self):
        from kvrecon.npu import NpuBackend
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as d:
            b=NpuBackend.__new__(NpuBackend)
            b.transport='file';b.rpc_broken=False
            b.seq=0;b.remote='fake';b.output=Path(d);b.decoder_seconds=0.;b.calls=[]
            b.write=Mock();b.shell=Mock(side_effect=subprocess.CalledProcessError(1,['mv'],stderr=b'late error'))
            b._wait=Mock(return_value=dict(seq=1,ok=True,payload_bytes=0,position=1,decoder_seconds=0.))
            self.assertEqual(b.rpc('commit',id=3),b'')
            self.assertEqual(b.write.call_count,1);self.assertEqual(b.shell.call_count,1)
            b._wait.assert_called_once_with('response.json',1)
            b._wait=Mock(side_effect=TimeoutError('no acknowledgement'))
            with self.assertRaises(TimeoutError):b.rpc('commit',id=4)
            self.assertEqual(b.position,1)

    def test_external_serial_adapter_cli_end_to_end(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            (d/'model').write_text('synthetic-model')
            (d/'manifest.json').write_text(json.dumps(dict(vocab_size=8,embedding_dim=2,context_capacity=32,batch_capacity=1)))
            (d/'tokenizer.json').write_text('{}')
            np.save(d/'embeddings.npy',np.zeros((8,2),dtype=np.float32))
            (d/'profile.json').write_text(json.dumps(dict(name='synthetic',model='model',manifest='manifest.json',
                tokenizer='tokenizer.json',embeddings='embeddings.npy',vocabulary=None,default_layer=0,
                adapter='support_cpu_adapter:Adapter')))
            (d/'prefix.json').write_text('[2]');(d/'ids.json').write_text('[3,5]')
            env=dict(os.environ,PYTHONPATH=str(root/'tests')+os.pathsep+str(root))
            def run(*args):
                subprocess.run([sys.executable,str(root/'run.py'),*map(str,args)],env=env,check=True,capture_output=True)
            run('fixture','--model-config',d/'profile.json','--prefix-json',d/'prefix.json',
                '--token-ids-json',d/'ids.json','--output',d/'fixture')
            run('run','--model-config',d/'profile.json','--observation',d/'fixture/observation',
                '--method','collision_plus','--epsilon','.01','--batch-size','8','--output',d/'result')
            result=json.loads((d/'result/result.json').read_text())
            self.assertEqual(result['recovered_token_ids'],[3,5])
            protocol=json.loads((d/'result/protocol.json').read_text())
            self.assertEqual(protocol['capabilities']['batch_capacity'],1)
            self.assertEqual(protocol['adapter'],'support_cpu_adapter:Adapter')
            run('check-adapter','--model-config',d/'profile.json','--prefix-json',d/'prefix.json',
                '--candidate-ids-json',d/'ids.json','--output',d/'check')
            self.assertTrue(json.loads((d/'check/validation.json').read_text())['passed'])
            profile=json.loads((d/'profile.json').read_text());profile['adapter']='support_cpu_adapter:FirstAdapter'
            (d/'profile.json').write_text(json.dumps(profile))
            run('fixture','--model-config',d/'profile.json','--prefix-json',d/'prefix.json',
                '--token-ids-json',d/'ids.json','--output',d/'first_fixture')
            run('build-v-table','--model-config',d/'profile.json','--output',d/'vtable')
            run('run','--model-config',d/'profile.json','--observation',d/'first_fixture/observation',
                '--v-table',d/'vtable','--output',d/'first_result')
            first=json.loads((d/'first_result/result.json').read_text())
            self.assertEqual(first['method'],'first_v');self.assertEqual(first['recovered_token_ids'],[3,5])
            run('evaluate','--run',d/'first_result','--truth',d/'first_fixture/private_eval/truth.json')
            self.assertTrue(json.loads((d/'first_result/evaluation.json').read_text())['exact_match'])

    def test_serial_backend_contract_and_reset(self):
        b=CpuBackend()
        r=validate_contract(b,[2],[1,3,5])
        self.assertTrue(r['passed']);self.assertEqual(b.prefix,[2])
        self.assertEqual(len(r['checks']),5)

    def test_real_batch_and_serial_search_have_same_semantics(self):
        # Multiple passing candidates: Collision+ must keep probability order;
        # guided must choose stage minimum, including across physical calls.
        for method in ('guided','collision_plus'):
            results=[]
            for capacity in (1,3,8):
                b=CpuBackend();b.batch_capacity=capacity;b.reset([2])
                wrapper=CandidateBatchBackend(b,8)
                result=recover_position(wrapper,np.array([5.,7.]),np.arange(8),0,
                    SearchConfig(method=method,batch_size=8,probability_seeds=8,random_seeds=0,epsilon=4.))
                results.append(result)
                self.assertEqual(b.prefix,[2])
                self.assertEqual(result['stages'][0]['physical_slots'],capacity)
            for r in results[1:]:
                for key in ('accepted_token_id','queried_token_ids','distances','passing_ids_in_accepting_stage'):
                    self.assertEqual(r[key],results[0][key])

    def test_hidden_cache_mutation_is_detected(self):
        class Broken(CpuBackend):
            def query(self,ids):
                value=super().query(ids);self.prefix[0]+=1
                return value
        with self.assertRaises(AssertionError):
            validate_contract(Broken(),[2],[1,3])

    def test_commit_mismatch_detected(self):
        class Broken(CpuBackend):
            def commit(self,token):
                return super().commit(token)+1
        with self.assertRaises(AssertionError):
            validate_contract(Broken(),[2],[1,3])

    def test_report_numeric_tolerance(self):
        class Noisy(CpuBackend):
            def commit(self,token):
                return super().commit(token)+0.001
        r=validate_contract(Noisy(),[2],[1,3],atol=.002)
        error=next(x['maximum_absolute_error'] for x in r['checks'] if x['name']=='commit_vs_query')
        self.assertGreater(error,0)

    def test_external_adapter_factory_and_bad_plugin(self):
        # No registry/source edits needed for a local module:factory plugin.
        module=types.ModuleType('kvrecon_test_plugin')
        module.Factory=lambda:get_adapter('litert_tensor_gemma')
        module.Broken=lambda:object()
        sys.modules[module.__name__]=module
        try:
            self.assertTrue(callable(get_adapter('kvrecon_test_plugin:Factory').open))
            with self.assertRaises(TypeError):get_adapter('kvrecon_test_plugin:Broken')
            with self.assertRaises(ValueError):get_adapter('does_not_exist')
        finally:
            del sys.modules[module.__name__]


if __name__=='__main__':unittest.main()
