"""Metric migration checks: real integer codes, no dequantization in guided search."""
from pathlib import Path
import sys
import tempfile
import unittest
import json
import os
import subprocess
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.common import code_values, distance_values, validate_quantization_grid
from kvrecon.search import SearchConfig, recover_position
from kvrecon.engine import reconstruct
from kvrecon.first_v import nearest_v
from kvrecon.observation import save_canonical_observation, load_observation
from kvrecon.cli import build_parser

QUANT=[dict(width=1,scale=.001,zero_point=-3,dtype='int8'),
       dict(width=1,scale=.02,zero_point=7,dtype='int8')]


class IntegerBackend:
    batch_capacity=128
    decoder_seconds=0.
    quant=QUANT
    def __init__(self, table):
        self.table=np.asarray(table,dtype=np.int8)
        self.vocab_size=len(table);self.position=1;self.commits=[]
    def reset(self, prefix):self.position=len(prefix)
    def probabilities(self):return np.arange(self.vocab_size)
    def query(self, ids):return self.table[ids].copy()
    def commit(self, token):
        self.commits.append(token);self.position+=1
        return self.table[token:token+1].copy()
    def values(self, codes):raise AssertionError('Guided LSB path must not dequantize')


class LsbTests(unittest.TestCase):
    def test_widen_before_subtraction_and_no_scale_weight(self):
        a=np.array([[127,-128]],dtype=np.int8)
        b=np.array([[-128,127]],dtype=np.int8)
        delta=distance_values(a,QUANT)-distance_values(b,QUANT)
        np.testing.assert_array_equal(delta,[[255,-255]])
        self.assertEqual(float(np.linalg.norm(code_values(np.array([3,4],dtype=np.int8)))),5.)

    def test_ranking_changes_when_scales_are_not_used(self):
        # Raw units prefer [10,0] (0.01 vs 0.02); LSB prefers [0,1].
        backend=IntegerBackend([[10,0],[0,1]])
        row=recover_position(backend,[0.,0.],backend.probabilities(),0,
            SearchConfig(probability_seeds=2,random_seeds=0,epsilon=20))
        self.assertEqual(row['accepted_token_id'],1)
        np.testing.assert_allclose(row['distances'],np.array([10.,1.])/np.sqrt(2))

    def test_fit_targets_and_batch_events_stay_in_code_space(self):
        backend=IntegerBackend([[20,0],[10,10],[0,0]])
        class Predictor:
            def scores(self, ids, values, target, distances):
                np.testing.assert_array_equal(values,[[20,0],[10,10]])
                np.testing.assert_array_equal(target,[0,0])
                np.testing.assert_allclose(distances,[np.sqrt(200),10])
                return [2.,1.,0.]
        events=[]
        row=recover_position(backend,[0.,0.],backend.probabilities(),0,
            SearchConfig(probability_seeds=2,random_seeds=0,proposals=1),Predictor(),on_batch=events.append)
        self.assertEqual(row['accepted_token_id'],2)
        self.assertEqual([e['distance_space'] for e in events],['lsb','lsb'])

    def test_strict_l2_gate_and_commit_recheck(self):
        backend=IntegerBackend([[4,0],[3,0]])
        row=recover_position(backend,[0.,0.],backend.probabilities(),0,
            SearchConfig(method='collision_plus',epsilon=4,distance_metric='l2'))
        self.assertEqual(row['accepted_token_id'],1)  # exactly 4 is excluded
        class Drift(IntegerBackend):
            def commit(self, token):
                super().commit(token)
                return np.array([[4,0]],dtype=np.int8)
        with tempfile.TemporaryDirectory() as folder:
            result=reconstruct(Drift([[0,0]]),np.array([[0.,0.],[0.,0.]]),[1,2],[2],
                SearchConfig(method='collision_plus'),folder)
        self.assertEqual(result['status'],'commit_mismatch')
        self.assertEqual(result['recovered_token_ids'],[])

    def test_observation_and_first_v_do_not_dequantize(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'obs'
            save_canonical_observation(output,np.array([[10,20]],dtype=np.int8),[1],[2],4,QUANT,'model',{})
            with patch('kvrecon.common.dequantize',side_effect=AssertionError('unexpected conversion')):
                _,_,target,_,_=load_observation(output,distance_space='auto')
                np.testing.assert_array_equal(target,[[10,20]])
                row=nearest_v(np.array([[20],[25]],dtype=np.int8),np.array([1,2]),[QUANT[1]],target[0,1:],4)
        self.assertEqual(row['accepted_token_id'],1)
        self.assertEqual(row['second_l2'],5.)

    def test_grid_mismatch_and_float_lsb_rejected(self):
        validate_quantization_grid(QUANT,[{k:v for k,v in q.items() if k!='dtype'} for q in QUANT])
        for key,value in [('scale',.5),('zero_point',5),('dtype','float32')]:
            changed=[dict(QUANT[0],**{key:value}),QUANT[1]]
            with self.assertRaisesRegex(ValueError,'grids differ'):
                validate_quantization_grid(QUANT,changed)
        with self.assertRaisesRegex(ValueError,'native integer'):
            code_values(np.array([1.],dtype=np.float32))

    def test_cpa_retains_original_value_acceptance(self):
        class CpaBackend(IntegerBackend):
            def values(self,codes):return distance_values(codes,QUANT,'original')
            def query_retained(self,ids):return self.query(ids)
        backend=CpaBackend([[100,100]])
        # Raw differences .1 and 2 pass; LSB differences 100 would fail.
        target=distance_values(np.array([0,0],dtype=np.int8),QUANT,'original')
        row=recover_position(backend,target,backend.probabilities(),0,
            SearchConfig(method='collision_plus_cpa',kv_thresholds=(.2,3.),k_width=1,distance_space='lsb'))
        self.assertEqual(row['accepted_token_id'],0)

    def test_cli_default_is_auto(self):
        args=build_parser().parse_args(['run','--observation','obs','--output','out'])
        self.assertEqual(args.distance_space,'auto')
        self.assertEqual(args.epsilon,.125)
        self.assertEqual(args.distance_metric,"rmse")

    def test_integer_cli_end_to_end_without_dequantization(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder)
            (d/'model').write_text('synthetic-integer-model')
            (d/'manifest.json').write_text(json.dumps(dict(vocab_size=8,embedding_dim=2,context_capacity=32,batch_capacity=1)))
            (d/'tokenizer.json').write_text('{}')
            np.save(d/'embeddings.npy',np.zeros((8,2),dtype=np.float32))
            (d/'profile.json').write_text(json.dumps(dict(name='integer',model='model',manifest='manifest.json',
                tokenizer='tokenizer.json',embeddings='embeddings.npy',vocabulary=None,default_layer=0,
                adapter='support_cpu_adapter:IntegerAdapter')))
            (d/'prefix.json').write_text('[2]');(d/'ids.json').write_text('[3,5]')
            env=dict(os.environ,PYTHONPATH=str(root/'tests')+os.pathsep+str(root))
            def run(*args):
                result=subprocess.run([sys.executable,str(root/'run.py'),*map(str,args)],env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            run('fixture','--model-config',d/'profile.json','--prefix-json',d/'prefix.json',
                '--token-ids-json',d/'ids.json','--output',d/'fixture')
            run('run','--model-config',d/'profile.json','--observation',d/'fixture/observation',
                '--method','collision_plus','--epsilon','.1','--batch-size','8','--live-events','--output',d/'result')
            result=json.loads((d/'result/result.json').read_text())
            protocol=json.loads((d/'result/protocol.json').read_text())
            self.assertEqual(result['recovered_token_ids'],[3,5])
            self.assertEqual(result['config']['distance_space'],'lsb')
            self.assertEqual(protocol['metric'],'lsb_kv_rmse')
            events=[json.loads(line) for line in (d/'result/events.jsonl').read_text().splitlines()]
            self.assertTrue(all(e['distance_space']=='lsb' for e in events if e['type']=='batch'))


if __name__=='__main__':unittest.main()
