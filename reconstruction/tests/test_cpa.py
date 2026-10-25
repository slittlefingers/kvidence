import ast
import json
from pathlib import Path
import sys
import tempfile
from typing import List, Tuple
import unittest
from unittest.mock import patch
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.common import ROOT, dump
from kvrecon.cpa import fit_thresholds, summarize_distances, component_distances, collect, load_calibration
from kvrecon.search import SearchConfig, recover_position, acceptance
from kvrecon.engine import reconstruct
from kvrecon.backend import CandidateBatchBackend
from kvrecon.cli import build_parser,resolve_search_defaults
from test_core import FakeBackend


class RetainedFake(FakeBackend):
    batch_capacity=2
    def begin_candidates(self):self.saved={}
    def query(self,ids):
        codes=super().query(ids)
        if hasattr(self,'saved'):
            for token,row in zip(ids,codes):self.saved[token]=row.copy()
        return codes
    def accept_query(self,token):
        self.position+=1
        return self.saved[token][None,:]


class CPATests(unittest.TestCase):
    def test_thresholds_against_unmodified_upstream_functions(self):
        # Extract only pure threshold definitions; never import/execute attacks.
        path=ROOT.parent/'references/kvcloak/attack/collision.py'
        tree=ast.parse(path.read_text())
        nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('ThresholdClassifier','get_collision_threshold')]
        namespace=dict(np=np,norm=norm,minimize_scalar=minimize_scalar,Path=Path,json=json,List=List,Tuple=Tuple)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
        cases=[dict(target_mean=[.2,.3],target_std=[.1,.2],target_max=[.4,.6],others_min=[1.,2.],others_mean=[3.,4.],others_std=[.5,.6]),
               dict(target_mean=[2.,3.],target_std=[.3,.5],target_max=[5.,6.],others_min=[1.,2.],others_mean=[4.,5.],others_std=[.8,1.]),
               dict(target_mean=[0.,.3],target_std=[0.,.2],target_max=[0.,.6],others_min=[0.,2.],others_mean=[3.,4.],others_std=[.5,.6])]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'stats.json'
            for stats in cases:
                dump(p,[stats])
                for rank in (8,64):
                    expected,_=namespace['get_collision_threshold'](str(p),0,rank)
                    np.testing.assert_array_equal(fit_thresholds(stats,rank)['thresholds'],expected)

    def test_dual_norm_strict_boundary_first_pass_not_minimum(self):
        b=CandidateBatchBackend(RetainedFake([[.5,2.],[1.,1.],[0.,0.],[9.,9.]]),4)
        c=SearchConfig(method='collision_plus_cpa',batch_size=4,epsilon=.001,k_width=1,kv_thresholds=(1.,1.))
        r=recover_position(b,np.zeros(2),np.arange(4),0,c)
        self.assertEqual(r['accepted_token_id'],1)
        self.assertEqual(r['queries'],4)
        np.testing.assert_array_equal(b.accept_query(1),[[1.,1.]])
        # Small concatenated L2 is not sufficient when one component fails.
        self.assertFalse(acceptance(np.array([.1,1.02]),np.zeros(2),c)[0])
        # Zero threshold stays zero, matching upstream's strict predicate.
        z=SearchConfig(method='collision_plus_cpa',k_width=1,kv_thresholds=(0.,1.))
        self.assertFalse(acceptance(np.zeros(2),np.zeros(2),z)[0])

    def test_retained_commit_does_not_recompute_and_epsilon_is_unused(self):
        b=RetainedFake([[1.,1.],[9.,9.]])
        b.commit=lambda token: (_ for _ in ()).throw(AssertionError('decode commit forbidden'))
        c=SearchConfig(method='collision_plus_cpa',batch_size=2,epsilon=.001,k_width=1,kv_thresholds=(2.,2.))
        with tempfile.TemporaryDirectory() as tmp:
            result=reconstruct(CandidateBatchBackend(b,2),np.zeros((2,2)),[1,2],[0],c,tmp)
            self.assertEqual(result['status'],'complete');self.assertEqual(result['recovered_token_ids'],[0,0])

    def test_pruning_includes_final_logical_batch_like_upstream(self):
        b=CandidateBatchBackend(RetainedFake(np.full((10,2),9.)),4)
        c=SearchConfig(method='collision_plus_cpa',batch_size=4,k_width=1,kv_thresholds=(1.,1.),stop_partition=3)
        r=recover_position(b,np.zeros(2),np.arange(10),0,c)
        self.assertEqual(r['queried_token_ids'],[0,1,2,3])

    def test_calibration_resume_component_statistics_and_identity(self):
        b=RetainedFake([[0.,0.],[2.,3.],[5.,7.]])
        identity=dict(model_sha256='m',layer_index=4,observation_sha256='calibration')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            asset=collect(b,[[0.,0.],[2.,3.]],[1,2],[2],[0,1],np.arange(3),1,p,identity,chunk_size=2)
            self.assertEqual(asset['statistics']['target_count'],2)
            self.assertEqual(asset['statistics']['others_count'],4)
            self.assertEqual(asset['statistics']['others_min'],[2.,3.])
            with patch.object(b,'query',side_effect=AssertionError('resume must reuse completed positions')):
                # Fake commit consults table directly; no model query here.
                again=collect(b,[[0.,0.],[2.,3.]],[1,2],[2],[0,1],np.arange(3),1,p,identity,chunk_size=2)
            self.assertEqual(again['fit'],asset['fit'])
            loaded=load_calibration(p/'calibration.json',dict(model_sha256='m'),'independent')
            self.assertEqual(loaded['fit'],asset['fit'])
            with self.assertRaisesRegex(ValueError,'independent'):
                load_calibration(p/'calibration.json',{},'calibration')
            with self.assertRaisesRegex(ValueError,'mismatch'):
                load_calibration(p/'calibration.json',dict(model_sha256='wrong'),'independent')
            asset['coverage']['full_vocabulary']=False;dump(p/'calibration.json',asset)
            with self.assertRaisesRegex(ValueError,'pilot'):
                load_calibration(p/'calibration.json',{},'independent')

    def test_float32_per_component_metric(self):
        d=component_distances([[3.,4.,12.]],[0.,0.,0.],2)
        np.testing.assert_array_equal(d,[[5.,12.]])
        self.assertEqual(d.dtype,np.float32)

    def test_paper_defaults_and_explicit_overrides(self):
        p=build_parser()
        for method,size in [('collision_plus_cpa',256),('collision_plus',128),('guided',128)]:
            a=p.parse_args(['run','--observation','obs','--output','out','--method',method]);resolve_search_defaults(a)
            self.assertEqual(a.batch_size,size)
        a=p.parse_args(['run','--observation','obs','--output','out','--method','collision_plus_cpa','--batch-size','128']);resolve_search_defaults(a)
        self.assertEqual(a.batch_size,128)
        a=p.parse_args(['calibrate-cpa','--observation','obs','--output','out','--known-tokens-json','ids'])
        self.assertEqual(a.rank,8);self.assertIsNone(a.pilot_candidates)


if __name__=='__main__':unittest.main()
