"""Dimension normalisation, strict gates, and measured commit validation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.search import SearchConfig, kv_distance, acceptance
from kvrecon.engine import reconstruct
from kvrecon.first_v import nearest_v
from test_lsb import IntegerBackend


class RmseTests(unittest.TestCase):
    def test_dimension_invariance_and_widening(self):
        for size in (2,32,1024):
            self.assertEqual(float(kv_distance(np.full(size,127,dtype=np.int8),np.full(size,-128,dtype=np.int8))),255.)
        self.assertEqual(float(kv_distance([3.,4.],[0.,0.],'l2')),5.)
        self.assertAlmostEqual(float(kv_distance([3.,4.],[0.,0.])),np.sqrt(12.5))

    def test_1024_coordinate_old_gate_equivalence(self):
        values=np.zeros((3,1024));values[:,0]=[3,4,5]
        new=SearchConfig();old=SearchConfig(epsilon=4,distance_metric='l2')
        new_pass,new_d=acceptance(values,np.zeros(1024),new)
        old_pass,old_d=acceptance(values,np.zeros(1024),old)
        np.testing.assert_array_equal(new_pass,[True,False,False])
        np.testing.assert_array_equal(new_pass,old_pass)
        np.testing.assert_allclose(new_d,old_d/32)

    def test_commit_and_telemetry_use_rmse_not_l2(self):
        class Drift(IntegerBackend):
            def commit(self,token):
                super().commit(token)
                return np.ones((1,4),dtype=np.int8)
        with tempfile.TemporaryDirectory() as d:
            result=reconstruct(Drift([[0,0,0,0]]),np.zeros((1,4)),[1],[2],
                SearchConfig(method='collision_plus',epsilon=1.1),d,live_events=True)
            row=json.loads((Path(d)/'trace.jsonl').read_text())
            self.assertEqual(result['status'],'complete')
            self.assertEqual(row['commit_distance'],1.)
            self.assertEqual(row['commit_l2'],2.)
            events=[json.loads(x) for x in (Path(d)/'events.jsonl').read_text().splitlines()]
            end=next(x for x in events if x['type']=='position_end')
            self.assertEqual(end['distance_metric'],'rmse')
            self.assertEqual(end['commit_distance'],1.)
        with tempfile.TemporaryDirectory() as d:
            result=reconstruct(Drift([[0,0,0,0]]),np.zeros((1,4)),[1],[2],
                SearchConfig(method='collision_plus',epsilon=1.),d)
            self.assertEqual(result['status'],'commit_mismatch')

    def test_first_v_uses_v_dimensions_only(self):
        q=[dict(width=4,scale=.01,zero_point=0,dtype='int8')]
        row=nearest_v(np.ones((1,4),dtype=np.int8),np.array([3]),q,np.zeros(4),1.)
        self.assertEqual(row['minimum_distance'],1.)
        self.assertEqual(row['minimum_l2'],2.)
        self.assertIsNone(row['accepted_token_id'])
        row=nearest_v(np.ones((1,4),dtype=np.int8),np.array([3]),q,np.zeros(4),1.01)
        self.assertEqual(row['accepted_token_id'],3)

    def test_cpa_keeps_separate_l2(self):
        config=SearchConfig(method='collision_plus_cpa',k_width=2,kv_thresholds=(1.1,1.1))
        passed,distances=acceptance([1,1,1,1],np.zeros(4),config)
        self.assertFalse(passed)
        np.testing.assert_allclose(distances,[np.sqrt(2),np.sqrt(2)])

if __name__=='__main__':unittest.main()
