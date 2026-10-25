import json
from pathlib import Path
import sys
import tempfile
import unittest
from dataclasses import replace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.media.image_trust import TrustConfig,fit_local,trust_step
from kvrecon.media.image import recover_trust_region
from kvrecon.search import SearchConfig

class ImageTrustTests(unittest.TestCase):
    def test_constrained_step_and_ridge_linear_recovery(self):
        step=trust_step(np.eye(2),np.array([3.,4.]),1.,.001)
        self.assertAlmostEqual(np.linalg.norm(step),1.)
        with tempfile.TemporaryDirectory() as d:
            z,result=fit_local(lambda x:x@np.diag([2.,1.]),np.array([4.,2.]),np.zeros(2),
                TrustConfig(directions=2,max_queries=50,radius=4.,epsilon=.01),Path(d)/'fit')
            self.assertTrue(result['passes_query'])
            np.testing.assert_allclose(z,[2,2],atol=.01)
            self.assertLessEqual(result['queries'],50)

    def test_budget_and_no_false_quantized_acceptance(self):
        with tempfile.TemporaryDirectory() as d:
            z,r=fit_local(lambda x:np.zeros((len(x),4)),np.ones(4),np.zeros(8),
                TrustConfig(directions=3,max_queries=18),Path(d)/'fit')
            self.assertFalse(r['passes_query']);self.assertLessEqual(r['queries'],18)
            self.assertEqual(r['rmse'],1.)
            trace=json.loads((Path(d)/'fit/optimization.json').read_text())
            self.assertTrue(all(x['ratio'] is None for x in trace))
            self.assertTrue(all(x['next_radius']<=x['radius'] for x in trace))

    def test_nonlinear_fit_is_measured_and_deterministic(self):
        f=lambda rows:np.column_stack((rows[:,0]**2,rows[:,1]+.1*rows[:,0]**2))
        with tempfile.TemporaryDirectory() as d:
            cfg=TrustConfig(directions=2,max_queries=101,radius=.5,epsilon=.01)
            a,ra=fit_local(f,np.array([1.,2.]),np.array([.5,0.]),cfg,Path(d)/'a')
            b,rb=fit_local(f,np.array([1.,2.]),np.array([.5,0.]),cfg,Path(d)/'b')
            np.testing.assert_array_equal(a,b)
            self.assertTrue(ra['passes_query'])
            self.assertAlmostEqual(ra['rmse'],float(np.sqrt(np.mean((f(a[None])[0]-[1,2])**2))))

    def test_sequential_commit_gate_and_no_commit_on_failed_query(self):
        class Backend:
            batch_capacity=128
            position=5
            commits=0
            def query_vision(self,rows):return np.zeros((len(rows),2),dtype=np.int8)
            def project_vision(self,rows):return np.zeros((len(rows),1536),dtype=np.float32)
            def commit_embeddings(self,rows):
                self.position+=len(rows);self.commits+=1
                return np.ones((len(rows),2),dtype=np.int8)
        with tempfile.TemporaryDirectory() as d:
            b=Backend()
            cfg=SearchConfig(distance_space='lsb',epsilon=.125)
            req=dict(modality='image',positions=np.array([5,6]),targets=np.zeros((2,2)),config=cfg,output=Path(d)/'first')
            result=recover_trust_region(b,req,TrustConfig(max_queries=4))
            self.assertEqual(result['accepted_positions'],0);self.assertEqual(b.commits,1)
            self.assertEqual(len(result['runs']),1)
            b=Backend();req.update(targets=np.ones((2,2)),output=Path(d)/'second')
            result=recover_trust_region(b,req,TrustConfig(max_queries=4))
            self.assertEqual(b.commits,0);self.assertEqual(result['status'],'unresolved')

if __name__=='__main__':unittest.main()
