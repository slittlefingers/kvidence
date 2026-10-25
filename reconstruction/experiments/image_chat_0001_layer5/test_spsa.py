import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from spsa_adam import Config,estimate_gradient,fit,unit


class Tests(unittest.TestCase):
    def test_paired_estimator_exact_for_orthogonal_quadratic(self):
        delta=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1.]])
        u=np.array([.1,.2,-.4,.8]);target=np.array([.7,-.8,.1,.3]);c=.05
        loss=lambda x:.5*np.mean((x-target)**2,axis=-1)
        g=estimate_gradient(loss(u+c*delta),loss(u-c*delta),delta,c)
        np.testing.assert_allclose(g,(u-target)/4,atol=1e-12)

    def test_blackbox_reachable_sphere(self):
        target=unit(np.random.default_rng(999).standard_normal(16)).astype(float)
        seen=[]
        def oracle(rows):seen.append(len(rows));return unit(rows).astype(float)
        with tempfile.TemporaryDirectory() as temp:
            _,r=fit(oracle,target,Config(dimension=16,steps=400,acceptance_l2=.15,lr=.1),Path(temp)/'run')
            self.assertTrue(r['passes_query']);self.assertLess(r['best_l2'],r['initial_l2'])
            self.assertEqual(sum(seen),r['queries']);self.assertLessEqual(max(seen),128)
            self.assertEqual(len(seen),r['oracle_calls'])

    def test_flat_oracle_does_not_false_accept(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'run'
            _,r=fit(lambda rows:np.zeros((len(rows),4)),np.ones(4),
                Config(dimension=4,directions=2,initial_candidates=4,steps=3,acceptance_l2=.1),out)
            self.assertEqual(r['status'],'iteration_limit');self.assertFalse(r['passes_query'])
            trace=[json.loads(line) for line in (out/'trace.jsonl').read_text().splitlines()]
            self.assertTrue(all(x['zero_slope_fraction']==1 and x['gradient_norm']==0 for x in trace[1:]))


if __name__=='__main__':unittest.main()
