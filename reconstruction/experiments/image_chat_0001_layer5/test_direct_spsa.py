import tempfile
from pathlib import Path
import unittest
import numpy as np
from spsa_adam import Config
from direct_spsa import fit_direct


class DirectSpsaTests(unittest.TestCase):
    def test_free_vector_can_reach_non_unit_target(self):
        target=np.array([3.,-2.,4.,1.])
        initial=np.random.default_rng(1).standard_normal((8,4)).astype(np.float32)
        calls=[]
        def oracle(z):
            calls.append(len(z))
            return z.copy()
        with tempfile.TemporaryDirectory() as tmp:
            best,result=fit_direct(oracle,target,initial,
                Config(dimension=4,initial_candidates=8,directions=32,steps=500,
                       lr=.1,acceptance_l2=.05),Path(tmp)/'run')
        self.assertTrue(result['passes_query'])
        self.assertLess(np.linalg.norm(best-target),.05)
        self.assertGreater(np.sqrt(np.mean(best**2)),2.)
        self.assertEqual(result['queries'],sum(calls))
        self.assertEqual(result['oracle_calls'],len(calls))

    def test_constant_oracle_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,result=fit_direct(lambda z:np.zeros((len(z),2)),np.ones(2),
                np.ones((4,3)),Config(dimension=3,initial_candidates=4,directions=2,
                    steps=3,acceptance_l2=.1),Path(tmp)/'run')
        self.assertFalse(result['passes_query'])
        self.assertEqual(result['status'],'iteration_limit')


if __name__=='__main__':unittest.main()
