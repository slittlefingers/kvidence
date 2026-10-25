import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvrecon.media.image_geometry import orthogonal_directions, centered_steps

class ImageGeometryTests(unittest.TestCase):
    def test_linear_step_reduces_measured_error(self):
        delta=orthogonal_directions(np.random.default_rng(17),4,4)
        matrix=np.diag([1.,2.,3.,4.])
        center=np.zeros(4);target=np.array([.02,-.01,.03,.01]);c=.2
        rows=np.concatenate([center+c*delta,center-c*delta])
        values=np.einsum('ij,jk->ik',rows,matrix,optimize=False)
        steps=centered_steps(delta,values,center,target,c)
        errors=np.linalg.norm(np.einsum('ij,jk->ik',steps,matrix,optimize=False)-target,axis=1)
        self.assertLess(errors.min(),np.linalg.norm(target)*.01)
        self.assertLessEqual(float(np.sqrt(np.mean(steps**2,axis=1)).max()),2*c+1e-7)

    def test_nonlinear_midpoint_does_not_reverse_step(self):
        # f(x)=x² at x=1: center 1, paired midpoint 1.25, target 1.2.
        # A midpoint residual incorrectly asks for a negative displacement.
        steps=centered_steps(np.ones((1,1)),np.array([[2.25],[.25]]),
            np.array([1.]),np.array([1.2]),.5)
        self.assertTrue(np.all(steps>0))
        self.assertLess(np.min(np.abs((1+steps[:,0])**2-1.2)),.02)

    def test_quantized_plateau_and_invalid_geometry(self):
        delta=orthogonal_directions(np.random.default_rng(29),3,8)
        steps=centered_steps(delta,np.zeros((6,4)),np.zeros(4),np.ones(4),.1)
        np.testing.assert_array_equal(steps,np.zeros((16,8)))
        with self.assertRaises(ValueError):
            centered_steps(np.ones((2,8)),np.zeros((4,4)),np.zeros(4),np.ones(4),.1)

if __name__=='__main__':unittest.main()
