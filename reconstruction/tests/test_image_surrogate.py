import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.media.image_surrogate import LocalSurrogate

class ImageSurrogateTests(unittest.TestCase):
    def test_nonlinear_holdout_beats_linear_and_center_is_anchored(self):
        x=np.linspace(-1,1,40)[:,None];y=x*x
        xt=np.linspace(-.95,.95,31)[:,None]
        models=[LocalSurrogate(nonlinear=a,ridge=.001).fit(x,y,np.zeros(1),np.zeros(1),1.) for a in (0.,1.)]
        errors=[np.mean((m.predict(xt)-xt*xt)**2) for m in models]
        self.assertLess(errors[1],errors[0]*.1)
        np.testing.assert_array_equal(models[1].predict(np.zeros((1,1))),[[0.]])

    def test_analytic_surrogate_gradient_and_proposal_radius(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(32,4));y=rng.normal(size=(32,6))
        model=LocalSurrogate(hidden=16).fit(x,y,np.zeros(4),np.zeros(6),.5)
        row=rng.normal(size=(1,4))*.1;target=np.ones(6)
        _,gradient=model.loss_and_grad(row,target)
        for i in range(4):
            e=np.zeros((1,4));e[0,i]=1e-5
            a=model.loss_and_grad(row+e,target)[0][0];b=model.loss_and_grad(row-e,target)[0][0]
            self.assertAlmostEqual(gradient[0,i],(a-b)/2e-5,places=6)
        starts=np.zeros((4,4));proposals=model.proposals(target,starts,max_rms=.2,iterations=8)
        self.assertTrue(np.isfinite(proposals).all())
        self.assertLessEqual(np.sqrt(np.mean(proposals**2,axis=1)).max(),.200001)
        self.assertTrue(np.all(model.loss_and_grad(proposals,target)[0]<=model.loss_and_grad(starts,target)[0]+1e-6))

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):LocalSurrogate(ridge=0)
        with self.assertRaises(ValueError):LocalSurrogate().fit(np.zeros((2,3)),np.ones((1,2)),np.zeros(3),np.zeros(2),1.)

if __name__=='__main__':unittest.main()
