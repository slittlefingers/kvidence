import sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.media.image_mlp import TinyMLPSurrogate
from kvrecon.media.image_surrogate import LocalSurrogate

class ImageMLPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_learned_nonlinearity_generalizes_and_warm_fit_trains_hidden(self):
        rng=np.random.default_rng(7);x=rng.uniform(-1,1,(128,4)).astype(np.float32)
        f=lambda a:np.column_stack((a[:,0]**2,a[:,1]**2,a[:,2]*a[:,3])).astype(np.float32)
        y=f(x);xt=rng.uniform(-.8,.8,(40,4)).astype(np.float32)
        mlp=TinyMLPSurrogate(steps=300,warm_steps=20).fit(x,y,np.zeros(4),np.zeros(3),1.)
        linear=LocalSurrogate(nonlinear=0.).fit(x,y,np.zeros(4),np.zeros(3),1.)
        self.assertLess(np.mean((mlp.predict(xt)-f(xt))**2),np.mean((linear.predict(xt)-f(xt))**2)*.5)
        np.testing.assert_allclose(mlp.predict(np.zeros((1,4))),0,atol=1e-7)
        before=mlp.net[0].weight.detach().clone();mlp.fit(x,y,np.zeros(4),np.zeros(3),1.)
        self.assertFalse(torch.equal(before,mlp.net[0].weight));self.assertEqual(mlp.fits,2)

    def test_gradient_matches_finite_difference_and_proposals_are_bounded(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(32,4)).astype(np.float32);y=rng.normal(size=(32,3)).astype(np.float32)
        mlp=TinyMLPSurrogate(steps=20).fit(x,y,np.zeros(4),np.zeros(3),.5)
        point=np.array([[.05,.1,-.1,.15]],dtype=np.float32);target=np.ones(3)
        _,grad=mlp.loss_and_grad(point,target);fd=np.zeros_like(grad)
        for k in range(4):
            e=np.zeros_like(point);e[0,k]=.001
            fd[0,k]=(mlp.loss_and_grad(point+e,target)[0][0]-mlp.loss_and_grad(point-e,target)[0][0])/.002
        np.testing.assert_allclose(grad,fd,atol=.003,rtol=.003)
        candidates=mlp.proposals(target,np.zeros((4,4)),max_rms=.1,iterations=5)
        self.assertTrue(np.isfinite(candidates).all());self.assertLessEqual(np.sqrt(np.mean(candidates**2,axis=1)).max(),.100001)

if __name__=='__main__':unittest.main()
