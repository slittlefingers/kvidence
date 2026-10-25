"""Stable full-vocabulary ridge scores, using a QR residual factorization.

Same regression objective and anchor policy as the prior fast implementation.
Avoid the cancellation of g(BB^T)g^T - 2g(Bt) + t^Tt in float32. For B^T=QR:
||gB-t||^2 = ||gR^T-tQ||^2 + ||t-(tQ)Q^T||^2.
"""
import numpy as np
import torch

MAX_ANCHORS=512


class StableSurrogate:
    def __init__(self,rt,embeddings):
        self.rt,self.embeddings=rt,embeddings
        self.e=rt.mx.array(embeddings.contiguous().numpy())
        rt.mx.eval(self.e)
        self.stats=dict(fit_calls=0,anchor_capped_fit_calls=0,maximum_training_anchors=0)

    def fit(self,queried,measured,target,distances,geometry=False):
        assert not geometry, 'This benchmark does not use adaptive trust regions'
        q=np.asarray(queried)
        keep=np.arange(len(q))
        if len(keep)>MAX_ANCHORS:
            nearest=np.argsort(distances,kind='stable')[:MAX_ANCHORS//2]
            rest=np.setdiff1d(keep,nearest)
            diverse=rest[np.linspace(0,len(rest)-1,MAX_ANCHORS-len(nearest),dtype=int)]
            keep=np.r_[nearest,diverse]
            self.stats['anchor_capped_fit_calls']+=1
        self.stats['fit_calls']+=1
        self.stats['maximum_training_anchors']=max(self.stats['maximum_training_anchors'],len(keep))
        x=self.embeddings[torch.tensor(q[keep])]
        mean,scale=x.mean(0),x.std(0).clamp_min(1e-4)
        x=(x-mean)/scale
        y=measured[torch.tensor(keep)]
        ym,ys=y.mean(0),y.std(0).clamp_min(1e-4)
        y=(y-ym)/ys
        xm,ymean=x.mean(0),y.mean(0)
        xc,yc=x-xm,y-ymean
        system=xc@xc.T+torch.eye(len(x))*(.01*len(x))
        dual=torch.linalg.solve(system,yc)
        # QR in float64; the full-table matrix operations remain GPU float32.
        b=dual.double()*ys.double()
        t=target.double()-(ym.double()+ymean.double()*ys.double())
        orthogonal,upper=torch.linalg.qr(b.T,mode='reduced')
        projected=t@orthogonal
        orthogonal_residual=t-projected@orthogonal.T
        constant=float(orthogonal_residual.square().sum())
        mx=self.rt.mx
        gpu=lambda a:mx.array(a.float().contiguous().numpy())
        g=self.e@gpu((xc/scale).T)-gpu((mean/scale+xm)@xc.T)
        residual=g@gpu(upper.T)-gpu(projected)
        scores=mx.sum(residual.square(),axis=-1)+constant
        mx.eval(scores)
        result=np.array(scores)
        assert np.isfinite(result).all()
        return dict(scores=result,anchors=len(x),
            reference_factors=(mean,scale,xm,xc,dual,ymean,ys,ym))
