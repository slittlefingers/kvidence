"""Tiny fixed-tanh residual surrogate with a linear skip and dual ridge fit.

Fits only measured local queries. No neural-network framework or backpropagation
through the phone is needed. Input/output centers are measured, not inferred.
"""
import time
import numpy as np


def mm(a, b):
    return np.einsum('ij,jk->ik', a, b, optimize=False)


class LocalSurrogate:
    def __init__(self, hidden=64, nonlinear=1., ridge=.1, seed=217):
        if type(hidden) is not int or hidden<1 or not np.isfinite(nonlinear) or nonlinear<0 or not np.isfinite(ridge) or ridge<=0:
            raise ValueError('Invalid surrogate configuration')
        self.hidden=hidden;self.nonlinear=nonlinear;self.ridge=ridge;self.seed=seed

    def fit(self, inputs, values, center, center_kv, radius):
        start=time.perf_counter()
        x=np.asarray(inputs,dtype=np.float64);y=np.asarray(values,dtype=np.float64)
        self.center=np.asarray(center,dtype=np.float64).copy()
        self.center_kv=np.asarray(center_kv,dtype=np.float64).copy()
        if (x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or not len(x)
                or self.center.shape!=(x.shape[1],) or self.center_kv.shape!=(y.shape[1],)
                or not x.shape[1] or not y.shape[1] or not np.isfinite(radius) or radius<=0
                or not all(np.isfinite(v).all() for v in (x,y,self.center,self.center_kv))):
            raise ValueError('Expected aligned finite local queries and measured center')
        self.radius=float(radius);self.width=x.shape[1]
        self.x=(x-self.center)/self.radius
        rng=np.random.default_rng(self.seed)
        self.weights=rng.standard_normal((self.width,self.hidden))/np.sqrt(self.width)
        self.bias=rng.uniform(-1.,1.,self.hidden)
        self.h=self._hidden(self.x)
        gram=mm(self.x,self.x.T)/self.width+self.nonlinear*mm(self.h,self.h.T)/self.hidden
        # Scaling by mean kernel diagonal makes ridge comparable across widths.
        regularizer=self.ridge*max(float(np.trace(gram)/len(gram)),1e-8)
        self.alpha=np.linalg.solve(gram+regularizer*np.eye(len(x)),y-self.center_kv)
        if not np.isfinite(self.alpha).all():raise ValueError('Nonfinite fitted surrogate')
        self.fit_seconds=time.perf_counter()-start
        return self

    def _hidden(self,x):
        return np.tanh(mm(x,self.weights)+self.bias)-np.tanh(self.bias)

    def predict(self,rows):
        rows=np.asarray(rows,dtype=np.float64)
        if rows.ndim!=2 or rows.shape[1]!=self.width or not np.isfinite(rows).all():
            raise ValueError('Expected finite candidate rows')
        x=(rows-self.center)/self.radius
        kernel=mm(x,self.x.T)/self.width+self.nonlinear*mm(self._hidden(x),self.h.T)/self.hidden
        return self.center_kv+mm(kernel,self.alpha)

    def loss_and_grad(self,rows,target):
        rows=np.asarray(rows,dtype=np.float64);target=np.asarray(target,dtype=np.float64)
        if rows.ndim!=2 or rows.shape[1]!=self.width or target.shape!=self.center_kv.shape or not np.isfinite(rows).all() or not np.isfinite(target).all():
            raise ValueError('Expected aligned finite candidates and target')
        x=(rows-self.center)/self.radius
        activation=np.tanh(mm(x,self.weights)+self.bias)
        h=activation-np.tanh(self.bias)
        kernel=mm(x,self.x.T)/self.width+self.nonlinear*mm(h,self.h.T)/self.hidden
        residual=self.center_kv+mm(kernel,self.alpha)-target
        loss=.5*np.mean(residual**2,axis=1)
        dual=mm(residual,self.alpha.T)/len(target)
        grad=mm(dual,self.x)/self.width
        hidden_grad=self.nonlinear*mm(dual,self.h)/self.hidden
        grad+=mm(hidden_grad*(1-activation**2),self.weights.T)
        return loss,grad/self.radius

    def proposals(self,target,starts,max_rms=.2,iterations=12):
        """Optimize surrogate cheaply, constrain every branch around fitted center.

        These proposals have no acceptance status until measured by real NPU.
        """
        rows=np.asarray(starts,dtype=np.float64).copy()
        if not np.isfinite(max_rms) or max_rms<=0 or type(iterations) is not int or iterations<1:
            raise ValueError('Invalid surrogate proposal budget')
        def project(a):
            dx=a-self.center
            rms=np.sqrt(np.mean(dx**2,axis=1,keepdims=True))
            return self.center+dx*np.minimum(1.,max_rms/np.maximum(rms,1e-12))
        rows=project(rows);best=rows.copy();best_loss,_=self.loss_and_grad(rows,target)
        m=np.zeros_like(rows);v=m.copy()
        for step in range(1,iterations+1):
            _,g=self.loss_and_grad(rows,target)
            m=.9*m+.1*g;v=.999*v+.001*g*g
            rows=project(rows-(max_rms/8)*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8))
            loss,_=self.loss_and_grad(rows,target)
            better=loss<best_loss;best[better]=rows[better];best_loss=np.minimum(best_loss,loss)
        return best.astype(np.float32)
