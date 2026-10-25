"""Trainable tiny SiLU correction to a local linear KV surrogate (CPU PyTorch).

Inspired by learned nonlinear embedding maps, not a MindEye reproduction.
Only public model queries are fitted; all proposed inputs require NPU validation.
"""
import time
import numpy as np
import torch
from torch import nn
from .image_surrogate import LocalSurrogate


class TinyMLPSurrogate(LocalSurrogate):
    def __init__(self,steps=100,warm_steps=20,lr=.003,seed=217):
        if min(steps,warm_steps)<1 or not np.isfinite(lr) or lr<=0:raise ValueError('Invalid MLP training budget')
        self.steps=steps;self.warm_steps=warm_steps;self.lr=lr;self.seed=seed;self.net=None;self.fits=0
        self.generator=torch.Generator(device='cpu').manual_seed(seed)

    def fit(self,inputs,values,center,center_kv,radius):
        started=time.perf_counter()
        self.base=LocalSurrogate(nonlinear=0.,ridge=.1,seed=self.seed).fit(inputs,values,center,center_kv,radius)
        self.center=self.base.center;self.center_kv=self.base.center_kv;self.radius=self.base.radius;self.width=self.base.width
        y=np.asarray(values,dtype=np.float32);x=np.asarray(self.base.x,dtype=np.float32)
        out=y.shape[1];cold=self.net is None
        if cold:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed)
                self.net=nn.Sequential(nn.Linear(self.width,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,out,bias=False))
                nn.init.zeros_(self.net[-1].weight)
            self.optimizer=torch.optim.AdamW(self.net.parameters(),lr=self.lr,weight_decay=1e-4)
        elif self.net[0].in_features!=self.width or self.net[-1].out_features!=out:
            raise ValueError('Warm-fit dimensions changed')
        self.parameter_count=sum(p.numel() for p in self.net.parameters())
        self.center_t=torch.tensor(self.center,dtype=torch.float32)
        self.center_q_t=torch.tensor(self.center_kv,dtype=torch.float32)
        self.linear=(torch.tensor(self.base.x,dtype=torch.float32).T@torch.tensor(self.base.alpha,dtype=torch.float32)/self.width).detach()
        self.zero=torch.zeros((1,self.width),dtype=torch.float32)
        self.output_scale=max(float(np.sqrt(np.mean((y-self.center_kv)**2))),1.)
        xt=torch.from_numpy(x);yt=torch.from_numpy(y)
        with torch.no_grad():residual=(yt-self.center_q_t-xt@self.linear)/self.output_scale
        if not torch.isfinite(residual).all() or not torch.isfinite(self.linear).all():
            raise ValueError('Nonfinite MLP training data or frozen linear map')
        budget=self.steps if cold else self.warm_steps
        self.net.train();tick=time.perf_counter()
        for _ in range(budget):
            ids=torch.randint(len(x),(min(64,len(x)),),generator=self.generator)
            correction=self.net(xt[ids])-self.net(self.zero)
            loss=.5*torch.mean((correction-residual[ids])**2)
            if not torch.isfinite(loss):raise ValueError('Nonfinite MLP training loss')
            self.optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.,error_if_nonfinite=True)
            self.optimizer.step()
        self.net.eval();self.train_seconds=time.perf_counter()-tick;self.fits+=1
        with torch.no_grad():
            pred=self._forward(xt)
            if not torch.isfinite(pred).all():raise ValueError('Nonfinite MLP post-training prediction')
            self.train_rmse=float(torch.sqrt(torch.mean((pred-yt)**2)))
        self.fit_seconds=time.perf_counter()-started
        return self

    def _forward(self,normalized):
        return self.center_q_t+normalized@self.linear+self.output_scale*(self.net(normalized)-self.net(self.zero))

    def predict(self,rows):
        rows=np.asarray(rows,dtype=np.float32)
        if rows.ndim!=2 or rows.shape[1]!=self.width or not np.isfinite(rows).all():raise ValueError('Invalid MLP candidate rows')
        with torch.no_grad():return self._forward((torch.from_numpy(rows)-self.center_t)/self.radius).numpy().astype(np.float64)

    def loss_and_grad(self,rows,target):
        rows=np.asarray(rows,dtype=np.float32);target=np.asarray(target,dtype=np.float32)
        if rows.ndim!=2 or rows.shape[1]!=self.width or target.shape!=self.center_kv.shape or not np.isfinite(rows).all() or not np.isfinite(target).all():
            raise ValueError('Invalid MLP gradient arguments')
        inputs=torch.tensor(rows,requires_grad=True)
        residual=self._forward((inputs-self.center_t)/self.radius)-torch.from_numpy(target)
        loss=.5*torch.mean(residual**2,dim=1)
        gradient=torch.autograd.grad(loss.sum(),inputs)[0]
        return loss.detach().numpy().astype(np.float64),gradient.detach().numpy().astype(np.float64)
