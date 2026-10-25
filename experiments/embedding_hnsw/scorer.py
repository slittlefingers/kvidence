"""HNSW shortlist followed by the existing standardized forward ridge score.

The inverse fit is a new heuristic, not an exact rewriting of the KV objective.
After two indexed attempts, revert to full-table scoring for this position.
The fallback fit excludes indexed-only probes to avoid carrying their bias into
the global regression; their measured values still remain in the query log.
"""
from pathlib import Path
import sys
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'ciphertext_full_vocabulary'))
from stable_surrogate import StableSurrogate, MAX_ANCHORS
from index import EmbeddingIndex


class IndexedSurrogate(StableSurrogate):
    def __init__(self, rt, embeddings):
        super().__init__(rt, embeddings)
        self.index=EmbeddingIndex()
        self.round=0
        self.records=[]
        self.stats.update(index_rounds=0, full_table_rounds=0,
            inverse_seconds=0., retrieval_seconds=0., forward_fit_seconds=0.,
            candidate_scoring_seconds=0., scored_candidates=0)

    def fit(self, queried, measured, target, distances, geometry=False):
        assert not geometry
        if len(queried)==200:
            self.round=0
        self.round+=1
        self.stats['fit_calls']+=1
        q=np.asarray(queried)
        full=self.round>=3
        keep=np.r_[np.arange(200),np.arange(240,len(q))] if full else np.arange(len(q))
        if len(keep)>MAX_ANCHORS:
            nearest=keep[np.argsort(np.asarray(distances)[keep],kind='stable')[:MAX_ANCHORS//2]]
            rest=np.setdiff1d(keep,nearest)
            diverse=rest[np.linspace(0,len(rest)-1,MAX_ANCHORS-len(nearest),dtype=int)]
            keep=np.r_[nearest,diverse]
            self.stats['anchor_capped_fit_calls']+=1
        self.stats['maximum_training_anchors']=max(self.stats['maximum_training_anchors'],len(keep))
        tick=time.perf_counter()
        raw=self.embeddings[torch.tensor(q[keep])]
        mean,scale=raw.mean(0),raw.std(0).clamp_min(1e-4)
        x=(raw-mean)/scale
        y=measured[torch.tensor(keep)]
        ym,ys=y.mean(0),y.std(0).clamp_min(1e-4)
        y=(y-ym)/ys
        xm,ymean=x.mean(0),y.mean(0)
        xc,yc=x-xm,y-ymean
        system=xc@xc.T+torch.eye(len(x))*(.01*len(x))
        dual=torch.linalg.solve(system,yc)
        b=dual.double()*ys.double()
        t=target.double()-(ym.double()+ymean.double()*ys.double())
        orthogonal,upper=torch.linalg.qr(b.T,mode='reduced')
        projected=t@orthogonal
        constant=float((t-projected@orthogonal.T).square().sum())
        forward_fit=time.perf_counter()-tick
        inverse_seconds=retrieval_seconds=0.
        selected=None
        if not full:
            tick=time.perf_counter()
            # Inverse ridge: standardized measured KV -> original raw embedding.
            # Uses only queried pairs at THIS position, never target token labels.
            z=(target-ym)/ys-ymean
            weights=torch.linalg.solve(yc@yc.T+torch.eye(len(y))*(.01*len(y)),yc@z)
            estimate=mean+weights@(raw-mean)
            inverse_seconds=time.perf_counter()-tick
            tick=time.perf_counter()
            width=512 if self.round==1 else 1024
            found,_=self.index.query(estimate.numpy(),k=width,ef=2*width)
            selected=np.setdiff1d(found[0].astype(np.int64),q)
            retrieval_seconds=time.perf_counter()-tick
            if len(selected)<20:
                full=True
        tick=time.perf_counter()
        mx=self.rt.mx
        gpu=lambda a:mx.array(a.float().contiguous().numpy())
        e=self.e if full else self.e[mx.array(selected)]
        # Same numerically checked QR scoring as the full-table control.
        g=e@gpu((xc/scale).T)-gpu((mean/scale+xm)@xc.T)
        residual=g@gpu(upper.T)-gpu(projected)
        score=mx.sum(residual.square(),axis=-1)+constant
        mx.eval(score)
        values=np.array(score)
        assert np.isfinite(values).all()
        if full:
            scores=values
        else:
            scores=np.full(len(self.embeddings),np.inf,dtype=np.float32)
            scores[selected]=values
        scoring_seconds=time.perf_counter()-tick
        detail=dict(round=self.round,full_table=full,scored_candidates=len(values),
            inverse_seconds=inverse_seconds,retrieval_seconds=retrieval_seconds,
            forward_fit_seconds=forward_fit,candidate_scoring_seconds=scoring_seconds)
        self.records.append(detail)
        self.stats['full_table_rounds' if full else 'index_rounds']+=1
        for key in ('inverse_seconds','retrieval_seconds','forward_fit_seconds',
                    'candidate_scoring_seconds','scored_candidates'):
            self.stats[key]+=detail[key]
        return dict(scores=scores,anchors=len(x),
            reference_factors=(mean,scale,xm,xc,dual,ymean,ys,ym))
