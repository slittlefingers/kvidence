"""Bounded, measured-improvement-only scale expansion after stagnation."""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass
class ExpansionPolicy:
    initial_radius: float
    patience: int = 1
    factor: float = 2.
    trial_rounds: int = 1
    max_multiplier: float = 4.
    radius: float = 0.
    good_radius: float = 0.
    stale: int = 0
    trial_age: int = 0
    trial: bool = False
    stopped: bool = False
    reason: str = ''

    def __post_init__(self):
        if (not np.isfinite(self.initial_radius) or self.initial_radius<=0 or self.patience<1
                or self.trial_rounds<1 or not np.isfinite(self.factor) or self.factor<=1
                or not np.isfinite(self.max_multiplier) or self.max_multiplier<1):
            raise ValueError('Invalid expansion policy')
        if self.radius==0:self.radius=self.initial_radius
        if self.good_radius==0:self.good_radius=self.radius

    def state(self):return asdict(self)

    def update(self,improved):
        if self.stopped:return dict(action='stopped',reason=self.reason)
        if improved:
            action='accept_expansion' if self.trial else 'improve'
            self.good_radius=self.radius;self.stale=0;self.trial_age=0;self.trial=False
            return dict(action=action,radius=self.radius)
        self.stale+=1
        cap=self.initial_radius*self.max_multiplier
        if self.trial:
            self.trial_age+=1
            if self.trial_age<self.trial_rounds:return dict(action='trial',radius=self.radius)
            rejected=self.radius;self.radius=self.good_radius
            if rejected>=cap*(1-1e-12):
                self.stopped=True;self.reason='expansion_exhausted';self.trial=False
                return dict(action='reject_and_stop',rejected=rejected,restored=self.radius)
            self.radius=min(cap,rejected*self.factor);self.trial_age=0
            return dict(action='reject_and_expand',rejected=rejected,restored=self.good_radius,radius=self.radius)
        if self.stale<self.patience:return dict(action='continue',radius=self.radius)
        if self.radius>=cap*(1-1e-12):
            self.stopped=True;self.reason='max_scale_plateau'
            return dict(action='stop_at_cap',radius=self.radius)
        self.radius=min(cap,self.radius*self.factor);self.trial=True;self.trial_age=0
        return dict(action='expand',previous=self.good_radius,radius=self.radius)
