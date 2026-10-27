"""ALS with a graded and time-aware target; confidence is unchanged."""
import numpy as np
from scipy import sparse
from .ials import IALS


def custom_targets(history,shape,cutoff,half_life_days=180):
    age=(int(cutoff)-history.timestamp.to_numpy())/86400
    if np.any(age<0):raise ValueError('Future event in custom-target history')
    grade=np.clip((history.rating.to_numpy()-3)/2,0,1)
    if half_life_days is None:decay=np.ones(len(history))
    else:
        if half_life_days<=0:raise ValueError('Half-life must be positive')
        decay=.5+.5*np.exp2(-age/half_life_days)
    return sparse.csr_matrix((grade*decay,(history.u,history.i)),shape=shape)


class CustomALS(IALS):
    def __init__(self,half_life_days=180,**kwargs):
        super().__init__(**kwargs);self.half_life_days=half_life_days
    def target_matrix(self,history,shape,cutoff):
        return custom_targets(history,shape,cutoff,self.half_life_days)
