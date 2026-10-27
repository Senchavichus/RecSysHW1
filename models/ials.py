"""Exact implicit ALS with independent targets and confidence weights."""
from time import perf_counter
import numpy as np
from scipy import sparse
from utils.data import history_matrices


def als_half_step(observed,targets,fixed,reg=1.,alpha=10.):
    observed=sparse.csr_matrix(observed,dtype=float)
    targets=sparse.csr_matrix(targets,dtype=float)
    if reg<=0 or alpha<0:raise ValueError('Require reg>0 and alpha>=0')
    base=fixed.T@fixed+reg*np.eye(fixed.shape[1])
    rhs=(1+alpha)*(targets@fixed)
    result=np.zeros((observed.shape[0],fixed.shape[1]))
    for row in np.flatnonzero(np.diff(observed.indptr)):
        ids=observed.indices[observed.indptr[row]:observed.indptr[row+1]]
        local=fixed[ids]
        result[row]=np.linalg.solve(base+alpha*(local.T@local),rhs[row])
    return result


def als_loss(observed,targets,X,Y,reg,alpha):
    coo=observed.tocoo()
    scores=np.einsum('ij,ij->i',X[coo.row],Y[coo.col])
    q=np.asarray(targets[coo.row,coo.col]).ravel()
    background=np.sum((X.T@X)*(Y.T@Y))
    return float(background+np.sum((1+alpha)*(q-scores)**2-scores**2)
                 +reg*(np.sum(X**2)+np.sum(Y**2)))


class IALS:
    def __init__(self,factors=24,reg=1.,alpha=10.,seed=42):
        if factors<1 or reg<=0 or alpha<0:raise ValueError('Invalid ALS parameters')
        self.factors=factors;self.reg=reg;self.alpha=alpha;self.seed=seed

    def fit(self,observed,targets,epochs=15,callback=None):
        self.observed=sparse.csr_matrix(observed,dtype=float)
        self.targets=sparse.csr_matrix(targets,dtype=float)
        if self.observed.shape!=self.targets.shape:raise ValueError('Target/observation shapes differ')
        if np.any(self.observed.data!=1):raise ValueError('Observed matrix must be binary')
        if (self.targets-self.targets.multiply(self.observed)).nnz:
            raise ValueError('Targets must be zero outside observed pairs')
        if not np.isfinite(self.targets.data).all():raise ValueError('Nonfinite targets')
        ot=self.observed.T.tocsr();qt=self.targets.T.tocsr()
        rng=np.random.default_rng(self.seed)
        self.X=rng.normal(0,.1,(observed.shape[0],self.factors))
        self.Y=rng.normal(0,.1,(observed.shape[1],self.factors))
        self.X[np.diff(self.observed.indptr)==0]=0
        self.Y[np.diff(ot.indptr)==0]=0
        self.fit_seconds=0.;self.history=[]
        for epoch in range(1,epochs+1):
            start=perf_counter()
            self.X=als_half_step(self.observed,self.targets,self.Y,self.reg,self.alpha)
            self.Y=als_half_step(ot,qt,self.X,self.reg,self.alpha)
            self.fit_seconds+=perf_counter()-start
            self.history.append({'epoch':epoch,'loss':als_loss(self.observed,self.targets,self.X,self.Y,self.reg,self.alpha)})
            if callback:callback(self,epoch)
        return self

    def target_matrix(self,history,shape,cutoff):
        _,binary=history_matrices(history,shape);return binary

    def fit_history(self,history,shape,cutoff,epochs=15,callback=None):
        observed,_=history_matrices(history,shape)
        return self.fit(observed,self.target_matrix(history,shape,cutoff),epochs,callback)

    def score(self,users):return self.X[np.asarray(users)]@self.Y.T

    @property
    def parameter_bytes(self):return self.X.nbytes+self.Y.nbytes
