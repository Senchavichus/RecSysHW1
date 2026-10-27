"""Closed-form zero-diagonal linear reconstruction, no fitted sklearn models."""
from time import perf_counter
import numpy as np
from scipy import sparse,linalg


class EASE:
    def __init__(self,reg=100.):
        if reg<=0:raise ValueError('Require positive regularization')
        self.reg=reg
    def fit(self,binary):
        start=perf_counter();self.binary=sparse.csr_matrix(binary,dtype=float)
        gram=(self.binary.T@self.binary).toarray()
        gram.flat[::len(gram)+1]+=self.reg
        factor=linalg.cho_factor(gram,lower=True,check_finite=False,overwrite_a=True)
        precision=linalg.cho_solve(factor,np.eye(len(gram)),check_finite=False)
        self.W=-precision/np.diag(precision)[None,:]
        np.fill_diagonal(self.W,0)
        self.fit_seconds=perf_counter()-start
        return self
    def score(self,users):return self.binary[np.asarray(users)]@self.W
    @property
    def parameter_bytes(self):return self.W.nbytes
