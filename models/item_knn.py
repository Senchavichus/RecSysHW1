"""Cosine item neighborhoods with overlap shrinkage and deterministic pruning."""
from time import perf_counter
import numpy as np
from scipy import sparse


class ItemKNN:
    def __init__(self,neighbors=40,shrinkage=10.,min_overlap=2):
        if neighbors<1 or shrinkage<0 or min_overlap<1:raise ValueError('Invalid neighbors')
        self.neighbors=neighbors;self.shrinkage=shrinkage;self.min_overlap=min_overlap
    def fit(self,binary):
        start=perf_counter();self.binary=sparse.csr_matrix(binary,dtype=float)
        overlap=(self.binary.T@self.binary).toarray()
        norms=np.sqrt(np.diag(overlap));den=norms[:,None]*norms[None,:]
        sim=np.divide(overlap,den,out=np.zeros_like(overlap),where=den>0)
        sim*=np.divide(overlap,overlap+self.shrinkage,out=np.zeros_like(overlap),
                       where=overlap+self.shrinkage>0)
        sim[overlap<self.min_overlap]=0;np.fill_diagonal(sim,0)
        # Each source keeps its strongest candidate neighbors; scores are B @ W.
        rows=[];cols=[];values=[]
        for i,row in enumerate(sim):
            ids=np.flatnonzero(row>0)
            ids=ids[np.lexsort((ids,-row[ids]))[:self.neighbors]]
            rows.extend([i]*len(ids));cols.extend(ids);values.extend(row[ids])
        self.W=sparse.csr_matrix((values,(rows,cols)),shape=sim.shape)
        self.fit_seconds=perf_counter()-start;return self
    def score(self,users):return (self.binary[np.asarray(users)]@self.W).toarray()
    @property
    def parameter_bytes(self):return self.W.data.nbytes+self.W.indices.nbytes+self.W.indptr.nbytes
