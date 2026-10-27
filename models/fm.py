"""Order-two FM, explicit BPR derivatives and NumPy Adam optimizer."""
from time import perf_counter
import numpy as np
from scipy.special import expit
from utils.data import history_matrices


def sample_negatives(rng,users,times,seen,first_item):
    """Uniform among known-before-time and unseen items; -1 if none exist."""
    users=np.asarray(users,dtype=int);times=np.asarray(times,dtype=np.int64)
    result=np.full(len(users),-1,dtype=int)
    pending=np.arange(len(users))
    for _ in range(64):
        if not len(pending):break
        candidates=rng.integers(0,seen.shape[1],size=len(pending))
        unobserved=np.asarray(seen[users[pending],candidates]).ravel()==0
        ok=unobserved&(first_item[candidates]<times[pending])
        result[pending[ok]]=candidates[ok];pending=pending[~ok]
    for j in pending:
        possible=np.flatnonzero(first_item<times[j])
        a,b=seen.indptr[users[j]:users[j]+2]
        possible=possible[~np.isin(possible,seen.indices[a:b])]
        if len(possible):result[j]=rng.choice(possible)
    return result


class FactorizationMachine:
    def __init__(self,features,factors=16,lr=.003,reg=1e-5,seed=42,
                 use_metadata=True,batch_size=2048):
        self.features=features;self.factors=factors;self.lr=lr;self.reg=reg
        self.seed=seed;self.use_metadata=use_metadata;self.batch_size=batch_size
        rng=np.random.default_rng(seed)
        self.w=np.zeros(features.n_features)
        self.V=rng.normal(0,.03,(features.n_features,factors))
        self.mw=np.zeros_like(self.w);self.vw=np.zeros_like(self.w)
        self.mv=np.zeros_like(self.V);self.vv=np.zeros_like(self.V)
        self.step_number=0;self.fit_seconds=0.;self.history=[]

    def forward(self,ids):
        embeddings=self.V[ids];sums=embeddings.sum(axis=1)
        return self.w[ids].sum(axis=1)+.5*np.sum(sums*sums-(embeddings*embeddings).sum(axis=1),axis=1)

    def gradients(self,positive,negative):
        delta=self.forward(positive)-self.forward(negative)
        error=-expit(-delta)/len(delta)
        loss=float(np.logaddexp(0,-delta).mean()+.5*self.reg*(np.sum(self.w**2)+np.sum(self.V**2)))
        gw=self.reg*self.w.copy();gv=self.reg*self.V.copy()
        for ids,err in [(positive,error),(negative,-error)]:
            embeddings=self.V[ids];sums=embeddings.sum(axis=1)
            np.add.at(gw,ids.ravel(),np.broadcast_to(err[:,None],ids.shape).ravel())
            contribution=err[:,None,None]*(sums[:,None,:]-embeddings)
            np.add.at(gv,ids.ravel(),contribution.reshape(-1,self.factors))
        return loss,gw,gv

    def update(self,gw,gv):
        self.step_number+=1;t=self.step_number
        for param,grad,m,v in [(self.w,gw,self.mw,self.vw),(self.V,gv,self.mv,self.vv)]:
            m*=.9;m+=.1*grad;v*=.999;v+=.001*grad*grad
            param-=self.lr*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)

    def fit_epoch(self,history,epoch):
        begin=perf_counter()
        positive=history[history.rating>=4]
        users=positive.u.to_numpy(dtype=int);items=positive.i.to_numpy(dtype=int)
        times=positive.timestamp.to_numpy(dtype=np.int64)
        seen,_=history_matrices(history,self.features.shape)
        rng=np.random.default_rng(self.seed+epoch*1009)
        negative=sample_negatives(rng,users,times,seen,self.features.first_item)
        keep=negative>=0
        pos=self.features.encode(users[keep],items[keep],times[keep],self.use_metadata)
        neg=self.features.encode(users[keep],negative[keep],times[keep],self.use_metadata)
        losses=[]
        order=rng.permutation(len(pos))
        for start in range(0,len(order),self.batch_size):
            rows=order[start:start+self.batch_size]
            loss,gw,gv=self.gradients(pos[rows],neg[rows]);self.update(gw,gv);losses.append(loss)
        self.fit_seconds+=perf_counter()-begin
        self.history.append({'epoch':epoch,'loss':float(np.mean(losses)) if losses else 0.,
                             'positive_pairs':len(positive),'skipped_no_negative':int((~keep).sum())})
        return self

    def score(self,users):
        left,right=self.features.snapshot_sides(users,self.use_metadata)
        def side(ids):
            emb=self.V[ids];s=emb.sum(axis=1)
            own=self.w[ids].sum(axis=1)+.5*np.sum(s*s-(emb*emb).sum(axis=1),axis=1)
            return s,own
        a,sa=side(left);b,sb=side(right)
        return sa[:,None]+sb[None,:]+a@b.T

    @property
    def parameter_bytes(self):return self.w.nbytes+self.V.nbytes
