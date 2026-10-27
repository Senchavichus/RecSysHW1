"""One full-catalog evaluation protocol shared by all methods."""
from time import perf_counter
import numpy as np
import pandas as pd
from .data import history_matrices


def stable_topk(scores,k=10):
    candidates=np.flatnonzero(np.isfinite(scores))
    order=np.lexsort((candidates,-scores[candidates]))
    return candidates[order[:k]]


def user_metrics(top,relevant,k=10):
    hits=np.array([int(i) in relevant for i in top],dtype=float)
    dcg=np.sum(hits/np.log2(np.arange(len(hits))+2))
    ideal=np.sum(1/np.log2(np.arange(min(k,len(relevant)))+2))
    return {f'NDCG@{k}':float(dcg/ideal) if ideal else 0.,
            f'Recall@{k}':float(hits.sum()/len(relevant)) if relevant else 0.}


class Evaluation:
    def __init__(self,history,heldout,shape):
        self.observed,self.positive=history_matrices(history,shape)
        self.candidates=np.asarray(self.positive.sum(axis=0)).ravel()>0
        warm=np.diff(self.observed.indptr)>0
        relevant=heldout[(heldout.rating>=4)&heldout.i.map(lambda i:self.candidates[int(i)])
                         &heldout.u.map(lambda u:warm[int(u)])]
        self.targets={}
        for u,g in relevant.groupby('u',sort=True):
            seen=set(self.observed.indices[self.observed.indptr[u]:self.observed.indptr[u+1]])
            goals=set(g.i)-seen
            if goals:self.targets[int(u)]=goals
        self.users=np.array(sorted(self.targets),dtype=np.int64)
        if not len(self.users):raise ValueError('No eligible held-out users')

    def evaluate(self,model,k=10,batch_size=128):
        rows=[];recommendations={};covered=set();start=perf_counter()
        for begin in range(0,len(self.users),batch_size):
            users=self.users[begin:begin+batch_size]
            scores=np.asarray(model.score(users),dtype=float).copy()
            if scores.shape!=(len(users),self.observed.shape[1]) or not np.isfinite(scores).all():
                raise ValueError('Scorer returned invalid shape or nonfinite values')
            scores[:,~self.candidates]=-np.inf
            for j,u in enumerate(users):
                seen=self.observed.indices[self.observed.indptr[u]:self.observed.indptr[u+1]]
                scores[j,seen]=-np.inf
                top=stable_topk(scores[j],k);recommendations[int(u)]=top
                covered.update(top.tolist())
                rows.append({'u':int(u),'history_length':len(seen),'relevant_items':len(self.targets[int(u)]),
                             **user_metrics(top,self.targets[int(u)],k)})
        details=pd.DataFrame(rows)
        summary={key:float(details[key].mean()) for key in [f'NDCG@{k}',f'Recall@{k}']}
        summary.update(eval_users=len(self.users),eligible_positives=sum(map(len,self.targets.values())),
                       candidate_items=int(self.candidates.sum()),coverage=len(covered)/self.candidates.sum(),
                       retrieval_seconds=perf_counter()-start)
        return summary,details,recommendations
