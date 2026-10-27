"""Categorical metadata reconstructed strictly from earlier interactions."""
import numpy as np
import pandas as pd

SPAN=10_000_000_000
COUNT_BINS=np.array([0,1,2,4,8,16,32,64,128])
AGE_BINS=np.array([0,30,90,365,1095])


class TemporalFeatures:
    def __init__(self,history,shape,cutoff):
        self.shape=shape;self.cutoff=int(cutoff)
        n_users,n_items=shape
        t=history.timestamp.to_numpy(dtype=np.int64)
        if np.any((t<0)|(t>=SPAN)): raise ValueError('Timestamps outside encoder range')
        self.user_keys=np.sort(history.u.to_numpy(dtype=np.int64)*SPAN+t)
        self.item_keys=np.sort(history.i.to_numpy(dtype=np.int64)*SPAN+t)
        self.user_start=np.r_[0,np.cumsum(np.bincount(history.u,minlength=n_users))][:-1]
        self.item_start=np.r_[0,np.cumsum(np.bincount(history.i,minlength=n_items))][:-1]
        self.first_item=np.full(n_items,SPAN,dtype=np.int64)
        np.minimum.at(self.first_item,history.i,t)
        self.offsets=np.array([0,n_users,n_users+n_items,n_users+n_items+10,
                               n_users+n_items+20,n_users+n_items+26])
        self.n_features=n_users+n_items+38

    def counts(self,users,items,times):
        users=np.asarray(users,dtype=np.int64);items=np.asarray(items,dtype=np.int64)
        times=np.broadcast_to(np.asarray(times,dtype=np.int64),users.shape)
        uc=np.searchsorted(self.user_keys,users*SPAN+times,side='left')-self.user_start[users]
        ic=np.searchsorted(self.item_keys,items*SPAN+times,side='left')-self.item_start[items]
        return uc,ic

    def encode(self,users,items,times,metadata=True):
        users=np.asarray(users,dtype=np.int64);items=np.asarray(items,dtype=np.int64)
        times=np.broadcast_to(np.asarray(times,dtype=np.int64),users.shape)
        base=np.column_stack([users,items+self.offsets[1]])
        if not metadata: return base
        uc,ic=self.counts(users,items,times)
        age=np.where(ic>0,(times-self.first_item[items])/86400,-1)
        month=pd.to_datetime(times,unit='s',utc=True).month.to_numpy()-1
        return np.column_stack([base,
             self.offsets[2]+np.searchsorted(COUNT_BINS,uc,side='right'),
             self.offsets[3]+np.searchsorted(COUNT_BINS,ic,side='right'),
             self.offsets[4]+np.searchsorted(AGE_BINS,age,side='right'),
             self.offsets[5]+month]).astype(np.int64)

    def snapshot_sides(self,users,metadata=True):
        users=np.asarray(users,dtype=np.int64);items=np.arange(self.shape[1])
        up=self.encode(users,np.zeros(len(users),dtype=int),self.cutoff,metadata)
        ip=self.encode(np.zeros(len(items),dtype=int),items,self.cutoff,metadata)
        return (up[:,[0,2,5]],ip[:,[1,3,4]]) if metadata else (up[:,[0]],ip[:,[1]])
