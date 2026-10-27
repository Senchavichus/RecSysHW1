"""Official data download, deterministic cleaning and train-only selection."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
import gzip
import hashlib
import html
import json
import re
import time
import numpy as np
import pandas as pd
from scipy import sparse

SOURCE='https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/'
BASE='https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/'
FILES={
    'Video_Games.csv':(BASE+'categoryFilesSmall/Video_Games.csv',107048392,
                      'c3fec1cce4ee67f0d51f8b53a7e294d8bbe6f75ca1d8732256e900b3c46606d8'),
    'meta_Video_Games.json.gz':(BASE+'metaFiles2/meta_Video_Games.json.gz',53279669,
                              '4069e17c7efa33e7574c2e77f1fe0b0c6f2a42a7ac29c4422696d788d8a8e0a6')}
TRAIN_END=int(pd.Timestamp('2016-01-01',tz='UTC').timestamp())
VALID_END=int(pd.Timestamp('2017-01-01',tz='UTC').timestamp())
TEST_END=int(pd.Timestamp('2018-01-01',tz='UTC').timestamp())


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(2**20): digest.update(chunk)
    return digest.hexdigest()


def download_sources(cache_dir):
    """Bounded range downloads avoid stalled large-file transfers; verify full hash."""
    root=Path(cache_dir);root.mkdir(parents=True,exist_ok=True)
    for name,(url,size,digest) in FILES.items():
        destination=root/name
        if destination.exists():
            if destination.stat().st_size==size and sha256(destination)==digest: continue
            raise ValueError(f'Cached source {name} differs from the pinned file; move it aside and retry')
        print(f'Downloading {name} ({size/2**20:.1f} MiB)',flush=True)
        temporary=destination.with_suffix(destination.suffix+'.part')
        chunk_size=4*2**20
        def get(start):
            end=min(start+chunk_size,size)-1
            for attempt in range(3):
                try:
                    request=Request(url,headers={'Range':f'bytes={start}-{end}'})
                    with urlopen(request,timeout=60) as response:
                        if response.status!=206: raise IOError('Server did not honor byte range')
                        content=response.read(end-start+2)
                        if len(content)!=end-start+1: raise IOError('Incomplete download range')
                    return start,content
                except (OSError,TimeoutError):
                    if attempt==2: raise
                    time.sleep(attempt+1)
        with temporary.open('wb') as out,ThreadPoolExecutor(max_workers=4) as pool:
            jobs=[pool.submit(get,start) for start in range(0,size,chunk_size)]
            for job in as_completed(jobs):
                start,content=job.result();out.seek(start);out.write(content)
        if temporary.stat().st_size!=size or sha256(temporary)!=digest:
            raise IOError(f'Integrity check failed: {name}')
        temporary.replace(destination)
    return root


def clean_text(value):
    if not isinstance(value,str): return ''
    return re.sub(r'\s+',' ',re.sub(r'<[^>]*>',' ',html.unescape(value))).strip()


def read_catalog(path):
    records={};rows=0
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            d=json.loads(line);rows+=1;item=d.get('asin');title=clean_text(d.get('title'))
            if not item or len(title)<2: continue
            cats=[clean_text(v) for v in d.get('category',[]) if isinstance(v,str)]
            record={'item_id':item,'title':title,'brand':clean_text(d.get('brand')),
                    'category':' / '.join(v for v in cats if v)}
            # File-order first usable title is deterministic for pinned source bytes.
            if item not in records: records[item]=record
    catalog=pd.DataFrame(records.values()).sort_values('item_id').reset_index(drop=True)
    return catalog,{'metadata_rows':rows,'metadata_unique_usable_items':len(catalog)}


def clean_reviews(raw,valid_titles):
    f=raw.dropna(subset=['user_id','item_id','rating','timestamp']).copy()
    f=f[f.rating.between(1,5)&(f.timestamp>0)]
    f=f.sort_values(['timestamp','user_id','item_id','rating'],kind='stable')
    f=f.drop_duplicates(['user_id','item_id'],keep='first')
    return f[f.item_id.isin(valid_titles)].reset_index(drop=True)


def history_matrices(frame,shape):
    observed=sparse.csr_matrix((np.ones(len(frame)),(frame.u,frame.i)),shape=shape)
    positive=frame[frame.rating>=4]
    binary=sparse.csr_matrix((np.ones(len(positive)),(positive.u,positive.i)),shape=shape)
    observed.sum_duplicates();observed.data[:]=1
    binary.sum_duplicates();binary.data[:]=1
    return observed,binary


def prepare_data(cache_dir):
    root=download_sources(cache_dir)
    raw=pd.read_csv(root/'Video_Games.csv',header=None,
                    names=['item_id','user_id','rating','timestamp'],
                    dtype={'item_id':str,'user_id':str,'rating':float,'timestamp':np.int64})
    catalog,meta_audit=read_catalog(root/'meta_Video_Games.json.gz')
    clean=clean_reviews(raw,set(catalog.item_id))
    train_raw=clean[clean.timestamp<TRAIN_END]
    ranking=(train_raw[train_raw.rating>=4].groupby('item_id').user_id.nunique()
             .reset_index(name='positive_users').sort_values(['positive_users','item_id'],ascending=[False,True]))
    chosen=set(ranking.head(4000).item_id)
    train_subset=train_raw[train_raw.item_id.isin(chosen)]
    counts=train_subset.groupby('user_id').size()
    users=set(counts[counts>=5].index)&set(train_subset[train_subset.rating>=4].user_id)
    train_subset=train_subset[train_subset.user_id.isin(users)]
    items=sorted(train_subset.item_id.unique());user_ids=sorted(users)
    item_map={v:i for i,v in enumerate(items)};user_map={v:i for i,v in enumerate(user_ids)}
    selected=clean[clean.user_id.isin(users)&clean.item_id.isin(item_map)&(clean.timestamp<TEST_END)].copy()
    selected['u']=selected.user_id.map(user_map).astype(np.int32)
    selected['i']=selected.item_id.map(item_map).astype(np.int32)
    train=selected[selected.timestamp<TRAIN_END].copy()
    valid=selected[(selected.timestamp>=TRAIN_END)&(selected.timestamp<VALID_END)].copy()
    test=selected[selected.timestamp>=VALID_END].copy()
    shape=(len(user_ids),len(items))
    assert 100000<=len(selected)<=2000000, 'Cleaned interaction count outside homework bounds'
    assert min(shape)>=1000,'Not enough users/items'
    assert not selected.duplicated(['u','i']).any()
    assert train.timestamp.max()<TRAIN_END<=valid.timestamp.min()<VALID_END<=test.timestamp.min()
    catalog=catalog.set_index('item_id').loc[items].reset_index();catalog['i']=np.arange(len(items))
    raw_audit={'source_rows':len(raw),'source_users':raw.user_id.nunique(),'source_items':raw.item_id.nunique(),
               'missing':raw.isna().sum().to_dict(),'exact_duplicates':int(raw.duplicated().sum()),
               'repeated_pairs':int(raw.duplicated(['user_id','item_id']).sum()),
               'clean_unique_pairs':len(clean),'selected_pairs':len(selected),
               'rating_distribution':{str(k):int(v) for k,v in raw.rating.value_counts().sort_index().items()},
               'first_timestamp':int(raw.timestamp.min()),'last_timestamp':int(raw.timestamp.max()),
               'users':shape[0],'items':shape[1],**meta_audit}
    # Coverage counts use all held-out source pairs, before warm-ID selection.
    coverage=[]
    for name,lo,hi,part in [('validation',TRAIN_END,VALID_END,valid),('test',VALID_END,TEST_END,test)]:
        future=clean[(clean.timestamp>=lo)&(clean.timestamp<hi)]
        coverage.append({'period':name,'source_pairs':len(future),'source_users':future.user_id.nunique(),
                         'source_positive_pairs':int((future.rating>=4).sum()),
                         'pairs_with_unknown_user':int((~future.user_id.isin(user_map)).sum()),
                         'pairs_with_unknown_item':int((~future.item_id.isin(item_map)).sum()),
                         'unknown_user_ids':int(future.loc[~future.user_id.isin(user_map),'user_id'].nunique()),
                         'unknown_item_ids':int(future.loc[~future.item_id.isin(item_map),'item_id'].nunique()),
                         'selected_pairs':len(part),'selected_positive_pairs':int((part.rating>=4).sum()),
                         'positive_pair_retention':float((part.rating>=4).sum()/max(1,(future.rating>=4).sum()))})
    return dict(train=train,valid=valid,test=test,catalog=catalog,shape=shape,
                raw_audit=raw_audit,coverage=coverage,user_ids=user_ids,
                train_end=TRAIN_END,valid_end=VALID_END,test_end=TEST_END)
