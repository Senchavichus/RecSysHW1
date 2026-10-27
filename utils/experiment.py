"""Validation search, explicit freeze, refit and one final common test."""
from itertools import product
from datetime import datetime,timezone
from time import perf_counter
import json
import numpy as np
import pandas as pd
from models.item_knn import ItemKNN
from models.ials import IALS
from models.custom_als import CustomALS
from models.ease import EASE
from models.fm import FactorizationMachine
from .features import TemporalFeatures
from .data import history_matrices
from .evaluation import Evaluation

SEED=42


def search_space(quick=False):
    if quick:
        return dict(knn=[dict(neighbors=2,shrinkage=1,min_overlap=1)],ease=[dict(reg=2.)],
                    als=[dict(factors=2,reg=1.,alpha=10.)],halves=[180],als_epochs=[1,2],
                    fm=[dict(factors=2,lr=.01)],fm_epochs=[1,2])
    return dict(knn=[dict(neighbors=k,shrinkage=s,min_overlap=2) for k,s in product([40,100],[50.,10.])],
                ease=[dict(reg=r) for r in [1000.,300.,100.,30.]],
                als=[dict(factors=d,reg=r,alpha=10.) for d,r in product([24,48],[10.,1.])],
                halves=[180,730],als_epochs=[5,10,15],
                fm=[dict(factors=d,lr=lr) for d,lr in product([16,32],[.003,.01])],fm_epochs=[3,6,10])


def run_experiments(data,quick=False):
    train=data['train'];shape=data['shape'];space=search_space(quick)
    observed,binary=history_matrices(train,shape)
    validation=Evaluation(train,data['valid'],shape)
    features=TemporalFeatures(train,shape,data['train_end'])
    rows=[];curves=[];chosen={}

    def record(name,params,model,epoch=None):
        metrics,_,_=validation.evaluate(model)
        settings={**params,**({'epochs':epoch} if epoch is not None else {})}
        row={'model':name,'settings':json.dumps(settings,sort_keys=True),
             'fit_seconds':getattr(model,'fit_seconds',0.),**metrics}
        rows.append(row)
        old=chosen.get(name)
        if old is None or row['NDCG@10']>old['metric']+1e-12:
            chosen[name]={'metric':row['NDCG@10'],'params':settings}
        if name!='FM IDs':print(f'validation {name} {settings}: NDCG={metrics["NDCG@10"]:.6f}',flush=True)

    for p in space['knn']:record('ItemKNN',p,ItemKNN(**p).fit(binary))
    for p in space['ease']:record('EASE',p,EASE(**p).fit(binary))

    for name,cls,parameter_list in [
        ('iALS',IALS,space['als']),
        ('CustomALS',CustomALS,[dict(p,half_life_days=h) for p,h in product(space['als'],space['halves'])])]:
        for p in parameter_list:
            def callback(model,epoch):
                curves.append({'model':name,'settings':json.dumps(p,sort_keys=True),**model.history[-1]})
                if epoch in space['als_epochs']:record(name,p,model,epoch)
            cls(**p,seed=SEED).fit_history(train,shape,data['train_end'],
                                          max(space['als_epochs']),callback)

    for metadata in [False,True]:
        name='FM metadata' if metadata else 'FM IDs'
        for p in space['fm']:
            model=FactorizationMachine(features,**p,seed=SEED,use_metadata=metadata)
            for epoch in range(1,max(space['fm_epochs'])+1):
                model.fit_epoch(train,epoch)
                curves.append({'model':name,'settings':json.dumps(p,sort_keys=True),**model.history[-1]})
                if epoch in space['fm_epochs']:record(name,p,model,epoch)

    # Fixed validation-only diagnostic isolates graded ratings without decay.
    p=dict(chosen['iALS']['params']);epochs=p.pop('epochs')
    grade_only=CustomALS(**p,half_life_days=None,seed=SEED).fit_history(
        train,shape,data['train_end'],epochs)
    grade_metrics,_,_=validation.evaluate(grade_only)
    grade_control=pd.DataFrame([{'model':'ALS graded, no decay','settings':json.dumps(dict(p,epochs=epochs)),
                                 **grade_metrics}])

    frozen={'primary_metric':'NDCG@10','seed':SEED,'quick_smoke_test':quick,
            'search_space':space,'selected':chosen,
            'frozen_before_test_utc':datetime.now(timezone.utc).isoformat(),
            'tie_rule':'first configuration in documented simple-first order; earliest checkpoint'}
    print('Settings frozen. Starting refit and final test.',flush=True)

    final_history=pd.concat([train,data['valid']],ignore_index=True)
    _,final_binary=history_matrices(final_history,shape)
    final_features=TemporalFeatures(final_history,shape,data['valid_end'])
    # Test targets are inspected for evaluation only after the freeze above.
    test_eval=Evaluation(final_history,data['test'],shape)
    final_models={};test_rows=[];details={};recs={}
    for name in ['ItemKNN','iALS','EASE','FM IDs','FM metadata','CustomALS']:
        p=dict(chosen[name]['params']);epochs=p.pop('epochs',None)
        start=perf_counter()
        if name=='ItemKNN':model=ItemKNN(**p).fit(final_binary)
        elif name=='EASE':model=EASE(**p).fit(final_binary)
        elif name in ['iALS','CustomALS']:
            cls=IALS if name=='iALS' else CustomALS
            model=cls(**p,seed=SEED).fit_history(final_history,shape,data['valid_end'],epochs)
        else:
            model=FactorizationMachine(final_features,**p,seed=SEED,use_metadata=name=='FM metadata')
            for epoch in range(1,epochs+1):model.fit_epoch(final_history,epoch)
        fit_seconds=perf_counter()-start
        metrics,detail,recommendations=test_eval.evaluate(model)
        parameter_bytes=getattr(model,'parameter_bytes',getattr(model,'counts',np.empty(0)).nbytes)
        row={'model':name,'settings':json.dumps(chosen[name]['params'],sort_keys=True),
             'final_fit_seconds':fit_seconds,'parameter_MiB':parameter_bytes/2**20,**metrics}
        test_rows.append(row);details[name]=detail;recs[name]=recommendations;final_models[name]=model
        if name!='FM IDs':print(f'test {name}: NDCG={metrics["NDCG@10"]:.6f}, Recall={metrics["Recall@10"]:.6f}',flush=True)
    test=pd.DataFrame(test_rows)
    assert test[['eval_users','eligible_positives','candidate_items']].nunique().eq(1).all()
    intervals=[]
    for left,right in [('CustomALS','iALS'),('FM metadata','FM IDs')]:
        a=details[left].set_index('u')['NDCG@10'];b=details[right].set_index('u')['NDCG@10']
        assert a.index.equals(b.index)
        delta=(a-b).to_numpy();rng=np.random.default_rng(SEED)
        draws=rng.integers(0,len(delta),size=(2000,len(delta)))
        low,high=np.quantile(delta[draws].mean(axis=1),[.025,.975])
        intervals.append(dict(comparison=f'{left} minus {right}',mean_delta=delta.mean(),lower=low,upper=high))
    bootstrap=pd.DataFrame(intervals)
    # ID-only is an ablation of the same FM, never a sixth main model.
    primary=['ItemKNN','iALS','EASE','FM metadata','CustomALS']
    fm_ablation=test[test.model.isin(['FM IDs','FM metadata'])].copy()
    frozen['fm_ablation_settings']=frozen['selected']['FM IDs']
    frozen['selected']={name:chosen[name] for name in primary}
    return dict(validation=pd.DataFrame(rows).query('model != \"FM IDs\"'),
                settings=frozen,test=test[test.model.isin(primary)].reset_index(drop=True),
                details={k:details[k] for k in primary},
                recommendations={k:recs[k] for k in primary},
                models={k:final_models[k] for k in primary},
                curves=pd.DataFrame(curves).query('model != \"FM IDs\"'),
                grade_control=grade_control,bootstrap=bootstrap,
                test_eval=test_eval,fm_ablation=fm_ablation)
