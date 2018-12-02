
# coding: utf-8

# In[ ]:


##数据的预处理
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import warnings
import gc
import datetime
from sklearn.preprocessing import minmax_scale
warnings.filterwarnings('ignore')
def labelcoder(x):
    if x=='pd':
        return 0
    if x=='fq':
        return 1
    return 2

def minday(x):
    import datetime
    data_time1 = datetime.datetime.strptime('2017-04-04', '%Y-%m-%d')
    data_time2 = datetime.datetime.strptime('2017-05-01', '%Y-%m-%d')
    data_time3 = datetime.datetime.strptime('2017-05-20', '%Y-%m-%d')
    data_time4 = datetime.datetime.strptime('2017-05-30', '%Y-%m-%d')
    data_time5 = datetime.datetime.strptime('2017-08-28', '%Y-%m-%d')
    data_time6 = datetime.datetime.strptime('2017-10-01', '%Y-%m-%d')
    data_time7 = datetime.datetime.strptime('2017-10-04', '%Y-%m-%d')
    data_time8 = datetime.datetime.strptime('2017-10-28', '%Y-%m-%d')
    data_time9 = datetime.datetime.strptime('2017-10-11', '%Y-%m-%d')
    data_time10 = datetime.datetime.strptime('2017-12-25', '%Y-%m-%d')
    data_time11 = datetime.datetime.strptime('2018-01-01', '%Y-%m-%d')
    data_time12 = datetime.datetime.strptime('2018-02-26', '%Y-%m-%d')
    data_time13 = datetime.datetime.strptime('2018-03-02', '%Y-%m-%d')
    data_time14 = datetime.datetime.strptime('2018-03-07', '%Y-%m-%d')
    data_time15 = datetime.datetime.strptime('2018-04-05', '%Y-%m-%d')
    data_time16 = datetime.datetime.strptime('2018-04-05', '%Y-%m-%d')
    zz=[data_time1,data_time2,data_time3,data_time4,data_time5,data_time6,data_time7,data_time8,data_time9,data_time10,data_time11,data_time12,data_time13,data_time14,data_time15,data_time16]
    diffdate=[]
    for t in zz:
        diffdate.append(abs((x-t).days))
    mindate=min(diffdate)
    if mindate<=7:
        return mindate
    else:
        return -1
def lgb_data(X_train,y_train,X_valid,y_valid,test_set):
    for t in label_feature:
        train_df=X_train.join(y_train)
        train_df=train_df.fillna(-1)
        train_df=train_df.reset_index(drop=True)
        test_set=test_set.fillna(-1)
        train_df['label'] = train_df['label'].apply(float).apply(int)
        uid_cvr = train_df.groupby(t).label.agg(np.mean).reset_index()
        uid_cvr.columns = [t,t+'mean']
        X_valid = pd.merge(X_valid, uid_cvr, how='left', on=t)
        X_valid[t+'mean'].fillna(0, inplace=True)
        X_valid[t+'mean'] = minmax_scale(X_valid[t+'mean'], feature_range=(0.5, 1))   # 最小的数值设置为0.5以上
        test_set = pd.merge(test_set, uid_cvr, how='left', on=t)
        test_set[t+'mean'].fillna(0, inplace=True)
        test_set[t+'mean'] = minmax_scale(test_set[t+'mean'], feature_range=(0.5, 1))   # 最小的数值设置为0.5以上
        del uid_cvr
        gc.collect()

        uid_cvr = train_df[[t, 'label']]
        uid_label_cnt = uid_cvr.groupby(t).agg(sum).reset_index()
        uid_label_cnt.columns = [t, t+'uid_label_cnt']
        uid_cnt = uid_cvr.groupby(t).agg(len).reset_index()
        uid_cnt.columns = [t,t+ 'uid_cnt']
        uid_cvr = pd.merge(uid_cvr, uid_cnt, how='left', on=t)
        uid_cvr = pd.merge(uid_cvr, uid_label_cnt, how='left', on=t)
        uid_cvr[t+'true_uid_cnt'] = uid_cvr[t+'uid_cnt'] - 1
        uid_cvr[t+'true_uid_label_cnt'] = uid_cvr[t+'uid_label_cnt'] - uid_cvr['label']
        uid_cvr[t+'true_uid_cnt'].replace(0, 1, inplace=True)
  
        uid_cvr[t+'cvr_of_uid'] = uid_cvr[t+'true_uid_label_cnt'] / uid_cvr[t+'true_uid_cnt']

        uid_cvr[t+'cvr_of_uid'] = minmax_scale(uid_cvr[t+'cvr_of_uid'], feature_range=(0.5, 1))


        uid_cvr.pop(t+'true_uid_cnt')
        uid_cvr.pop(t+'true_uid_label_cnt')
        uid_cvr.pop(t+'uid_label_cnt')
        uid_cvr.pop(t+'uid_cnt')
        uid_cvr.pop('label')

        X_train[t+'mean']=uid_cvr[t+'cvr_of_uid']
        del uid_cvr
        gc.collect()    

    return X_train,y_train,X_valid,y_valid,test_set

def mindayindex(x):
    import datetime
    data_time1 = datetime.datetime.strptime('2017-04-04', '%Y-%m-%d')
    data_time2 = datetime.datetime.strptime('2017-05-01', '%Y-%m-%d')
    data_time3 = datetime.datetime.strptime('2017-05-20', '%Y-%m-%d')
    data_time4 = datetime.datetime.strptime('2017-05-30', '%Y-%m-%d')
    data_time5 = datetime.datetime.strptime('2017-08-28', '%Y-%m-%d')
    data_time6 = datetime.datetime.strptime('2017-10-01', '%Y-%m-%d')
    data_time7 = datetime.datetime.strptime('2017-10-04', '%Y-%m-%d')
    data_time8 = datetime.datetime.strptime('2017-10-28', '%Y-%m-%d')
    data_time9 = datetime.datetime.strptime('2017-10-11', '%Y-%m-%d')
    data_time10 = datetime.datetime.strptime('2017-12-25', '%Y-%m-%d')
    data_time11 = datetime.datetime.strptime('2018-01-01', '%Y-%m-%d')
    data_time12 = datetime.datetime.strptime('2018-02-26', '%Y-%m-%d')
    data_time13 = datetime.datetime.strptime('2018-03-02', '%Y-%m-%d')
    data_time14 = datetime.datetime.strptime('2018-03-07', '%Y-%m-%d')
    data_time15 = datetime.datetime.strptime('2018-04-05', '%Y-%m-%d')
    data_time16 = datetime.datetime.strptime('2018-04-05', '%Y-%m-%d')
    zz=[data_time1,data_time2,data_time3,data_time4,data_time5,data_time6,data_time7,data_time8,data_time9,data_time10,data_time11,data_time12,data_time13,data_time14,data_time15,data_time16]
    diffdate=[]
    for t in zz:
        diffdate.append(abs((x-t).days))
    mindate=min(diffdate)
    minindex=diffdate.index(min(diffdate))
    if mindate<=7:
        return minindex
    else:
        return -1
for i in range(1,6):
    filename = 'train_'+str(i)
    if i==1:
        train_x = pd.read_csv('train/'+filename,sep='\t')
        col_all = train_x.columns
    else:
        temp = pd.read_csv('train/'+filename,sep='\t',header=None)
        temp.columns = col_all
        train_x = pd.concat([train_x,temp],axis=0,ignore_index=True)
del temp
res = train_x[['id','tag']]
del train_x['id']
del train_x['tag']
gc.collect()
train_y = train_x.pop('label')
train_x['day'] = (train_x['loan_dt'].apply(lambda x:x.split('-')[2])).astype(int)
train_x['month'] = (train_x['loan_dt'].apply(lambda x:x.split('-')[1])).astype(int)
train_x['year'] = (train_x['loan_dt'].apply(lambda x:x.split('-')[0])).astype(int)
del train_x['loan_dt']
gc.collect()
train = train_x[train_y>=0].reset_index()
del train['index']
gc.collect()

train_y_cat = pd.Series(train_y[train_y>=0].values)


params = {
    'num_leaves':10, 
    'objective':'binary',
    'learning_rate':0.01,
    'metric':'auc',
    'boosting':'gbdt',
    'min_child_samples':10,
    'bagging_fraction':0.7, 
    'bagging_freq':1,
    'feature_fraction':0.7, 
    'reg_alpha':0,
    'reg_lambda':1
    }
skf = StratifiedKFold(n_splits=5,random_state=2020,shuffle=True)
for train_part_index,evals_index in skf.split(train,train_y_cat):
    EVAL_RESULT = {}
    train_part = lgb.Dataset(train.loc[train_part_index],label=train_y_cat.loc[train_part_index])
    evals = lgb.Dataset(train.loc[evals_index],label=train_y_cat.loc[evals_index])
    bst = lgb.train(params,train_part, 
          num_boost_round=10000, valid_sets=[train_part,evals],
          valid_names=['train','evals'],early_stopping_rounds=100,
          evals_result=EVAL_RESULT, verbose_eval=50)
    break
fse = pd.Series(bst.feature_importance(),index=train.columns).sort_values(ascending=False)
fse = fse[fse>0]
col = fse.index.tolist()
train_x[col].to_csv('process2/train_x.csv',index=False)
train_y.to_csv('process2/train_y.csv',index=False)
res.to_csv('process2/train_res.csv',index=False)


# In[ ]:


# 预处理后的特征
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

train_x = pd.read_csv('process2/train_x.csv')
col = train_x.columns
test_x = pd.read_csv('test.txt',sep='\t')#valid.txt
test_x['day'] = (test_x['loan_dt'].apply(lambda x:x.split('-')[2])).astype(int)
test_x['month'] = (test_x['loan_dt'].apply(lambda x:x.split('-')[1])).astype(int)
test_x['year'] = (test_x['loan_dt'].apply(lambda x:x.split('-')[0])).astype(int)
test_x = test_x[col]
sub = pd.read_csv('test_id.txt',sep='\t')
train_y = pd.read_csv('process2/train_y.csv',header=None)[0]
n = len(train_y.dropna())
train_x = train_x[:n]
train_y = pd.Series(train_y[train_y>=0].values)
params = {
    'num_leaves':80, 
    'learning_rate':0.01, 
    'boosting':'gbdt',
    'min_child_samples':10,
    'bagging_fraction':0.7, 
    'bagging_freq':1,
    'feature_fraction':0.7, 
    'reg_alpha':0,
    'reg_lambda':5, 
    'metric':'auc',
    'objective':'binary'
}

skf = StratifiedKFold(n_splits=5,random_state=2020,shuffle=True)
params_ = params.copy()
best_params = params_.copy()
score = []
res = pd.Series(0,index=test_x.index)
fse = pd.Series(0,index=train_x.columns)
for s in [0]:
    skf = StratifiedKFold(n_splits=5,random_state=s,shuffle=True)
    for train_part_index,evals_index in skf.split(train_x,train_y):
        EVAL_RESULT = {}
        train_part = lgb.Dataset(train_x.loc[train_part_index],label=train_y.loc[train_part_index])
        evals = lgb.Dataset(train_x.loc[evals_index],label=train_y.loc[evals_index])
        bst = lgb.train(best_params,train_part, 
              num_boost_round=10000, valid_sets=[train_part,evals],
              valid_names=['train','evals'],early_stopping_rounds=200,
              evals_result=EVAL_RESULT, verbose_eval=50)
        lst = EVAL_RESULT['evals']['auc']
        best_score = max(lst)
        print(best_score)
        score.append(best_score)
        best_iter = lst.index(best_score)+1
        res =  pd.Series(res.values+bst.predict(test_x,num_iteration = best_iter))
        fse = fse+pd.Series(bst.feature_importance(),index=train_x.columns)

sub['prob'] = res/5
sub.to_csv('result_sub/lgb2.txt',sep=",",index=False)

