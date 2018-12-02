
# coding: utf-8

# In[ ]:


#数据的预处理
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
        print('                ',train_x.shape)
    else:
        temp = pd.read_csv('train/'+filename,sep='\t',header=None)
        temp.columns = col_all
        train_x = pd.concat([train_x,temp],axis=0,ignore_index=True)
        print('                ',train_x.shape)
del temp
res = train_x[['id','tag']]
del train_x['id']
del train_x['tag']
train_y = train_x.pop('label')
train_x['day'] = (train_x['loan_dt'].apply(lambda x:x.split('-')[2])).astype(int)
train_x['month'] = (train_x['loan_dt'].apply(lambda x:x.split('-')[1])).astype(int)
train_x['year'] = (train_x['loan_dt'].apply(lambda x:x.split('-')[0])).astype(int)
load_dt= train_x.pop('loan_dt')


train = train_x[train_y>=0].reset_index()
del train['index']
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
train_x[col].to_csv('process1/train_x.csv',index=False)
train_y.to_csv('process1/train_y.csv',index=False)
res.to_csv('process1/train_res.csv',index=False)
load_dt.to_csv('process1/date.csv',index=False)


# In[ ]:


del train_x,train_y,res,load_dt


# In[ ]:


#model 1
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import minmax_scale


# In[ ]:


train = pd.read_csv('process1/train_x.csv')
load_dt=pd.read_csv('process1/date.csv',header=None)
train['loan_dt']=load_dt
col_feature = train.columns


# In[ ]:


val = pd.read_csv('test.txt',sep='\t')#val.txt
val['day'] = (val['loan_dt'].apply(lambda x:x.split('-')[2])).astype(int)
val['month'] = (val['loan_dt'].apply(lambda x:x.split('-')[1])).astype(int)
val['year'] = (val['loan_dt'].apply(lambda x:x.split('-')[0])).astype(int)
val = val[col_feature]


# In[ ]:


sub = pd.read_csv('test_id.txt',sep='\t')#valid_id.txt


# In[ ]:


y = pd.read_csv('process1/train_y.csv',header=None)[0]
n = len(y.dropna())
train = train[:n]
y = pd.Series(y[y>=0].values)


# In[ ]:


data=pd.concat([train,val])
data=data.reset_index(drop=True)
import gc
gc.collect()


# In[ ]:


drop_columns=['f4428','f2927','f3502','f3302','f4936','f2226','f1028','f403','f4942','f4402','f4385','f5521','f4306','f4956','f5405','f4937',
              'f3584','f885','f3445','f3324','f4221','f3507','f3441','f4938','f3471','f3222','tag']
for col in drop_columns:
    if col in col_feature:
        data.pop(col)


# In[ ]:


data['queshi']=data.shape[1] - data.count(axis=1)
data['loan_dt']=pd.to_datetime(data['loan_dt'])
data['dayofweek']=data['loan_dt'].dt.dayofweek
data['is_month_start']=data['loan_dt'].dt.is_month_start
data['is_month_end']=data['loan_dt'].dt.is_month_end
data['is_quarter_start']=data['loan_dt'].dt.is_quarter_start
data['is_quarter_end']=data['loan_dt'].dt.is_quarter_end


# In[ ]:


import pandas as pd
ff=pd.read_table('fee.txt',header=None)
ff.columns=['a']
zz=ff.a.map(lambda x:x.split(' ')[0])
label_feature=list(zz)
label_feature.remove('label')
label_feature = [i for i in label_feature if i in col_feature]
print(len(label_feature))
label_feature=label_feature[0:80]
data_temp = data[label_feature]
data_temp=data_temp.fillna(-1).astype(int)
data_temp=data_temp.astype(int)
df_feature = pd.DataFrame()
data_temp['cnt']=1
print('Begin ratio clcik...')
col_type = label_feature.copy()
n = len(col_type)
for i in range(n):
    col_name = "ratio_click_of_"+col_type[i]
    df_feature[col_name] =(data_temp[col_type[i]].map(data_temp[col_type[i]].value_counts())/len(data)*100).astype(int)
n = len(col_type)
for i in range(n):
    for j in range(n):
        if i!=j:
            col_name = "ratio_click_of_"+col_type[j]+"_in_"+col_type[i]
            se = data_temp.groupby([col_type[i],col_type[j]])['cnt'].sum()
            se=se.fillna(se.mean())
            dt = data_temp[[col_type[i],col_type[j]]]
            dt=dt.fillna(-1)
            cnt = data_temp[col_type[i]].map(data[col_type[i]].value_counts())
            cnt=cnt.fillna(cnt.mean())
            try:
                df_feature[col_name] = ((pd.merge(dt,se.reset_index(),how='left',on=[col_type[i],col_type[j]]).sort_index()['cnt'].fillna(value=0)/cnt)*100).astype(int).values
            except:
                print(i,j)
                continue
            
data = pd.concat([data, df_feature], axis=1)
print('The end')


# In[ ]:


data.pop('loan_dt')


# In[ ]:


data=data.reset_index(drop=True)
n=100000
train=data[0:n]
val=data[n:]


# In[ ]:


del data
gc.collect()


# In[ ]:


X_loc_train = train.values
del train
gc.collect()
y_loc_train = y.values
X_loc_test = val.values
del val
gc.collect()


# In[ ]:


res=pd.DataFrame()
res['id']=sub['id']


# In[ ]:


import lightgbm as lgb
import xgboost as xgb
lgb_clf = lgb.LGBMClassifier(num_leaves=80, learning_rate=0.01,boosting='gbdt',min_child_samples=10,bagging_fraction=0.7,n_estimators=40000,
                             bagging_freq=1,feature_fraction=0.7,reg_alpha=0,reg_lambda=1,metric='auc',objective='binary')
xgb_clf = xgb.XGBClassifier(max_depth=5, learning_rate=0.08,n_estimators=40000,objective='binary:logistic',
                            gamma=0,max_delta_step=0, subsample=0.7, 
                            colsample_bytree=0.7, #colsample_bylevel=0.9,
                            reg_alpha=1, reg_lambda=1, eval_metric= 'auc', seed=2018)#scale_pos_weight=4,


# In[ ]:


# lgb
from sklearn.cross_validation import StratifiedKFold
skf=list(StratifiedKFold(y_loc_train, n_folds=5, shuffle=True, random_state=2018))
baseloss = []
loss = 0
for i, (train_index, test_index) in enumerate(skf):
    print("Fold", i)
    lgb_model = lgb_clf.fit(X_loc_train[train_index], y_loc_train[train_index],
                          eval_names =['train','valid'],
                          eval_metric='auc',
                          eval_set=[(X_loc_train[train_index], y_loc_train[train_index]), 
                                    (X_loc_train[test_index], y_loc_train[test_index])],early_stopping_rounds=50)
    baseloss.append(lgb_model.best_score_['valid']['auc'])
    loss += lgb_model.best_score_['valid']['auc']
    test_pred= lgb_model.predict_proba(X_loc_test, num_iteration=lgb_model.best_iteration_)[:, 1]
    print('test mean:', test_pred.mean())
    res['prob_%s' % str(i)] = test_pred
print('logloss:', baseloss, loss/5)


# In[ ]:


res['prob'] = 0
for i in range(5):
    res['prob'] += res['prob_%s' % str(i)]
res['prob'] = res['prob']/5


# In[ ]:


mean = res['prob'].mean()
print('mean:',mean)
res[['id', 'prob']].to_csv("./result_sub/lgb1.txt",sep=",",index=False)

