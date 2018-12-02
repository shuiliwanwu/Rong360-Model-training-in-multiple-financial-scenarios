
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


data1=pd.read_csv('result_sub/lgb1.txt')
data2=pd.read_csv('result_sub/lgb2.txt')


# In[ ]:


data1['prob']=0.55*data1['prob']+0.45*data2['prob']


# In[ ]:


data1.to_csv('result_sub/result.txt',sep=",",index=None)

