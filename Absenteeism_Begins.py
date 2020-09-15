#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np


# In[3]:


pd.options.display.max_columns = None
pd.options.display.max_rows = None


# In[7]:


df_preprocessed = pd.read_csv('C:/Users/abey.assefa/AbeyML/csv_Only/Absenteeism_data.csv')


# ### Create target

# In[8]:


df_preprocessed['Absenteeism Time in Hours'].median()


# In[9]:


df_preprocessed['Excessive_Absenteeism'] = np.where(df_preprocessed['Absenteeism Time in Hours'] > 
                                    df_preprocessed['Absenteeism Time in Hours'].median(), 1, 0)


# In[24]:


## check balace of data
df_preprocessed['Excessive_Absenteeism'].sum() / df_preprocessed.shape[0]


# In[25]:


data_with_target = df_preprocessed.drop(['ID', 'Date','Absenteeism Time in Hours'], axis=1)

# In[26]:


data_with_target is df_preprocessed


# In[30]:


# input for regression
unscaled_inputs = data_with_target.iloc[:, :-1]


# In[31]:


from sklearn.preprocessing import StandardScaler


# In[32]:


absenteeism_scale = StandardScaler()


# In[33]:


absenteeism_scale.fit(unscaled_inputs)


# In[34]:


scaled_inputs = absenteeism_scale.transform(unscaled_inputs)


# In[36]:


scaled_inputs.shape


# In[37]:


## split 


# In[39]:


from sklearn.model_selection import train_test_split
target = df_preprocessed['Excessive_Absenteeism']


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(scaled_inputs, target, train_size=0.8, random_state=20)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

