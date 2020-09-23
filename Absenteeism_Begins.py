#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

pd.options.display.max_columns = None
pd.options.display.max_rows = None
df_preprocessed = pd.read_csv('C:/Users/abey.assefa/AbeyML/csv_Only/Absenteeism_data.csv')

# ### Create target
df_preprocessed['Absenteeism Time in Hours'].median()
df_preprocessed['Excessive_Absenteeism'] = np.where(df_preprocessed['Absenteeism Time in Hours'] > 
                                    df_preprocessed['Absenteeism Time in Hours'].median(), 1, 0)

## check balace of data
df_preprocessed['Excessive_Absenteeism'].sum() / df_preprocessed.shape[0]

data_with_target = df_preprocessed.drop(['ID', 'Date','Absenteeism Time in Hours'],axis=1)
# check data is filtered
data_with_target is df_preprocessed

# input for regression
unscaled_inputs = data_with_target.iloc[:, :-1]

from sklearn.preprocessing import StandardScaler
absenteeism_scale = StandardScaler()
absenteeism_scale.fit(unscaled_inputs)
scaled_inputs = absenteeism_scale.transform(unscaled_inputs)

scaled_inputs.shape

## split 
target = df_preprocessed['Excessive_Absenteeism']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_inputs, target, train_size=0.8, random_state=20)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
