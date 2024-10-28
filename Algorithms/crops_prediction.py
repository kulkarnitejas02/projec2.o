# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 10:59:11 2024

@author: k_pos
"""

#### importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
### read data
df=pd.read_excel('datas.xlsx')

### load features
x=df.iloc[:,0:6]

###load target
y=df.iloc[:,6]

####split the data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)

### define random forest
model=RandomForestClassifier(n_estimators=50)

###train model
model.fit(x_train,y_train)

### prediction
predict=model.predict(x_test)

## Model name
filename = 'rf_model.sav'
## Save the model
pickle.dump(model, open(filename, 'wb'))