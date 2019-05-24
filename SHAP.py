#!/usr/bin/env python
# coding: utf-8

# In[49]:


import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
import shap
import pandas as pd
import numpy as np

shap.initjs()

df1=pd.read_csv('train.csv')
df2=pd.read_csv('test.csv')

X=df1[['WiFi density','RSI class','Honk_duration','Timelevel','Intersection density']].values
Y=df1['Class'].values
X_name=['WiFi density','RSI class','Honk_duration','Timelevel','Intersection density']

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

model=XGBClassifier(eta=0.05,n_estimators=121,max_depth=7,min_samples_split=50,min_samples_leaf=5,cv=5)

model.fit(X_train,y_train)
explainer=shap.TreeExplainer(model)

shap_values=explainer.shap_values(X_train)
#print ("a")
#print(shap_values)
#shap_l=shap_values.tolist()
#print(shap_values)
i=0
#while i<3:
    #print(explainer.expected_value[i])
    #print(shap_values[i])
#shap.dependence_plot(0, shap_values[0], X,X_name)
    
  #  shap.force_plot(explainer.expected_value[0],shap_values[0],X_train)
#    i=i+1
shap.summary_plot(shap_values, X,X_name)
#ax.set_yticklabels(['Intersection Density','WiFi Density','Honk_duration','RSI class','Timelevel'])


# In[34]:


print("z")


# In[9]:


import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
import shap
import pandas as pd
import numpy as np

shap.initjs()

df1=pd.read_csv('6mar.csv')

#road.head()
X=df1[['WiFi density','RSI class','Honk_duration','Timelevel','Intersection density']].values
y=df1['Class'].values



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

model=XGBClassifier(eta=0.05,n_estimators=121,max_depth=7,min_samples_split=50,min_samples_leaf=5,cv=5)

model.fit(X_train,y_train)
explainer=shap.TreeExplainer(model)

shap_values=explainer.shap_values(X_train)
print ("a")
#print(shap_values)
#shap_l=shap_values.tolist()
#print(shap_values)
i=0
while i<3:
    #print(explainer.expected_value[i])
    print(shap_values[i])
    print("---------------------------------------")
    print(X[i])
    
    
    
  #  shap.force_plot(explainer.expected_value[0],shap_values[0],X_train)
    i=i+1
    shap.dependence_plot(0, shap_values[0], X,X_name)
#shap.summary_plot(shap_values, X,X_name)
#ax.set_yticklabels(['Intersection Density','WiFi Density','Honk_duration','RSI class','Timelevel'])


# In[5]:


import xgboost
import shap

# load JS visualization code to notebook
shap.initjs()

# train XGBoost model
X,y = shap.datasets.boston()
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

# explain the model's predictions using SHAP values
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
shap.force_plot(explainer.expected_value, shap_values, X)
shap.dependence_plot("RM", shap_values, X)
shap.summary_plot(shap_values, X)


# In[54]:


import xgboost
import shap

# load JS visualization code to notebook
shap.initjs()

# train XGBoost model
X,y = shap.datasets.boston()
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

# explain the model's predictions using SHAP values
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
shap.force_plot(explainer.expected_value, shap_values, X)


# In[10]:


import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
import shap
import pandas as pd
import numpy as np

shap.initjs()

df1=pd.read_csv('6mar.csv')
#df2=pd.read_csv('test.csv')

X=df1[['WiFi density','RSI class','Honk_duration','Timelevel','Intersection density']].values
Y=df1['Class'].values
X_name=['WiFi density','RSI class','Honk_duration','Timelevel','Intersection density']

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

model=XGBClassifier(eta=0.05,n_estimators=121,max_depth=7,min_samples_split=50,min_samples_leaf=5,cv=5)

model.fit(X_train,y_train)
explainer=shap.TreeExplainer(model)

shap_values=explainer.shap_values(X_train)
#print ("a")
#print(shap_values)
#shap_l=shap_values.tolist()
#print(shap_values)
i=0
#while i<3:
    #print(explainer.expected_value[i])
    #print(shap_values[i])
#shap.dependence_plot(0, shap_values[0], X,X_name)
    
  #  shap.force_plot(explainer.expected_value[0],shap_values[0],X_train)
#    i=i+1
shap.summary_plot(shap_values, X,X_name)
#ax.set_yticklabels(['Intersection Density','WiFi Density','Honk_duration','RSI class','Timelevel'])


# In[ ]:




