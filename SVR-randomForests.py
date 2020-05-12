#!/usr/bin/env python
# coding: utf-8

# In[151]:


#imports

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.metrics import mean_squared_error

MSE=mean_squared_error


# In[2]:


from sklearn.datasets import load_diabetes
diabetes=load_diabetes()
diabetes= pd.DataFrame(data= np.c_[diabetes['data'], diabetes['target']],
                     columns= diabetes['feature_names'] + ['target'])


# In[3]:


X_diabetes = diabetes.iloc[:, 0:10]
Y_diabetes = diabetes.iloc[:, 10]

X_train = X_diabetes.loc[0:299,:]
X_test = X_diabetes.loc[300:442,:]

y_train = Y_diabetes.loc[0:299]
y_test = Y_diabetes.loc[300:442]


# In[ ]:


#---------------------------------------------Erwtima 2.1---------------------------------------------#


# In[21]:


#--------------------------------------------- Linear Models ---------------------------------------------#


# In[113]:


linreg=LinearRegression()


# In[116]:


scores = cross_val_score(linreg, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
scores.mean()


# In[157]:


ridge=Ridge()


# In[158]:


scores = cross_val_score(ridge, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
scores.mean()


# In[159]:


a_range=np.array([x * 0.01 for x in range(1, 20)])
a_scores=[]
for a in a_range:
    scores = cross_val_score(ridge, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    a_scores.append(scores.mean())

print(a_scores)


# In[154]:


lasso=Lasso()


# In[155]:


scores = cross_val_score(lasso, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
scores.mean()


# In[156]:


a_range=np.array([x * 0.01 for x in range(1, 20)])
a_scores=[]
for a in a_range:
    scores = cross_val_score(lasso, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    a_scores.append(scores.mean())

print(a_scores)


# In[31]:


#--------------------------------------------- SVMs ---------------------------------------------#


# In[ ]:


#--------------------------------------------- rbf ---------------------------------------------#


# In[139]:


svm_rbf=svm.SVR(kernel='rbf')


# In[140]:


C_range=[]
gamma_range=[]

for i in range(-5, 16):
    C_range.append(2**i)
    gamma_range.append(2**i)


# In[141]:


param_grid= dict(C=C_range, gamma=gamma_range)
grid_rbf=GridSearchCV(svm_rbf, param_grid, cv=10, scoring="neg_mean_squared_error", n_jobs=-1, return_train_score=True, verbose=1)


# In[142]:


grid_rbf.fit(X_train, y_train)


# In[143]:


print(grid_rbf.best_score_)
print(grid_rbf.best_params_)


# In[ ]:


#--------------------------------------------- poly ---------------------------------------------#


# In[105]:


C_range=[0.1, 1, 10, 100, 1000]
gamma_range=[0.1, 0.01, 0.001, 0.0001, 0.00001]


# In[106]:


svm_poly=svm.SVR(kernel='poly')


# In[107]:


param_grid= dict(C=C_range, gamma=gamma_range)


# In[108]:


grid_poly=GridSearchCV(svm_poly, param_grid, cv=10, scoring="neg_mean_squared_error", n_jobs=-1, return_train_score=True, verbose=1)


# In[109]:


grid_poly.fit(X_train, y_train)


# In[110]:


print(grid_poly.best_score_)
print(grid_poly.best_params_)


# In[111]:


scores = cross_val_score(svm_poly, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
scores.mean()


# In[ ]:


#--------------------------------------------- Random Forests ---------------------------------------------#


# In[121]:


from sklearn.ensemble import RandomForestRegressor


# In[118]:


diabetes


# In[125]:


regressor=RandomForestRegressor(n_jobs=-1)


# In[130]:


regressor.fit(X_train, y_train)


# In[131]:


print(regressor.feature_importances_)


# In[160]:


param_grid= {'max_depth': range(3,7),'n_estimators': (10, 20, 50, 100, 1000)}


# In[161]:


grid_regressor=GridSearchCV(regressor, 
                            param_grid, 
                            cv=10, 
                            scoring="neg_mean_squared_error", 
                            n_jobs=-1, 
                            return_train_score=True, 
                            verbose=1)


# In[162]:


grid_regressor.fit(X_train, y_train)


# In[163]:


print(grid_regressor.best_score_)
print(grid_regressor.best_params_)


# In[138]:


#---------------------------------------------Erwtima 2.2---------------------------------------------#


# In[144]:


y_pred=grid_rbf.predict(X_test)


# In[152]:


MSE(y_test,y_pred)


# In[ ]:


#---------------------------------------------Erwtima 2.3---------------------------------------------#


# In[ ]:


#---------------------------------------------Linear Models---------------------------------------------#


# In[164]:


# Linear Regression
scores = cross_val_score(linreg, X_diabetes, Y_diabetes, cv=10, scoring='neg_mean_squared_error')
scores.mean()


# In[165]:


# Ridge Linear Regression
scores = cross_val_score(ridge, X_diabetes, Y_diabetes, cv=10, scoring='neg_mean_squared_error')
scores.mean()


# In[166]:


# Lasso Linear Regression
scores = cross_val_score(lasso, X_diabetes, Y_diabetes, cv=10, scoring='neg_mean_squared_error')
scores.mean()


# In[ ]:


#---------------------------------------------SVRs---------------------------------------------#


# In[170]:


#---------------------------------------------rbf---------------------------------------------#

C_range=[]
gamma_range=[]

for i in range(-5, 16):
    C_range.append(2**i)
    gamma_range.append(2**i)
    
param_grid= dict(C=C_range, gamma=gamma_range)
grid_rbf=GridSearchCV(svm_rbf, param_grid, cv=10, scoring="neg_mean_squared_error", n_jobs=-1, return_train_score=True, verbose=1)

grid_rbf.fit(X_diabetes, Y_diabetes)

print(grid_rbf.best_score_)
print(grid_rbf.best_params_)


# In[171]:


#---------------------------------------------poly---------------------------------------------#

C_range=[0.1, 1, 10, 100, 1000]
gamma_range=[0.1, 0.01, 0.001, 0.0001, 0.00001]

param_grid= dict(C=C_range, gamma=gamma_range)
grid_poly=GridSearchCV(svm_poly, param_grid, cv=10, scoring="neg_mean_squared_error", n_jobs=-1, return_train_score=True, verbose=1)

grid_poly.fit(X_diabetes, Y_diabetes)

print(grid_poly.best_score_)
print(grid_poly.best_params_)


# In[172]:


#--------------------------------------------- Random Forest Regressor---------------------------------------------#

param_grid= {'max_depth': range(3,7),'n_estimators': (10, 20, 50, 100, 1000)}

grid_regressor=GridSearchCV(regressor, 
                            param_grid, 
                            cv=10, 
                            scoring="neg_mean_squared_error", 
                            n_jobs=-1, 
                            return_train_score=True, 
                            verbose=1)

grid_regressor.fit(X_diabetes, Y_diabetes)

print(grid_regressor.best_score_)
print(grid_regressor.best_params_)

