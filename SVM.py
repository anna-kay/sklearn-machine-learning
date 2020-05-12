#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import numpy as mp
import pandas as pd

from sklearn import svm  
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

minmax_scaler = MinMaxScaler(feature_range=(0,1))


# In[2]:


# Dataset import, train and test definition
covtype= pd.read_table('covtype.data',  sep=',' , header=None)

train=covtype.loc[0:15119, : ]
test=covtype.loc[15120: , : ]


# In[5]:


# Features - target
X_train = train.loc[ : , 0:53]
y_train = train.loc[ : , 54]

X_test = test.loc[ : , 0:53]
y_test = test.loc[ : , 54]


# In[3]:


# Train set sampling (2000 samples)
train_sample=resample(train, n_samples=2000, random_state=0)

X_train_sample=train_sample.loc[ : , 0:53]
y_train_sample=train_sample.loc[ : , 54]


# In[6]:


# Scale features data in [0,1] range
X_train_sample_minmax=minmax_scaler.fit_transform(X_train_sample)

X_train_minmax=minmax_scaler.fit_transform(X_train)

X_test_minmax=minmax_scaler.fit_transform(X_test)


# In[7]:


#-----------------------------------------Erwtima 1-1-----------------------------------------#


# In[8]:


# clf1 : C=1, gamma=0.1

clf1=svm.SVC(C=1, kernel='rbf', gamma=0.1, decision_function_shape='ovo')
scores = cross_val_score(clf1, X_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[9]:


# clf2 : C=4, gamma=0.1

clf2=svm.SVC(C=4, kernel='rbf', gamma=0.1, decision_function_shape='ovo')
scores = cross_val_score(clf2, X_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[10]:


# clf3 : C=4, gamma=1

clf3=svm.SVC(C=4, kernel='rbf', gamma=1, decision_function_shape='ovo')
scores = cross_val_score(clf3, X_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[11]:


# clf4 : C=16, gamma=1

clf4=svm.SVC(C=16, kernel='rbf', gamma=1, decision_function_shape='ovo')
scores = cross_val_score(clf4, X_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[12]:


# clf5 : C=16, gamma=2

clf5=svm.SVC(C=16, kernel='rbf', gamma=2, decision_function_shape='ovo')
scores = cross_val_score(clf5, X_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[13]:


# clf6 : C=32, gamma=2

clf6=svm.SVC(C=32, kernel='rbf', gamma=2, decision_function_shape='ovo')
scores = cross_val_score(clf6, X_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[14]:


# clf7 : C=32, gamma=4

clf7=svm.SVC(C=32, kernel='rbf', gamma=4, decision_function_shape='ovo')
scores = cross_val_score(clf7, X_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[15]:


# clf8 : C=64, gamma=4

clf8=svm.SVC(C=64, kernel='rbf', gamma=4, decision_function_shape='ovo')
scores = cross_val_score(clf8, X_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[16]:


# clf9 : C=128, gamma=4

clf9=svm.SVC(C=128, kernel='rbf', gamma=4, decision_function_shape='ovo')
scores = cross_val_score(clf9, X_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[17]:


# clf10 : C=256,  gamma=4

clf10=svm.SVC(C=256, kernel='rbf', gamma=4, decision_function_shape='ovo')
scores = cross_val_score(clf10, X_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[18]:


# clf11 : C=256, gamma=8

clf11=svm.SVC(C=256, kernel='rbf', gamma=8, decision_function_shape='ovo')
scores = cross_val_score(clf11, X_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[19]:


# clf12 : C=128, gamma=8

clf12=svm.SVC(C=128, kernel='rbf', gamma=8, decision_function_shape='ovo')
scores = cross_val_score(clf12, X_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[ ]:


#-----------------------------------------Erwtima 1-2-----------------------------------------#


# In[21]:


clf7.fit(X_train_minmax, y_train)

y_pred=clf7.predict(X_test_minmax)
accuracy_score(y_test, y_pred)


# In[ ]:


#-----------------------------------------Erwtima 1-3-----------------------------------------#


# In[22]:


# Print classes
print(clf7.classes_)
# Print support vectors per class
print(clf7.n_support_)
# Print sum of support vectors
len(clf7.support_vectors_)

