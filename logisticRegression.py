#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# orismos sunartisis upologismou MAPE 
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

mse=mean_squared_error
mae=mean_absolute_error
mape=mean_absolute_percentage_error


# In[2]:


print("------------------------------------------------1.1------------------------------------------------")
concrete= pd.read_csv('Concrete_Data.csv', header=None)

# xwrismos tou dataset set train kai test
X=concrete.loc[:,0:7]
y=concrete.loc[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# orismos tou eurous tis uperparamterou a pou 8a eksetastei, (for range(0, 30000)-> warning)
a=np.array([x * 0.001 for x in range(1, 30000)])

MSE_ridge=np.zeros(a.shape)
MAE_ridge=np.zeros(a.shape)
MAPE_ridge=np.zeros(a.shape)

MSE_lasso=np.zeros(a.shape)
MAE_lasso=np.zeros(a.shape)
MAPE_lasso=np.zeros(a.shape)

# ordinary least squre regression
ols=LinearRegression()
ols.fit(X_train, y_train)
y_pred=ols.predict(X_test)

print('\n--------------------Ordinary Least Square--------------------')
print("MSE= %.4f" % mse(y_test,y_pred))
print("MAE= %.4f" % mae(y_test,y_pred))
print("MAPE= %.4f" % mape(y_test,y_pred))

# upologismos twn metrikwn gia ta ridge kai lasso gia olo to eurow twn timwn tis a
for i in range(len(a)):
    
    ridge=Ridge(alpha=a[i])
    ridge.fit(X_train, y_train)
    y_pred=ridge.predict(X_test)
    MSE_ridge[i]=mse(y_test,y_pred)
    MAE_ridge[i]=mae(y_test,y_pred)
    MAPE_ridge[i]=mape(y_test,y_pred)
    
    lasso=Lasso(alpha=a[i])
    lasso.fit(X_train, y_train)
    y_pred=lasso.predict(X_test)
    MSE_lasso[i]=mse(y_test,y_pred)
    MAE_lasso[i]=mae(y_test,y_pred)
    MAPE_lasso[i]=mape(y_test,y_pred)
    
print('\n----------------------------Ridge----------------------------')    
print("MSE:\tmin= %.4f" % min(MSE_ridge), "\tmax= %.4f" % max(MSE_ridge))
print("MAE:\tmin= %.4f" % min(MAE_ridge), "\tmax= %.4f" % max(MAE_ridge))
print("MAPE:\tmin= %.4f" % min(MAPE_ridge), "\tmax= %.4f" % max(MAPE_ridge))

ax1=plt.subplot(321)
plt.plot(a, MSE_ridge)   
plt.xlabel('alpha')
plt.ylabel('MSE_Ridge')
plt.xlim(0, 30)
plt.ylim(68, 74)

ax2=plt.subplot(323)
plt.plot(a, MAE_ridge)   
plt.xlabel('alpha')
plt.ylabel('MAE_Ridge')
plt.xlim(0, 30)
plt.ylim(6.4,6.8)

ax3=plt.subplot(325)
plt.plot(a, MAPE_ridge)   
plt.xlabel('alpha')
plt.ylabel('MAPE_Ridge')
plt.xlim(0, 30)
plt.ylim(0.25,0.28)

print('\n----------------------------Lasso----------------------------') 
print("MSE:\tmin= %.4f" % min(MSE_lasso), "\tmax= %.4f" % max(MSE_lasso))
print("MAE:\tmin= %.4f" % min(MAE_lasso), "\tmax= %.4f" % max(MAE_lasso))
print("MAPE: \tmin= %.4f" % min(MAPE_lasso), "\tmax= %.4f" % max(MAPE_lasso))

ax4=plt.subplot(322, sharey=ax1)
plt.plot(a, MSE_lasso)   
plt.xlabel('alpha')
plt.ylabel('MSE_Lasso')
plt.xlim(0, 30)
plt.ylim(68, 74)

ax5=plt.subplot(324, sharey=ax2)
plt.plot(a, MAE_lasso)   
plt.xlabel('alpha')
plt.ylabel('MAE_Lasso')
plt.xlim(0, 30)
plt.ylim(6.4,6.8)

ax6=plt.subplot(326, sharey=ax3)
plt.plot(a, MAPE_lasso)   
plt.xlabel('alpha')
plt.ylabel('MAPE_Lasso')
plt.xlim(0, 30)
plt.ylim(0.25,0.28)

plt.subplots_adjust(bottom=0.5, right=1.5, top=2.5, wspace=0.3, hspace=0.3)

plt.savefig('alpha.pdf', bbox_inches='tight')


# In[3]:


print("------------------------------------------------1.3------------------------------------------------")

# Orismos tou tropou diaxwrismou tou dataset se splits
ss = ShuffleSplit(n_splits=10, test_size=0.3)

MSE_ols=[]
MAE_ols=[]
MAPE_ols=[]

MSE_ridge=[]
MAE_ridge=[]
MAPE_ridge=[]

MSE_lasso=[]
MAE_lasso=[]
MAPE_lasso=[]

for train_index, test_index in ss.split(X):
    
    X_train, X_test= X.loc[train_index], X.loc[test_index]
    y_train, y_test= y.loc[train_index], y.loc[test_index]

    ols=LinearRegression()
    ols.fit(X, y)
    y_pred=ols.predict(X_test)

    MSE_ols.append(mse(y_test,y_pred))
    MAE_ols.append(mae(y_test,y_pred))
    MAPE_ols.append(mape(y_test,y_pred))

    ridge=Ridge()
    ridge.fit(X, y)
    y_pred=ridge.predict(X_test)

    MSE_ridge.append(mse(y_test,y_pred))
    MAE_ridge.append(mae(y_test,y_pred))
    MAPE_ridge.append(mape(y_test,y_pred))
        
    lasso=Lasso()
    lasso.fit(X, y)
    y_pred=lasso.predict(X_test)

    MSE_lasso.append(mse(y_test,y_pred))
    MAE_lasso.append(mae(y_test,y_pred))
    MAPE_lasso.append(mape(y_test,y_pred))

print('\n----------------------Ordinary Least Square----------------------')
print("MSE:\t mean_MSE= %.4f" % np.mean(MSE_ols), "\tstd_MSE= %.4f" % np.std(MSE_ols))
print("MAE:\t mean_MSE= %.4f" % np.mean(MAE_ols), "\tstd_MAE= %.4f" % np.std(MAE_ols))
print("MAPE:\t mean_MSE= %.4f"% np.mean(MAPE_ols), "\tstd_MAPE= %.4f" % np.std(MAPE_ols))


print('\n----------------------------Ridge----------------------------') 
print("MSE:\t mean_MSE= %.4f" % np.mean(MSE_ridge), "\tstd_MSE= %.4f" % np.std(MSE_ridge))
print("MAE:\t mean_MSE= %.4f" % np.mean(MAE_ridge), "\tstd_MAE= %.4f" % np.std(MAE_ridge))
print("MAPE:\t mean_MSE= %.4f" % np.mean(MAPE_ridge), "\tstd_MAPE= %.4f" % np.std(MAPE_ridge))


print('\n----------------------------Lasso----------------------------') 
print("MSE:\t mean_MSE= %.4f" % np.mean(MSE_lasso), "\tstd_MSE= %.4f" % np.std(MSE_lasso))
print("MAE:\t mean_MSE= %.4f" % np.mean(MAE_lasso), "\tstd_MAE= %.4f" % np.std(MAE_lasso))
print("MAPE:\t mean_MSE= %.4f" % np.mean(MAPE_lasso), "\tstd_MAPE= %.4f" % np.std(MAPE_lasso))


# In[4]:


print("------------------------------------------------1.4------------------------------------------------")
 
def test_poly_regression(X_train, y_train, X_test, y_test, n):
    
    poly = PolynomialFeatures(n)
    
    # Dimiourgia neou sunolou xaraktiristikwn, arxika xaraktiristika upswmena se dunameis tou n 
    X_train_poly=poly.fit_transform(X_train)
    X_test_poly=poly.fit_transform(X_test)
    
    # Grammiki palindromisi gia to neo sunolo xaraktiristikwn
    ols.fit(X_train_poly, y_train)
    
    # Provlepsi me vasi to kainourio sunolo
    y_pred=ols.predict(X_test_poly)
    
    print("n = ",n, ":")
    print("mse = ",mse(y_test,y_pred))
    print("mae= ",mae(y_test,y_pred))
    print("mape= ",mape(y_test,y_pred))
    print("\n")
    
# Epanalipsi tis paranw diadikasias gia n apo 1 ews 10    
for n in range(1,11):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    test_poly_regression(X_train, y_train, X_test, y_test, n)


# In[5]:


# imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[6]:


covtype= pd.read_table('covtype.data',  sep=',' , header=None)

# xwrismos tou dataset set train kai test
X_train=covtype.loc[0:15119, 0:53]
y_train=covtype.loc[0:15119, 54]

X_test=covtype.loc[15120:, 0:53]
y_test=covtype.loc[15120:, 54]


# In[ ]:


print("------------------------------------------------2.2------------------------------------------------")
print('\n----------------------------Logistic Regression, lbfgs----------------------------')
print("\npenalty='l2', tol=0.001, solver='lbfgs', max_iter=10000")
clf_lbfgs_1=LogisticRegression(penalty='l2', tol=0.001, solver='lbfgs', max_iter=10000, verbose=1).fit(X_train, y_train)
y_pred=clf_lbfgs_1.predict(X_test)
print(accuracy_score(y_pred, y_test))
#55,39

print("\npenalty='l2', tol=0.0001, solver='lbfgs', max_iter=10000")
clf_lbfgs_2=LogisticRegression(penalty='l2', tol=0.0001, solver='lbfgs', max_iter=10000, verbose=1).fit(X_train, y_train)
y_pred=clf_lbfgs_2.predict(X_test)
print(accuracy_score(y_pred, y_test))
#55,39

print("\npenalty='l2', tol=0.001, C=0.5, solver='lbfgs', max_iter=10000")
clf_lbfgs_3=LogisticRegression(penalty='l2', tol=0.001, solver='lbfgs', max_iter=10000, verbose=1).fit(X_train, y_train)
y_pred=clf_lbfgs_3.predict(X_test)
print(accuracy_score(y_pred, y_test))
#55,39

print("\npenalty='l2', tol=0.001, C=10.0, solver='lbfgs', max_iter=10000")
clf_lbfgs_4=LogisticRegression(penalty='l2', tol=0.001, solver='lbfgs', max_iter=10000, verbose=1).fit(X_train, y_train)
y_pred=clf_lbfgs_4.predict(X_test)
print(accuracy_score(y_pred, y_test))
#55,39

print('\n----------------------------Logistic Regression, newton-cg----------------------------')

print("\npenalty='l2', tol=0.001, solver='newton-cg', max_iter=10000")
clf_newton_1=LogisticRegression(penalty='l2', tol=0.001, solver='newton-cg', max_iter=10000, verbose=1).fit(X_train, y_train)
y_pred=clf_newton_1.predict(X_test)
print(accuracy_score(y_pred, y_test))
#56,02%

print("\npenalty='l2', tol=0.0001, solver='newton-cg', max_iter=10000")
clf_newton_2=LogisticRegression(penalty='l2', tol=0.0001, solver='newton-cg', max_iter=10000, verbose=1).fit(X_train, y_train)
y_pred=clf_newton_2.predict(X_test)
print(accuracy_score(y_pred, y_test))
#56,02%

print("\npenalty='l2', tol=0.001, solver='newton-cg', max_iter=100000")
clf_newton_3=LogisticRegression(penalty='l2', tol=0.001, solver='newton-cg', max_iter=100000, verbose=1).fit(X_train, y_train)
y_pred=clf_newton_3.predict(X_test)
print(accuracy_score(y_pred, y_test))
#56,02%

print("\npenalty='l2', tol=0.001, C=0.5, solver='newton-cg', max_iter=10000")
clf_newton_4=LogisticRegression(penalty='l2', tol=0.001, C=0.5, solver='newton-cg', max_iter=10000, verbose=1).fit(X_train, y_train)
y_pred=clf_newton_4.predict(X_test)
print(accuracy_score(y_pred, y_test))
#56,02%

print('\n----------------------------Logistic Regression, saga----------------------------')

print("\npenalty='l2', tol=0.001, solver='saga', max_iter=10000")
clf_saga_1=LogisticRegression(penalty='l2', tol=0.001, solver='saga', max_iter=10000, verbose=1).fit(X_train, y_train)
y_pred=clf_saga_1.predict(X_test)
print(accuracy_score(y_pred, y_test))
#39,70%

#print("penalty='l2', tol=0.0001, solver='saga', max_iter=10000")
#clf_saga_2=LogisticRegression(penalty='l2', tol=0.0001, solver='saga', max_iter=10000, verbose=1).fit(X_train, y_train)
#y_pred=clf_saga_2.predict(X_test)
#print(accuracy_score(y_pred, y_test))
#42,18%

#print("penalty='l2', tol=0.0001, solver='saga', C=0.5, max_iter=10000")
#clf_saga2=LogisticRegression(penalty='l2', tol=0.0001, solver='saga', max_iter=10000, verbose=1).fit(X_train, y_train)
#y_pred=clf_saga2.predict(X_test)
#print(accuracy_score(y_pred, y_test))
#42,18%

print("penalty='l1', tol=0.001, solver='saga', max_iter=100000")
clf_saga2=LogisticRegression(penalty='l2', tol=0.0001, solver='saga', max_iter=10000, verbose=1).fit(X_train, y_train)
y_pred=clf_saga2.predict(X_test)
print(accuracy_score(y_pred, y_test))
#39,70%

print('\n----------------------------Logistic Regression, liblinear----------------------------')

print("\npenalty='l2', tol=0.001, solver='liblinear', max_iter=10000")
clf_liblinear_l2=LogisticRegression(penalty='l2', tol=0.001, solver='liblinear', max_iter=10000, verbose=1).fit(X_train, y_train)
y_pred=clf_liblinear_l2.predict(X_test)
print("\n")
print(accuracy_score(y_pred, y_test))
#42,79%

print("\npenalty='l1', tol=0.001, solver='liblinear', max_iter=10000")
clf_liblinear_l1=LogisticRegression(penalty='l1', tol=0.0001, solver='liblinear', max_iter=10000, verbose=1).fit(X_train, y_train)
y_pred=clf_liblinear_l1.predict(X_test)
print("\n")
print(accuracy_score(y_pred, y_test))
#55,90%


# In[ ]:


print("------------------------------------------------2.3------------------------------------------------")

print('\n----------------------------LDA----------------------------')
clf_lda = LDA()
clf_lda.fit(X_train, y_train)
y_pred=clf_lda.predict(X_test)
print("\n")
print(accuracy_score(y_pred, y_test))
#58,12%
