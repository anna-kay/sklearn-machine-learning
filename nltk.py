#!/usr/bin/env python
# coding: utf-8

# In[78]:


#imports
import numpy as np
import pandas as pd
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from matplotlib import pyplot as plt
import string

stemmer = PorterStemmer()


# In[79]:


IMDB = pd.read_csv('imdb_labelled.txt', sep="\t", quoting=3, header=None)
AMAZON= pd.read_csv('amazon_cells_labelled.txt', sep="\t", header=None)
YELP= pd.read_csv('yelp_labelled.txt', sep="\t", header=None)

# concatenate imdb, amazon ang yelp data
frames = [IMDB, AMAZON, YELP]
result = pd.concat(frames, ignore_index=True)

# extract separately sentences and scores
sentences=result.loc[:,0]
scores=result.loc[:,1]


# In[80]:


#-----------------------------------------Erwtima 3-1-----------------------------------------#

# words extraction
words=[]
words_temp=[]
for i in range(0, len(sentences)):   
    words_of_sentence=word_tokenize(sentences[i])
    words_temp.append(words_of_sentence)

words=[item for sublist in words_temp for item in sublist]

# stemming
stemmed_words = [stemmer.stem(word) for word in words]

# removing stop words
stop_words = set(stopwords.words('english'))
words = [word for word in stemmed_words if not word in stop_words]

# removing punctuation
words = [''.join(c for c in s if c not in string.punctuation) for s in words]

# removing empty strings
words =  [word for word in words if word]

# turning all words into lowercase
words = [w.lower() for w in words]

# removing duplicates
words = list(set(words))

# sorting list
words.sort()


# In[81]:


len(words)


# In[82]:


#-----------------------------------------Erwtima 3-2-----------------------------------------#

# initializing array that will contain all sentence vectors
sentence_vector=np.zeros((len(sentences), len(words)), dtype=int)

for s in range(0, len(sentences)):
    # turning the sentence lowercase
    sentence = sentences[s].lower()
    # removing punctuation
    sentence="".join(l for l in sentence if l not in string.punctuation)
    # words extraction
    words_of_sentence=word_tokenize(sentence)
    # stemming
    words_of_sentence_stemmed=[stemmer.stem(word) for word in words_of_sentence]
    # removing stop words
    words_of_sentence_filtered = [word for word in words_of_sentence_stemmed if not word in stop_words]
       
    # turning the list of the words in the sentence into a vector
    for i in range(0, len(words_of_sentence_filtered)):
        for j in range(0, len(words)):
            if (words_of_sentence_filtered[i]==words[j]):
                sentence_vector[s,j]=sentence_vector[s,j]+1


# In[83]:


#-----------------------------------------Erwtima 3-3-----------------------------------------#

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# In[84]:


# X -> sentence_vector
# y -> scores

X_train, X_test, y_train, y_test = train_test_split(sentence_vector, scores, test_size=0.25)


# In[85]:


clf=svm.SVC(kernel='linear', decision_function_shape='ovo')
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
accuracy_score(y_test, y_pred)


# In[ ]:


# tuning hyperparameter C


# In[69]:


from sklearn.model_selection import cross_val_score


# In[74]:


C_range=np.array([x * 0.1 for x in range(2, 10)])
C_scores=[]
for c in C_range:
    clf=svm.SVC(C=c, kernel='linear', decision_function_shape='ovo')
    scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
    C_scores.append(scores.mean())


# In[75]:


print(C_scores)


# In[76]:


plt.plot(C_range, C_scores)
plt.xlabel("Value C for SVM")
plt.ylabel("Cross-validated Accuracy")


# In[86]:


X_train, X_test, y_train, y_test = train_test_split(sentence_vector, scores, test_size=0.25)


# In[91]:


clf=svm.SVC(C=0.5, kernel='linear', decision_function_shape='ovo')
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
accuracy_score(y_test, y_pred)


# In[ ]:




