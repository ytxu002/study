#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB


# In[28]:


a = np.array([[0,2,0,0,0],
             [0,2,0,1,0],
             [1,2,0,0,1],
             [2,1,0,0,1],
             [2,0,1,0,1],
             [2,0,1,1,0],
             [1,0,1,1,1],
             [0,1,0,0,0],
             [0,0,1,0,1],
             [2,1,1,0,1],
             [0,1,1,1,1],
             [1,1,0,1,1],
             [1,2,1,0,1]])


# In[29]:


data = pd.DataFrame(a)


# In[30]:


X = data.iloc[:,0:4]
Y = data.iloc[:,4]


# In[31]:


test = np.array([[2,1,0,1]])


# In[32]:


clf = LogisticRegression()
clf.fit(X,Y)
y_pred = clf.predict(test)


# In[33]:


y_pred


# In[34]:


svc = svm.SVC(kernel='linear',C=1,gamma=1)
svc.fit(X,Y)
pred = svc.predict(test)


# In[35]:


pred


# In[37]:


M = MultinomialNB()
M.fit(X,Y)
p = M.predict(test)


# In[38]:


p


# In[ ]:




