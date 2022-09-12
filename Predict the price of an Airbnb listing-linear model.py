#!/usr/bin/env python
# coding: utf-8

# Context
# Airbnb, Inc. is an American vacation rental online marketplace company based in San Francisco, California, United States. Airbnb offers arrangements for lodging, primarily homestays, or tourism experiences. The company does not own any of the real estate listings, nor does it host events; it acts as a broker, receiving commissions from each booking. Reference
# 
# Since 2008, guests and hosts have used Airbnb to travel in a more unique, personalized way.
# 
# Objective
# Imagine you are Data Scientist who would help find the price for lodging or homestays based on different attributes mentioned in their 
# 
# listings. Oh wait, what are listings? Listings can include written descriptions, photographs with captions, and a user profile where potential guests can get to know a bit about the hosts.And you are given the listings of one of the most popular cities in central Europe: Amsterdam.
# 
# Now your job is to build a machine learning model that will automatically predict the price for lodging or homestays.
# 
# 
# 
# Link to Dataset: https://www.kaggle.com/c/dphi-amsterdam-airbnb-data/data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
import seaborn as sns
print("Done")


# In[2]:


df=pd.read_csv('data/airbnb_listing_train.csv')
df.head(5)


# In[3]:


df.isnull().sum()


# In[4]:


df.describe()


# In[5]:


df.info


# In[6]:


df['price'].value_counts()


# In[ ]:





# In[7]:


plt.figure(figsize=(15,6))
sns.countplot('price',data=df.head(100))
plt.xticks(rotation=90)
plt.show()


# In[8]:


sns.lmplot(x="number_of_reviews",y="price",data=df,order=2,ci=None)


# In[9]:


X=np.array(df['price']).reshape(-1,1)
y=np.array(df['number_of_reviews']).reshape(-1,1)

df.dropna(inplace=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

lg=LinearRegression() 
lg.fit(X_train,y_train)
print(lg.score(X_test,y_test))


# In[10]:


df.fillna(method='ffill',inplace=True)


# In[11]:


y_pred=lg.predict(X_test)
plt.scatter(X_test,y_test,color='b')
plt.plot(X_test,y_pred,color='k')
plt.show()


# In[ ]:




