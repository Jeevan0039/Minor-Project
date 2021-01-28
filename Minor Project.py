#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# # This data is business corrections to previously published it consist of 78 rows and 5 columns in this data we have Series reference, Description, Period,Previously published, Revised

# In[2]:


x = pd.read_csv("C://Users//mogil//Downloads//business-price-indexes-september-2020-quarter-corrections-to-previously-published-statistics.csv")
x


# In[3]:


x.describe()


# In[4]:


x.corr()


# In[5]:


x.isnull().sum()


# In[6]:


x.head()


# In[7]:


x.tail()


# In[8]:


x['% of deviation']=x['Previously published']/x['Revised']*100


# In[9]:


x


# # Here we can see the % of deviation after adding new column

# In[10]:


max_deviation = x.sort_values(by='% of deviation', ascending=False)


# In[11]:


max_deviation.head(10)


# # Here we can see in % of deviation is less.  
# The sectors we can see in Description those data is  updated almost same.
# As we can see by Previously published and Revised in above data.

# In[12]:


max_deviation = x.sort_values(by='% of deviation', ascending=True)


# In[13]:


max_deviation.head(10)


# # Here we can see % of devation is slightly more
# The sectors we can see in Description those data is updated having slightly difference. As we can see by Previously published and Revised in above data.

# In[14]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[16]:


x.plot(kind = 'hist', x='Previously published',y='Revised' ) 


# In[18]:


plt.bar(x['Previously published'], x['Revised']) 
plt.xlabel("Previously published") 
plt.ylabel("Revised") 
plt.show() 


# In[23]:


x.plot.bar() 


# In[ ]:




