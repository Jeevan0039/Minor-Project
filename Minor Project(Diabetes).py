#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
from sklearn import preprocessing


# In[2]:


df=pd.read_csv("C://Users//mogil//Downloads//diabetes.csv")


# In[3]:


df


# Diabetes data set It consist of variables like Pregnancies, Glucose,  
# Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree, Function, Age, Outcome.

# # Outcome is the target variable. Data has 768 rows and 9 columns.

# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.dtypes


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


df.describe()


# In[10]:


df.corr()


# In[11]:


features = df.columns[:8]
for feature in features:
    print(feature, np.count_nonzero(df[feature]==0))


# In[12]:


drop_Glu=df.index[df.Glucose == 0].tolist()
drop_BP=df.index[df.BloodPressure == 0].tolist()
drop_Skin = df.index[df.SkinThickness==0].tolist()
drop_Ins = df.index[df.Insulin==0].tolist()
drop_BMI = df.index[df.BMI==0].tolist()
c=drop_Glu+drop_BP+drop_Skin+drop_Ins+drop_BMI
df1=df.drop(df.index[c])


# In[13]:


df1


# In[14]:


plt.hist(df1['Outcome'])


# In[15]:


dia1 = df1[df.Outcome==1]
dia0 = df1[df.Outcome==0]
dia1.shape


# In[16]:


dia0.shape


# In[17]:


plt.subplot(1,1,1)
sns.distplot(dia0.Pregnancies,kde=False,color="Blue", label="Preg for Outome=0")
sns.distplot(dia1.Pregnancies,kde=False,color = "Gold", label = "Preg for Outcome=1")
plt.title("Histograms for Preg by Outcome")
plt.legend()


# In[18]:


plt.subplot(1,1,1)
sns.distplot(dia0.Glucose,kde=False,color="Gold", label="Gluc for Outcome=0")
sns.distplot(dia1.Glucose, kde=False, color="Blue", label = "Gloc for Outcome=1")
plt.title("Histograms for Glucose by Outcome")
plt.legend()


# In[19]:


plt.subplot(1,1,1)
sns.distplot(dia0.BloodPressure,kde=False,color="Gold",label="BP for Outcome=0")
sns.distplot(dia1.BloodPressure,kde=False, color="Blue", label="BP for Outcome=1")
plt.legend()


# In[20]:


plt.subplot(1,1,1)
sns.distplot(dia0.SkinThickness, kde=False, color="Gold", label="SkinThick for Outcome=0")
sns.distplot(dia1.SkinThickness, kde=False, color="Blue", label="SkinThick for Outcome=1")
plt.legend()


# In[21]:


plt.title("Histogram of Insulin")
plt.subplot(1,1,1)
sns.distplot(dia0.Insulin,kde=False, color="Gold", label="Insulin for Outcome=0")
sns.distplot(dia1.Insulin,kde=False, color="Blue", label="Insuline for Outcome=1")
plt.title("Histogram for Insulin by Outcome")
plt.legend()


# In[22]:


plt.title("Histogram for Age")
plt.subplot(1,1,1)
sns.distplot(dia0.Age,kde=False,color="Gold", label="Age for Outcome=0")
sns.distplot(dia1.Age,kde=False, color="Blue", label="Age for Outcome=1")
plt.legend()


# In[23]:


sns.pairplot(df1, vars=["Pregnancies", "Glucose","BloodPressure","SkinThickness","Insulin", "BMI","DiabetesPedigreeFunction", "Age"],hue="Outcome")
plt.title("Pairplot of Variables by Outcome")


# In[24]:


df1.hist(figsize=(10, 8))
plt.show()


# In[25]:


corr = df1[df1.columns].corr()
sns.heatmap(corr, annot = True)
plt.show()


# In[26]:


X = df1.drop('Outcome', axis=1)
y = df1['Outcome']


# In[27]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 


# In[28]:


from sklearn.linear_model import LogisticRegression   
  
classifier = LogisticRegression(random_state = 0) 
classifier.fit(X_train, y_train) 


# In[29]:


y_pred = classifier.predict(X_test)


# In[30]:


y_pred


# In[31]:


from sklearn.metrics import confusion_matrix 
  
cm = confusion_matrix(y_test, y_pred)


# In[32]:


cm


# In[ ]:




