# Minor-Project
import pandas as pd
import numpy as np

# This data is business corrections to previously published it consist of 78 rows and 5 columns it consist of variables like Series reference, Description, Period,Previously published, Revised¶

x = pd.read_csv("C://Users//mogil//Downloads//business-price-indexes-september-2020-quarter-corrections-to-previously-published-statistics.csv")
x

x.describe()

x.corr()

x.isnull().sum()

x.head()

x.tail()

x['% of deviation']=x['Previously published']/x['Revised']*100

x

# Here we can see the % of deviation after adding new column

max_deviation = x.sort_values(by='% of deviation', ascending=False)

max_deviation.head(10)

# Here we can see in % of deviation is less.  
The sectors we can see in Description those data is  updated almost same.
As we can see by Previously published and Revised in above data.

max_deviation = x.sort_values(by='% of deviation', ascending=True)

max_deviation.head(10)

# Here we can see % of devation is slightly more
The sectors we can see in Description those data is updated having slightly difference. As we can see by Previously published and Revised in above data.

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

x.plot(kind = 'hist', x='Previously published',y='Revised' ) 

plt.bar(x['Previously published'], x['Revised']) 
plt.xlabel("Previously published") 
plt.ylabel("Revised") 
plt.show() 

x.plot.bar() 

