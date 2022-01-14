#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#sourced from https://medium.com/bigdatarepublic/contextual-changepoint-detection-with-python-and-r-using-rpy2-fa7d86259ba9


# In[31]:


#importing data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/sarahcliff/Desktop/hydrology data/deseasoned data/ombh2detrended@2022-01-12.csv')
data = np.array(df['Data'])
time = np.array(df['Date'])


# In[32]:


#C. Truong, L. Oudre, N. Vayatis. Selective review of offline change point detection methods. Signal Processing, 167:107299, 2020. [journal] [pdf]
#to cite the ruptures package


# In[36]:


import ruptures as rpt

#Dynamic programming method
model = "l1"  
algo = rpt.Dynp(model=model, min_size=3, jump=5).fit(data)
my_bkps = algo.predict(n_bkps=10)
rpt.show.display(data, my_bkps, figsize=(5, 3))
plt.ylabel('Depth')
plt.xlabel('Number of Data Points')
plt.title('Change Point Detection: Dynamic Programming Search Method: OMBH2')

#RUPTURES PACKAGE
#Changepoint detection with the Pelt search method
model="rbf"
algo = rpt.Pelt(model=model).fit(data)
result = algo.predict(pen=10)
rpt.display(data, result, figsize=(5, 3))
plt.ylabel('Depth')
plt.xlabel('Number of Data Points')
plt.title('Change Point Detection: Pelt Search Method: OMBH2')
    
#Changepoint detection with the Binary Segmentation search method
model = "l2"  
algo = rpt.Binseg(model=model).fit(data)
my_bkps = algo.predict(n_bkps=10)
# show results
rpt.show.display(data, my_bkps, figsize=(5, 3))
plt.ylabel('Depth')
plt.xlabel('Number of Data Points')
plt.title('Change Point Detection: Binary Segmentation Search Method: OMBH2')
    
#Changepoint detection with window-based search method
model = "l2"  
algo = rpt.Window(width=40, model=model).fit(data)
my_bkps = algo.predict(n_bkps=10)
rpt.show.display(data, my_bkps, figsize=(5, 3))
plt.ylabel('Depth')
plt.xlabel('Number of Data Points')
plt.title('Change Point Detection: Window-Based Search Method: OMBH2')

plt.show()
    


# In[30]:


#changefinder method

import changefinder

f, (ax1, ax2) = plt.subplots(2, 1)
f.subplots_adjust(hspace=0.4)
ax1.plot(data)
ax1.set_title("data point")

#Initiate changefinder function
cf = changefinder.ChangeFinder()
scores = [cf.update(p) for p in data]
ax2.plot(scores)
ax2.set_title("anomaly score")
plt.show()


# In[ ]:




