#!/usr/bin/env python
# coding: utf-8

# In[1]:


#NEEDS REFERENCING!


# In[4]:


import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from datetime import datetime
import scipy.optimize
import scipy.fftpack
from scipy.fft import fft, fftfreq


# In[5]:


#useful functions

def interpolate(vector):
    for i in range(1, len(vector)):
        if np.isnan(vector[i]):
            vector[i] = vector[i-1] + (vector[i+1] - vector[i-1])/(2)
    return vector 

def pad(data):
    bad_indexes = np.isnan(data)
    good_indexes = np.logical_not(bad_indexes)
    good_data = data[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    data[bad_indexes] = interpolated
    return data

def degrade(time_steps, vector):
    newvector = []
    for i in range(0, len(vector)-1):
        if i % time_steps == 0:
            newvector.append(vector[i])
    return newvector

def to_datetime(dt):
    datetimevec= []
    for i in range (0,len(dt)):
        if isinstance(dt[i], str) == True:
            datetime_object = datetime.strptime(dt[i],'%d/%m/%Y %H:%S')
            str_time = datetime.strftime(datetime_object, '%d/%m/%Y' )
            datetimevec.append(datetime_object)
    return datetimevec

#if object is not in string format
def to_datetime_rd(dt):
    datetimevec= []
    for i in range (0,len(dt)):
        datetime_object = datetime.strptime(str(dt[i]),'%Y-%M-%d %H:%S')
        str_time = datetime.strftime(datetime_object, '%d/%m/%Y' )
        datetimevec.append(str_time)
    return datetimevec

def to_date(dt):
    datetimevec= []
    for i in range (0,len(dt)):
        if isinstance(dt[i], str) == True:
            datetime_object = datetime.strptime(dt[i],'%Y-%m-%d')
            str_time = datetime.strftime(datetime_object, '%d/%m/%Y' )
            datetimevec.append(datetime_object)
    return datetimevec

def to_date1(dt):
    datetimevec= []
    for i in range (0,len(dt)):
        if isinstance(dt[i], str) == True:
            datetime_object = datetime.strptime(dt[i],'%m/%d/%Y')
            str_time = datetime.strftime(datetime_object, '%d/%m/%Y' )
            datetimevec.append(datetime_object)
    return datetimevec


# In[6]:


#linear test
def chsquare_lin(x,y,yerr, initial_values):
    def linear_model(x, param_vals):
        return param_vals[0] + param_vals[1]*x
    def chi_squared(model_params, model, x, y, yerr):
        return np.sum((y - model(x, model_params))/yerr)**2
    #deg_freedom = y.size - initial_values.size # Make sure you understand why!
    fit = scipy.optimize.minimize(chi_squared, initial_values, args=(linear_model, x,y, yerr))
    print(fit)
    a_solution = fit.x[0]
    b_solution = fit.x[1]
    
    plt.figure(figsize=(8,6))
    plt.errorbar(x, 
             y, 
             yerr, 
             marker='o', 
             linestyle='None')

    plt.xlabel('x data (units)') # Axis labels
    plt.ylabel('y data (units)')

# Generate best fit line using model function and best fit parameters, and add to plot
    fit_line = linear_model(x, [a_solution, b_solution])
    plt.plot(x, fit_line, 'r')
    plt.show()
    
    line = []
    for i in range(0, len(ds)):
        line.append(a_solution + b_solution*i)

    return line


# In[118]:


#for later hydrology data

df = pd.read_excel('/Users/sarahcliff/Desktop/hydrology data/recent water table data/bolton fell water table 2019-2020.xlsx', sheet_name = 'DBH8', header = 9)
interim = to_datetime_rd(df['Date and Time'])
date = interim[26:-1]
Depth =df['Depth (mAOD)'][26:-1]

d = {'date': date, 'depth': Depth}
df_pad = pd.DataFrame(data=d)


# In[54]:


#For MODIS and sentinel and early hydrology data

df = pd.read_csv('/Users/sarahcliff/Desktop/MODIS data/Raw MODIS/MOD13A2_1_km_16_days_EVI_British National Grid_569000.0_349000.0_2009-01-08_2020-12-31.csv')
date = to_date(df['time'])
Depth = pad(np.array(df['1_km_16_days_EVI']))
d = {'date': date, 'depth': Depth}
df_pad = pd.DataFrame(data=d)
print(df_pad)


# In[41]:


#For early hydrology data
df = pd.read_csv('/Users/sarahcliff/Desktop/hydrology data/raw data/ds1MODIS2021-12-02.csv')
date = to_datetime(df['date'])
Depth = pad(np.array(df['Depth'])[27:-1])
print(len(date), len(Depth))
Depth = degrade(48, Depth)
date = degrade(48,date)
d = {'date': date, 'depth': Depth}
df_pad = pd.DataFrame(data=d)

print(df_pad)


# In[55]:


def set_date_index(input_df, col_name):
    """Given a pandas df, parse and set date column to index.
        col_name will be removed and set as datetime index.

    Args:
        input_df (pandas dataframe): Original pandas dataframe
        col_name (string): Name of date column

    Returns:
        pandas dataframe: modified and sorted dataframe
    """
    # Copy df to prevent changing original
    modified_df = input_df.copy()

    # Infer datetime from col
    modified_df[col_name] = pd.to_datetime(modified_df[col_name])

    # Sort and set index
    modified_df.sort_values(col_name, inplace=True)
    modified_df.set_index(col_name, inplace=True)

    return modified_df


# In[56]:


def combine_seasonal_cols(input_df, seasonal_model_results):
    """Adds inplace new seasonal cols to df given seasonal results

    Args:
        input_df (pandas dataframe)
        seasonal_model_results (statsmodels DecomposeResult object)
    """
    # Add results to original df
    input_df['observed'] = seasonal_model_results.observed
    #input_df['residual'] = seasonal_model_results.resid
    input_df['seasonal'] = seasonal_model_results.seasonal
    #input_df['trend'] = seasonal_model_results.trend


# In[58]:


# Seasonal decompose
df_new = set_date_index(df_pad, 'date')
sd = seasonal_decompose(df_new, period=23)
combine_seasonal_cols(df_new, sd)
print(df_new)


# In[59]:


plt.plot(date, df_new['seasonal'])
plt.plot(date, df_new['observed'])


# In[60]:


test = df_new['observed'] - df_new['seasonal']
plt.plot(date, test)


# In[61]:


date_np = np.array(date)
data_np = np.array(test)


# In[62]:


def days_since(vec):
    days_since = []
    for i in range(0, len(vec)):
        days_since.append((i+1))
    return days_since

ds = np.array(days_since(date_np))


# In[64]:


#fft

yf =  2.0/len(data_np) * np.abs(fft(data_np)[:len(data_np)//2])[1:]
xf = np.linspace(0.0, len(ds)/(2.0*(ds[-1]-ds[0])), len(ds)//2)[1:]

plt.plot(np.abs(xf), np.abs(yf))
plt.title('FFT after detrending - EVI')
plt.show()
max_freq_1=xf[np.argmax(yf)]
power_1 = yf[np.argmax(yf)]
days_max_1 = 1/(max_freq_1) 
print('Period =',days_max_1, 'days')


# In[51]:


#Testing for linear trends

mean = []
variance = []
for i in range(0, len(test)):
    mean.append(np.mean(test[0:i]))
    variance.append(np.var(test[0:i]))
    
plt.plot(ds, mean)
plt.title('mean')
plt.show()
plt.plot(ds, variance)
plt.title('variance')

#Dickey Fuller test

#dickey fuller 
X = data_np
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
    
#Kwiatkowski-Phillips-Schmidt-Shin test
result_kpss = kpss(data_np, regression = 'c', nlags = 'auto')
print('KPSS Statistic: %f' % result_kpss[0])
print('p-value: %f' % result_kpss[1])
print('Critical Values:')
for key, value in result_kpss[3].items():
	print('\t%s: %.3f' % (key, value))


# In[65]:


data=test

error = np.std(data)/np.sqrt(len(data_np))
errorv = np.linspace(error,error,len(data_np))
print((errorv))
print(type(ds))


# In[161]:


line = chsquare_lin(ds, data_np, errorv,[-230,0])


# In[162]:


fin_data = data - line
plt.plot(ds, fin_data)


# In[177]:


data = {'Date': date_np, 'Data': data_np}  
df_data = pd.DataFrame(data)
#creating filename
filename='ombh2detrended@'+ str(datetime.now().strftime("%Y-%m-%d"))+'.csv'
df_data.to_csv(filename)


# In[ ]:




