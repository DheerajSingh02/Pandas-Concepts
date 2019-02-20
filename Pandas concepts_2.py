
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
from datetime import datetime


# ### read from an Excel file

# In[7]:


file_name_string = "file_path\\wc-20140609-140000.csv"
world_cup_prediction = pd.read_csv(file_name_string)
#world_cup_prediction
world_cup_prediction.head(10)      #Returns first 10 rows of data.


# ### Grouping
# one or more of the following steps:
# - Splitting the data into groups based on some criteria
# - Applying a function to each group independently
# - Combining the results into a data structure

# ### group by country

# In[10]:


world_cup_prediction.groupby('country').sum().head(10)


# ### Categorical Data
# 
# Categoricals are a pandas data type, which correspond to categorical variables in statistics: a variable, which can take
# on only a limited, and usually fixed, number of possible values (categories; levels in R). Examples are gender, social
# class, blood types, country affiliations, observation time or ratings via Likert scales.
# 
# In contrast to statistical categorical variables, categorical data might have an order (e.g. ‘strongly agree’ vs ‘agree’ or
# ‘first observation’ vs. ‘second observation’), but numerical operations (additions, divisions, ...) are not possible.
# 
# All values of categorical data are either in categories or np.nan. Order is defined by the order of categories, not lexical
# order of the values.
# 
# documentation: http://pandas.pydata.org/pandas-docs/stable/categorical.html

# In[12]:


world_cup_prediction['country'] # this is categorical data.


# In[16]:


#world_cup_prediction.groupby('country').count()


# ### Resampling
# documentation: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html
# 
# For arguments to 'freq' parameter, please see [Offset Aliases](http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases)

# Resample time-series data.
# 
# Convenience method for frequency conversion and resampling of time series. Object must have a datetime-like index (DatetimeIndex, PeriodIndex, or TimedeltaIndex), or pass datetime-like values to the on or level keyword.

# In[18]:


# min: minutes
my_index = pd.date_range('9/1/2016', periods=9, freq='min')


# In[19]:


my_index


# create a time series that includes a simple pattern

# In[20]:


my_series = pd.Series(np.arange(9), index=my_index)


# In[21]:


my_series


# Downsample the series into 3 minute bins and sum the values of the timestamps falling into a bin

# In[35]:


my_series.resample('3min').sum()

# There series is divided in interval of 3 minutes. so,
# 0 + 1 + 2 =3
# 3 + 4 + 5 = 12
# 6 + 7 + 8 = 21


# Downsample the series into 3 minute bins as above, but label each bin using the right edge instead of the left
# 
# Notice the difference in the time indices; the sum in each bin is the same

# In[36]:


my_series.resample('3min', label='right').sum()


# Downsample the series into 3 minute bins as above, but close the right side of the bin interval
# 
# "count backwards" from end of time series

# In[37]:


my_series.resample('3min', label='right', closed='right').sum()


# Upsample the series into 30 second bins
# 
# [asfreq()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.asfreq.html)

# In[38]:


#select first 5 rows 
my_series.resample('30S').asfreq()[0:5] 


# ##### define a custom function to use with resampling

# In[39]:


def custom_arithmetic(array_like):
    temp = 3 * np.sum(array_like) + 5
    return temp


# ##### apply custom resampling function

# In[41]:


my_series.resample('3min').apply(custom_arithmetic)


# ### Series
# Series is a one-dimensional labeled array capable of holding any data type (integers, strings, floating point numbers,
# Python objects, etc.). The axis labels are collectively referred to as the index.
# 
# documentation: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html

# ##### Create series from NumPy array
# number of labels in 'index' must be the same as the number of elements in array

# In[43]:


my_simple_series = pd.Series(np.random.randn(8), index=['a', 'b', 'c', 'd', 'e','f','g','h'])
my_simple_series


# In[44]:


my_simple_series.index


# ##### Create series from NumPy array, without explicit index

# In[45]:


my_simple_series = pd.Series(np.random.randn(5))
my_simple_series


# Access a series like a NumPy array

# In[46]:


my_simple_series[:3]


# ##### Create series from Python dictionary

# In[48]:


my_dictionary = {'a' : 45., 'b' : -19.5, 'c' : 4444}
my_second_series = pd.Series(my_dictionary)
my_second_series


# Access a series like a dictionary

# In[49]:


my_second_series['b']


# note order in display; same as order in "index"
# 
# note NaN

# In[50]:


pd.Series(my_dictionary, index=['b', 'c', 'd', 'a'])


# In[51]:


my_second_series.get('a')


# In[52]:


unknown = my_second_series.get('f')
type(unknown)


# ##### Create series from scalar
# If data is a scalar value, an index must be provided. The value will be repeated to match the length of index

# In[54]:


pd.Series(5., index=['a', 'b', 'c', 'd', 'e'])


# <h2> Vectorized operations </h2>

# In[56]:


my_dictionary = {'a' : 45., 'b' : -19.5, 'c' : 4444}
my_series = pd.Series(my_dictionary)
my_series


# ###### add Series without loop

# In[57]:


my_series + my_series


# In[58]:


my_series


# ##### Series within arithmetic expression

# In[59]:


my_series + 5


# ##### Series used as argument to NumPy function

# In[61]:


np.exp(my_series)


# A key difference between Series and ndarray is that operations between Series automatically align the data based on
# label. Thus, you can write computations without giving consideration to whether the Series involved have the same labels.

# In[63]:


my_series[1:]


# In[64]:


my_series[:-1]


# In[65]:


my_series[1:] + my_series[:-1]


# ### Apply Python functions on an element-by-element basis

# In[71]:


def multiply_by_ten (input_element):
    return input_element * 10.0


# In[72]:


my_series.map(multiply_by_ten)


# ### Vectorized string methods
# Series is equipped with a set of string processing methods that make it easy to operate on each element of the array. Perhaps most importantly, these methods exclude missing/NA values automatically. 

# In[75]:


series_of_strings = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])


# In[76]:


series_of_strings.str.lower()


# ### Date Arithmetic
# documentation: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offsets
# 
# | Type      |   | Description                                                       |
# |-----------|---|-------------------------------------------------------------------|
# | date      |   | Store calendar date (year, month, day) using a Gregorian Calendar |
# | datetime  |   | Store both date and time                                          |
# | timedelta |   | Difference between two datetime values                            |
# 
# ##### common date arithmetic operations
# - calculate differences between date
# - generate sequences of dates and time spans
# - convert time series to a particular frequency

# ### Date, time, functions
# documentation: http://pandas.pydata.org/pandas-docs/stable/api.html#top-level-dealing-with-datetimelike
# 
# | to_datetime(*args, **kwargs)                      | Convert argument to datetime.                                               |   |
# |---------------------------------------------------|-----------------------------------------------------------------------------|---|
# | to_timedelta(*args, **kwargs)                     | Convert argument to timedelta                                               |   |
# | date_range([start, end, periods, freq, tz, ...])  | Return a fixed frequency datetime index, with day (calendar) as the default |   |
# | bdate_range([start, end, periods, freq, tz, ...]) | Return a fixed frequency datetime index, with business day as the default   |   |
# | period_range([start, end, periods, freq, name])   | Return a fixed frequency datetime index, with day (calendar) as the default |   |
# | timedelta_range([start, end, periods, freq, ...]) | Return a fixed frequency timedelta index, with day as the default           |   |
# | infer_freq(index[, warn])                         | Infer the most likely frequency given the input index.                      |   |

# In[79]:


now = datetime.now()
now


# In[80]:


now.year, now.month, now.day


# ##### delta
# source: http://pandas.pydata.org/pandas-docs/stable/timedeltas.html

# In[81]:


delta = now - datetime(2001, 1, 1)
delta


# In[82]:


delta.days


# ### Parsing Timedelta
# ##### from string

# In[85]:


pd.Timedelta('4 days 10.15 hours')


# ##### named keyword arguments

# In[87]:


# note: these MUST be specified as keyword arguments
pd.Timedelta(days=1, seconds=1)


# ##### integers with a unit

# In[88]:


pd.Timedelta(1, unit='d')


# ##### create a range of dates from Timedelta

# In[91]:


Indian_independence_day = datetime(2019, 8, 15)
print(Indian_independence_day)
Indian_republic_day = datetime(2019, 1, 26)
print(Indian_republic_day)
summer_time = Indian_independence_day - Indian_republic_day
print(summer_time)
type(summer_time)


# In[92]:


indian_summer_time_range = pd.date_range(Indian_republic_day, periods=summer_time.days, freq='D')


# In[93]:


indian_summer_time_range


# ##### summer_time time series with random data

# In[97]:


indian_summer_time_series = pd.Series(np.random.rand(summer_time.days), index=indian_summer_time_range)
indian_summer_time_series.tail()

