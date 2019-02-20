
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd

import numpy as np   # import numpy for later use in this tutorial.
import matplotlib.pyplot as plt

pd.__version__


# Pandas objects: Pandas objects can be thought of as enhanced versions of NumPy structured array in which rows and columns are identified with labels rather than simple integer indices.
# 
# Three fundamental Pandas data structures: the Series, DataFrame, and Index.

# <h2>The Pandas Series Object</h2>
# A Pandas Series is a one-dimensional array of indexed data. It can be created from a list or array.

# In[4]:


data = pd.Series([1,0.5,0.25,0.125])
data


# In[5]:


print(data.values)
print(data.index)


# In[6]:


# accessing the value-using index 
data[1]


# In[7]:


#data[5]   # it will give an error because index 5 is not present in the data.


# In[8]:


data[1:3] #index starts from zero. Here values for index 1 and index 2 are selected. Note that end index is not inclusive while selecting values.


# In[9]:


# Data can be scalar, which is repeated to fill the specified index.
pd.Series(5, index=[100,200,300])


# In[10]:


#pd.Series([5,6], index=[100,200,300]) # it will give and error because length of passed values is not same as no. of indices.


# In[11]:


# Series creation using dictionary:
pd.Series({2:'a',1:'b',3:'c'})


# In[12]:


# Index can be explicitly set if different result is preferred.
pd.Series({2:'a',1:'b',3:'c'}, index=[3,2])


# <h2>The Pandas DataFrame Object</h2>
# 
# DataFrame is an analog of a two-dimensional array with both flexible row indices and flexible column names.
# 
# Therefore, we can them as generalization of a NumPy array or as a specialization of a Python dictionary.

# In[13]:


# Let's create two dictionary objects:
population_dict = {'California': 38332521,
'Texas': 26448193,
'New York': 19651127,
'Florida': 19552860,
'Illinois': 12882135}

area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
'Florida': 170312, 'Illinois': 149995}


# In[14]:


# Now let's create series objects from dictionaries:

area = pd.Series(area_dict)
population = pd.Series(population_dict)

#area
#population


# In[15]:


# We can now create a DataFrame using both series objects:
states = pd.DataFrame({'population':population, 'area':area})

states


# In[16]:


print("Indices are:", states.index)
print("Columns are:", states.columns)


# <h2>Constructing DataFrame objects</h2>

# In[17]:


# From a single Series object:

pd.DataFrame(population, columns=['population'])


# In[18]:


# From a list of dicts:

data = [{'a': i, 'b': 2*i} for i in range(3)]
data


# In[19]:


pd.DataFrame(data)


# In[20]:


# Even if some keys in the dictionary are missing, Pandas will fill them in with NaN (i.e.,
#“not a number”) values:

pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])


# In[21]:


# From a dictionary of Series objects

pd.DataFrame({'population':population,
             'area':area})


# In[22]:


# From a two-dimensional NumPy array

pd.DataFrame(np.random.rand(3,2),
            columns = ['x','y'],
            index= ['a','b','c'])


# In[23]:


# From a NumPy structure array
A = np.zeros(3, dtype=[('A','i8'),('B','f8')])
A


# In[24]:


pd.DataFrame(A)


# In[25]:


#pd.Index?


# In[26]:


ind = pd.Index([2,3,4,13,5,7,11])
ind


# In[27]:


print(ind[1])
print(ind[::2])


# In[28]:


print(ind.size, ind.shape, ind.ndim, ind.dtype)


# In[29]:


# Note: Index objects are immutable while NumPy arrays are mutable.
# ind[1] = 0    # it will give an error: TypeError: Index does not support mutable operations


# In[30]:


indA = pd.Index([1,3,5,7,9])
indB = pd.Index([2,3,5,7,11])


# In[31]:


indA & indB            #intersection


# In[32]:


indA | indB           #union


# In[33]:


indA ^ indB           #symmetric difference


# In[34]:


# Other way to get above results through object methods:
indA.intersection(indB)


# <h2>Selection</h2>

# In[35]:


sample_numpy_data = np.array(np.arange(24)).reshape((6,4))
sample_numpy_data


# In[36]:


dates_index = pd.date_range('20160101', periods=6)
dates_index


# In[37]:


sample_df = pd.DataFrame(sample_numpy_data, index= dates_index, columns=list('ABCD'))
sample_df


# <h5>Selection using column name</h5>

# In[38]:


sample_df['C']


# <h5>Selection using slice</h5>
# 
# - note: last index is not included

# In[39]:


sample_df[1:4]          


# <h5>Selection using date time index</h5>
# - note: last index is included

# In[40]:


sample_df['2016-01-01':'2016-01-04']


# <h2>Selection by label</h2>
# 
# label-location based indexer for selection by label

# In[41]:


sample_df.loc[dates_index[1:3]]


# <h5>Selecting using multi-axis by label</h5>

# In[42]:


sample_df.loc[:, ['A','B']]


# <h5>Label slicing, both endpoints are included</h5>

# In[43]:


sample_df.loc['2016-01-01':'2016-01-03', ['A','B']]


# <h5>Reduce number of dimensions for returned object
# - notice order of 'D' and 'B'

# In[44]:


sample_df.loc['2016-01-03', ['D', 'B']]


# In[45]:


sample_df.loc['2016-01-03', ['D', 'B']][0] * 4


# <h5>Select a scalar</h5>

# In[46]:


sample_df.loc[dates_index[2],'C']


# <h2>Selection by Position</h2>
# integer-location based indexing for selection by position

# In[47]:


sample_df


# In[48]:


sample_numpy_data


# In[49]:


sample_numpy_data[3]


# In[50]:


sample_df.iloc[3]


# <h5>integer slices</h5>

# In[51]:


sample_df.iloc[1:3, 2:4]        #two rows, two columns, last index value not included in each case.


# In[52]:


#Note:
sample_df.iloc[1:13, 2:10]  #note that indices 13,10 are not available in rows and columns, but it doesn't 
#throw any error. from first position to last index available, values are selected. 


# <h5>list of integers</h5>

# In[53]:


sample_df.iloc[[0,1,3],[0,2]] #select first(index 0), second(index 1) and fourth(index 3) rows and first(index 0) and third(index 2) columns. 


# <h5>slicing rows explicitly</h5>
# implicitly selecting all columns

# In[54]:


sample_df.iloc[1:3,:]


# <h5>slicing columns explicitly</h5>
# implicitly selecting all rows

# In[55]:


sample_df.iloc[:, 1:3]


# ### NumPy Universal Functions
# 
# If the data within a DataFrame are numeric, NumPy's universal functions can be used on/with the DataFrame.

# In[88]:


df = pd.DataFrame(np.random.randn(10, 4), columns=['A', 'B', 'C', 'D'])
df2 = pd.DataFrame(np.random.randn(7, 3), columns=['A', 'B', 'C'])
sum_df = df + df2
sum_df


# ##### NaN are handled correctly by universal function

# In[89]:


np.exp(sum_df)


# ##### Transpose availabe T attribute

# In[90]:


sum_df.T


# In[91]:


np.transpose(sum_df.values)


# ##### dot method on DataFrame implements matrix multiplication
# Note: row and column headers

# In[93]:


A_df = pd.DataFrame(np.arange(15).reshape((3,5)))
B_df = pd.DataFrame(np.arange(10).reshape((5,2)))
A_df.dot(B_df)


# ##### dot method on Series implements dot product

# In[94]:


C_Series = pd.Series(np.arange(5,10))
C_Series.dot(C_Series)


# ### dictionary like operations
# ##### dictionary selection with string index

# In[95]:


cookbook_df = pd.DataFrame({'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]})
cookbook_df['BBB']


# ##### arithmetic vectorized operation using string indices

# In[96]:


cookbook_df['BBB'] * cookbook_df['CCC']


# ##### column deletion 

# In[97]:


del cookbook_df['BBB']
cookbook_df


# In[98]:


last_column = cookbook_df.pop('CCC')
last_column


# In[99]:


cookbook_df


# ##### add a new column using a Python list

# In[100]:


cookbook_df['DDD'] = [32, 21, 43, 'hike']
cookbook_df


# ##### insert function
# documentation: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.insert.html

# In[101]:


cookbook_df.insert(1, "new column", [3,4,5,6])
cookbook_df


# <h2>Boolean Indexing</h2>
# test based upon one column's data

# In[56]:


(sample_df['C']) >= 14

#sample_df.C >= 14                #both are equivalent.
#type((sample_df['C']) >= 14)     #pandas.core.series.Series   


# In[57]:


(sample_df[['C','A']]) >= 14
#type((sample_df[['C','A']]) >= 14) #pandas.core.frame.DataFrame


# <h5>test based upon entire data set </h5>

# In[58]:


sample_df[sample_df >= 11]


# <h5>isin() method
# ----------------------------
# Returns a boolean Series showing whether each element in the Series is exactly contained in the passed sequence of values.

# In[59]:


sample_df_2 = sample_df.copy()
sample_df_2['Fruits'] = ['apple', 'orange','banana','strawberry','blueberry','pineapple']
sample_df_2


# select rows where 'Fruits' column contains either 'banana' or 'pineapple'; notice 'smoothy', which is not in the column

# In[60]:


sample_df_2[sample_df_2['Fruits'].isin(['banana','pineapple', 'smoothy'])]


# <h2>Assignment Statements</h2>

# In[61]:


sample_series = pd.Series([1,2,3,4,5,6], index=pd.date_range('2016-01-01', periods=6))
sample_series


# In[62]:


sample_df_2['Extra Data'] = sample_series*3 + 1
sample_df_2


# <h5>Setting values by label</h5>

# In[63]:


sample_df_2.at[dates_index[3], 'Fruits'] = 'pear'
sample_df_2


# <h5>Setting values by position</h5>

# In[64]:


sample_df_2.iat[3,2] = 4444           #assign the value at position (3,2) i.e at intersection of 4th column and 3rd row.
sample_df_2                              


# <h5>Setting by assigning with a numpy array</h5>

# In[65]:


second_numpy_array = np.array(np.arange(len(sample_df_2))) * 100 + 7
second_numpy_array


# In[66]:


sample_df_2['G'] = second_numpy_array
sample_df_2


# <h2>Missing Data</h2>
# 
# pandas uses np.nan to represent missing data.Bye default, n.nan is not included in computations.
# 
# nan represents - 'not a number'

# In[67]:


browser_index = ['Firefox', 'Chrome', 'Safari', 'IE10', 'Konqueror']

browser_df = pd.DataFrame({
    'http_status': [200,200,404,404,301],
    'response_time': [0.04, 0.02, 0.07, 0.08, 1.0]},
    index = browser_index)
browser_df


# <h5>reindex() create a copy(not a view)</h5>

# In[68]:


new_index= ['Safari', 'Iceweasel', 'Comodo Dragon', 'IE10', 'Chrome']
browser_df_2 = browser_df.reindex(new_index)
browser_df_2


# <h5>drop rows that have missing data</h5>

# In[69]:


browser_df_3 = browser_df_2.dropna(how='any')
browser_df_3


# <h5>fill-in missing data</h5>

# In[70]:


browser_df_2.fillna(value=0.001)


# <h5>get boolean mask where values are nan</h5>

# In[71]:


pd.isnull(browser_df_2)


# <h5>NaN propagates during arithmetic operations</h5>

# In[72]:


browser_df_2 * 10


# <h2>Operations</h2>

# ### descriptive statistics

# In[73]:


pd.set_option('display.precision', 2)
sample_df_2.describe()


# ##### column mean

# In[74]:


sample_df_2.mean()


# ##### row mean

# In[75]:


sample_df_2.mean(1)  # 1 specifies mean value along rows.


# ### apply(a function to a data frame)

# In[76]:


sample_df_2.apply(np.cumsum) # values are cumulatively added from previous row value to current row value until last row.


# In[77]:


sample_df_2


# #### string methods

# In[78]:


s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower()


# In[79]:


s.str.len()


# # Merge
# - Concat 
# - Join   
# - Append 

# In[80]:


second_df_2 = sample_df.copy()
sample_df_2['Fruits'] = ['apple', 'orange','banana','strawberry','blueberry','pineapple']

sample_series = pd.Series([1,2,3,4,5,6], index=pd.date_range('2016-01-01', periods=6))
sample_df_2['Extra Data'] = sample_series *3 +1

second_numpy_array = np.array(np.arange(len(sample_df_2)))  *100 + 7
sample_df_2['G'] = second_numpy_array

sample_df_2


# ### concat()
# ##### separate data frame into a list with 3 elements

# In[81]:


pieces = [sample_df_2[:2], sample_df_2[2:4], sample_df_2[4:]]
pieces[0]


# In[82]:


pieces              # list elements


# ##### concatenate first and last elements

# In[83]:


new_list = pieces[0], pieces[2]
pd.concat(new_list)


# ### append()

# In[84]:


new_last_row = sample_df_2.iloc[2]
new_last_row


# In[85]:


sample_df_2.append(new_last_row)


# ### merge()
# Merge DataFrame objects by performing a database-style join operation by columns or indexes.
# 
# If joining columns on columns, the DataFrame indexes will be ignored. Otherwise if joining indexes on indexes or indexes on a column or columns, the index will be passed on.

# In[86]:


left = pd.DataFrame({'my_key': ['K0', 'K1', 'K2', 'K3'],
 'A': ['A0', 'A1', 'A2', 'A3'],
 'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'my_key': ['K0', 'K1', 'K2', 'K3'],
 'C': ['C0', 'C1', 'C2', 'C3'],
 'D': ['D0', 'D1', 'D2', 'D3']})

left


# In[87]:


result = pd.merge(left, right, on='my_key')
result

