
# coding: utf-8

# In[116]:


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

# In[3]:


data = pd.Series([1,0.5,0.25,0.125])
data


# In[4]:


print(data.values)
print(data.index)


# In[5]:


# accessing the value-using index 
data[1]


# In[6]:


#data[5]   # it will give an error because index 5 is not present in the data.


# In[7]:


data[1:3] #index starts from zero. Here values for index 1 and index 2 are selected. Note that end index is not inclusive while selecting values.


# In[8]:


# Data can be scalar, which is repeated to fill the specified index.
pd.Series(5, index=[100,200,300])


# In[9]:


#pd.Series([5,6], index=[100,200,300]) # it will give and error because length of passed values is not same as no. of indices.


# In[10]:


# Series creation using dictionary:
pd.Series({2:'a',1:'b',3:'c'})


# In[11]:


# Index can be explicitly set if different result is preferred.
pd.Series({2:'a',1:'b',3:'c'}, index=[3,2])


# <h2>The Pandas DataFrame Object</h2>
# 
# DataFrame is an analog of a two-dimensional array with both flexible row indices and flexible column names.
# 
# Therefore, we can them as generalization of a NumPy array or as a specialization of a Python dictionary.

# In[12]:


# Let's create two dictionary objects:
population_dict = {'California': 38332521,
'Texas': 26448193,
'New York': 19651127,
'Florida': 19552860,
'Illinois': 12882135}

area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
'Florida': 170312, 'Illinois': 149995}


# In[13]:


# Now let's create series objects from dictionaries:

area = pd.Series(area_dict)
population = pd.Series(population_dict)

#area
#population


# In[14]:


# We can now create a DataFrame using both series objects:
states = pd.DataFrame({'population':population, 'area':area})

states


# In[15]:


print("Indices are:", states.index)
print("Columns are:", states.columns)


# <h2>Constructing DataFrame objects</h2>

# In[16]:


# From a single Series object:

pd.DataFrame(population, columns=['population'])


# In[17]:


# From a list of dicts:

data = [{'a': i, 'b': 2*i} for i in range(3)]
data


# In[18]:


pd.DataFrame(data)


# In[19]:


# Even if some keys in the dictionary are missing, Pandas will fill them in with NaN (i.e.,
#“not a number”) values:

pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])


# In[20]:


# From a dictionary of Series objects

pd.DataFrame({'population':population,
             'area':area})


# In[21]:


# From a two-dimensional NumPy array

pd.DataFrame(np.random.rand(3,2),
            columns = ['x','y'],
            index= ['a','b','c'])


# In[22]:


# From a NumPy structure array
A = np.zeros(3, dtype=[('A','i8'),('B','f8')])
A


# In[23]:


pd.DataFrame(A)


# In[24]:


#pd.Index?


# In[25]:


ind = pd.Index([2,3,4,13,5,7,11])
ind


# In[26]:


print(ind[1])
print(ind[::2])


# In[27]:


print(ind.size, ind.shape, ind.ndim, ind.dtype)


# In[28]:


# Note: Index objects are immutable while NumPy arrays are mutable.
# ind[1] = 0    # it will give an error: TypeError: Index does not support mutable operations


# In[29]:


indA = pd.Index([1,3,5,7,9])
indB = pd.Index([2,3,5,7,11])


# In[30]:


indA & indB            #intersection


# In[31]:


indA | indB           #union


# In[32]:


indA ^ indB           #symmetric difference


# In[33]:


# Other way to get above results through object methods:
indA.intersection(indB)


# <h2>Selection</h2>

# In[66]:


sample_numpy_data = np.array(np.arange(24)).reshape((6,4))
sample_numpy_data


# In[37]:


dates_index = pd.date_range('20160101', periods=6)
dates_index


# In[40]:


sample_df = pd.DataFrame(sample_numpy_data, index= dates_index, columns=list('ABCD'))
sample_df


# <h5>Selection using column name</h5>

# In[45]:


sample_df['C']


# <h5>Selection using slice</h5>
# 
# - note: last index is not included

# In[47]:


sample_df[1:4]          


# <h5>Selection using date time index</h5>
# - note: last index is included

# In[48]:


sample_df['2016-01-01':'2016-01-04']


# <h2>Selection by label</h2>
# 
# label-location based indexer for selection by label

# In[51]:


sample_df.loc[dates_index[1:3]]


# <h5>Selecting using multi-axis by label</h5>

# In[53]:


sample_df.loc[:, ['A','B']]


# <h5>Label slicing, both endpoints are included</h5>

# In[55]:


sample_df.loc['2016-01-01':'2016-01-03', ['A','B']]


# <h5>Reduce number of dimensions for returned object
# - notice order of 'D' and 'B'

# In[58]:


sample_df.loc['2016-01-03', ['D', 'B']]


# In[60]:


sample_df.loc['2016-01-03', ['D', 'B']][0] * 4


# <h5>Select a scalar</h5>

# In[63]:


sample_df.loc[dates_index[2],'C']


# <h2>Selection by Position</h2>
# integer-location based indexing for selection by position

# In[64]:


sample_df


# In[67]:


sample_numpy_data


# In[68]:


sample_numpy_data[3]


# In[69]:


sample_df.iloc[3]


# <h5>integer slices</h5>

# In[73]:


sample_df.iloc[1:3, 2:4]        #two rows, two columns, last index value not included in each case.


# In[75]:


#Note:
sample_df.iloc[1:13, 2:10]  #note that indices 13,10 are not available in rows and columns, but it doesn't 
#throw any error. from first position to last index available, values are selected. 


# <h5>list of integers</h5>

# In[80]:


sample_df.iloc[[0,1,3],[0,2]] #select first(index 0), second(index 1) and fourth(index 3) rows and first(index 0) and third(index 2) columns. 


# <h5>slicing rows explicitly</h5>
# implicitly selecting all columns

# In[82]:


sample_df.iloc[1:3,:]


# <h5>slicing columns explicitly</h5>
# implicitly selecting all rows

# In[83]:


sample_df.iloc[:, 1:3]


# <h2>Boolean Indexing</h2>
# test based upon one column's data

# In[109]:


(sample_df['C']) >= 14

#sample_df.C >= 14                #both are equivalent.
#type((sample_df['C']) >= 14)     #pandas.core.series.Series   


# In[108]:


(sample_df[['C','A']]) >= 14
#type((sample_df[['C','A']]) >= 14) #pandas.core.frame.DataFrame


# <h5>test based upon entire data set </h5>

# In[112]:


sample_df[sample_df >= 11]


# <h5>isin() method
# ----------------------------
# Returns a boolean Series showing whether each element in the Series is exactly contained in the passed sequence of values.

# In[113]:


sample_df_2 = sample_df.copy()
sample_df_2['Fruits'] = ['apple', 'orange','banana','strawberry','blueberry','pineapple']
sample_df_2


# select rows where 'Fruits' column contains either 'banana' or 'pineapple'; notice 'smoothy', which is not in the column

# In[114]:


sample_df_2[sample_df_2['Fruits'].isin(['banana','pineapple', 'smoothy'])]


# <h2>Assignment Statements</h2>

# In[125]:


sample_series = pd.Series([1,2,3,4,5,6], index=pd.date_range('2016-01-01', periods=6))
sample_series


# In[126]:


sample_df_2['Extra Data'] = sample_series*3 + 1
sample_df_2


# <h5>Setting values by label</h5>

# In[129]:


sample_df_2.at[dates_index[3], 'Fruits'] = 'pear'
sample_df_2


# <h5>Setting values by position</h5>

# In[132]:


sample_df_2.iat[3,2] = 4444           #assign the value at position (3,2) i.e at intersection of 4th column and 3rd row.
sample_df_2                              


# <h5>Setting by assigning with a numpy array</h5>

# In[133]:


second_numpy_array = np.array(np.arange(len(sample_df_2))) * 100 + 7
second_numpy_array


# In[134]:


sample_df_2['G'] = second_numpy_array
sample_df_2


# <h2>Missing Data</h2>
# 
# pandas uses np.nan to represent missing data.Bye default, n.nan is not included in computations.
# 
# nan represents - 'not a number'

# In[136]:


browser_index = ['Firefox', 'Chrome', 'Safari', 'IE10', 'Konqueror']

browser_df = pd.DataFrame({
    'http_status': [200,200,404,404,301],
    'response_time': [0.04, 0.02, 0.07, 0.08, 1.0]},
    index = browser_index)
browser_df


# <h5>reindex() create a copy(not a view)</h5>

# In[138]:


new_index= ['Safari', 'Iceweasel', 'Comodo Dragon', 'IE10', 'Chrome']
browser_df_2 = browser_df.reindex(new_index)
browser_df_2


# <h5>drop rows that have missing data</h5>

# In[140]:


browser_df_3 = browser_df_2.dropna(how='any')
browser_df_3


# <h5>fill-in missing data</h5>

# In[142]:


browser_df_2.fillna(value=0.001)


# <h5>get boolean mask where values are nan</h5>

# In[144]:


pd.isnull(browser_df_2)


# <h5>NaN propagates during arithmetic operations</h5>

# In[147]:


browser_df_2 * 10


# <h2>Operations</h2>

# ### descriptive statistics

# In[150]:


pd.set_option('display.precision', 2)
sample_df_2.describe()


# ##### column mean

# In[152]:


sample_df_2.mean()


# ##### row mean

# In[154]:


sample_df_2.mean(1)  # 1 specifies mean value along rows.


# ### apply(a function to a data frame)

# In[160]:


sample_df_2.apply(np.cumsum) # values are cumulatively added from previous row value to current row value until last row.


# In[161]:


sample_df_2


# #### string methods

# In[162]:


s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower()


# In[164]:


s.str.len()


# # Merge
# - Concat 
# - Join   
# - Append 

# In[168]:


second_df_2 = sample_df.copy()
sample_df_2['Fruits'] = ['apple', 'orange','banana','strawberry','blueberry','pineapple']

sample_series = pd.Series([1,2,3,4,5,6], index=pd.date_range('2016-01-01', periods=6))
sample_df_2['Extra Data'] = sample_series *3 +1

second_numpy_array = np.array(np.arange(len(sample_df_2)))  *100 + 7
sample_df_2['G'] = second_numpy_array

sample_df_2


# ### concat()
# ##### separate data frame into a list with 3 elements

# In[175]:


pieces = [sample_df_2[:2], sample_df_2[2:4], sample_df_2[4:]]
pieces[0]


# In[176]:


pieces              # list elements


# ##### concatenate first and last elements

# In[179]:


new_list = pieces[0], pieces[2]
pd.concat(new_list)


# ### append()

# In[180]:


new_last_row = sample_df_2.iloc[2]
new_last_row


# In[182]:


sample_df_2.append(new_last_row)


# ### merge()
# Merge DataFrame objects by performing a database-style join operation by columns or indexes.
# 
# If joining columns on columns, the DataFrame indexes will be ignored. Otherwise if joining indexes on indexes or indexes on a column or columns, the index will be passed on.

# In[188]:


left = pd.DataFrame({'my_key': ['K0', 'K1', 'K2', 'K3'],
 'A': ['A0', 'A1', 'A2', 'A3'],
 'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'my_key': ['K0', 'K1', 'K2', 'K3'],
 'C': ['C0', 'C1', 'C2', 'C3'],
 'D': ['D0', 'D1', 'D2', 'D3']})

left


# In[190]:


result = pd.merge(left, right, on='my_key')
result

