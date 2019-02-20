
# coding: utf-8

# ### Inline Plotting
# [matplotlib](http://matplotlib.org/index.html)
# 
# [pandas visualization](http://pandas.pydata.org/pandas-docs/stable/visualization.html)
# 
# [ggplot stylesheet](http://matplotlib.org/examples/style_sheets/plot_ggplot.html)

# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

mu, sigma = 100, 15
data_set = mu + sigma*np.random.randn(10000)

# the histogram of the data

n, bins, patches = plt.hist(data_set, 50, normed =1, facecolor= 'g', alpha=0.75)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, 0.025, r'$\mu=100,\ \sigma=15$')
plt.axis([40,160,0,0.03])
plt.grid(True)
plt.show()


# <h3>Available Colors:</h3>
# <pre>
# <code>
# code       color
# ________________________
# &#39;k&#39;         black
# &#39;b&#39;         blue
# &#39;c&#39;         cyan
# &#39;g&#39;         green
# &#39;m&#39;         magenta
# &#39;r&#39;         red
# &#39;w&#39;         white
# &#39;y&#39;         yellow
# </code>
# </pre>

# ### Figures and Subplots
# - figure - container thats holds all elements of plot(s)
# - subplot - appears within a rectangular grid within a figure

# In[4]:


my_first_figure = plt.figure("My First Figure")


# In[5]:


subplot_1 = my_first_figure.add_subplot(2, 3, 1)
subplot_6 = my_first_figure.add_subplot(2, 3, 6)


# In[6]:


plt.plot(np.random.rand(50).cumsum(), 'k--')
plt.show()


# In[7]:


subplot_2 = my_first_figure.add_subplot(2, 3, 2)
plt.plot(np.random.rand(50), 'go')


# <h1>Multiples Lines, Single Plot</h1>

# <h2>random low and high temperature data</h2>

# In[15]:


data_set_size = 15
low_mu, low_sigma = 50, 4.3
low_data_set = low_mu + low_sigma*np.random.rand(data_set_size)
high_mu, high_sigma = 57, 5.2
high_data_set = high_mu + high_sigma*np.random.rand(data_set_size)

days = list(range(1, data_set_size+1))


# In[21]:


plt.plot(days, low_data_set)


# In[24]:


plt.plot(days, low_data_set,
        days, low_data_set, "vm",
        days, high_data_set,
        days, high_data_set, "^k")


# In[25]:


plt.plot(days, high_data_set)


# In[26]:


plt.plot(days, high_data_set, "^k")


# In[27]:


plt.plot(days, low_data_set,
         days, low_data_set, "vm",
         days, high_data_set, 
         days, high_data_set, "^k")
plt.xlabel('Day')
plt.ylabel('Temperature: degrees Farenheit')
plt.title('Randomized temperature data')
plt.show()


# In[28]:


plt.plot(days, low_data_set,
         
         days, high_data_set
         )
plt.xlabel('Day')
plt.ylabel('Temperature: degrees Farenheit')
plt.title('Randomized temperature data')
plt.show()


# In[29]:


plt.plot(
         days, high_data_set, "^k")
plt.xlabel('Day')
plt.ylabel('Temperature: degrees Farenheit')
plt.title('Randomized temperature data')
plt.show()


# Next example from: http://matplotlib.org/examples/pylab_examples/legend_demo2.html

# In[31]:


t1 = np.arange(0.0, 2.0, 0.1)
t2 = np.arange(0.0, 2.0, 0.01)

# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1, = plt.plot(t2, np.exp(-t2))
l2, l3 = plt.plot(t2, np.sin(2 * np.pi * t2), '--go', t1, np.log(1 + t1), '.')
l4, = plt.plot(t2, np.exp(-t2) * np.sin(2 * np.pi * t2), 'rs-.')

plt.legend((l2, l4), ('oscillatory', 'damped'), loc='upper right', shadow=True)
plt.xlabel('time')
plt.ylabel('volts')
plt.title('Damped oscillation')
plt.show()


# <h1>Tick Marks, Labels, and Grids</h1>

# In[32]:


number_of_data_points = 1000

my_figure = plt.figure()
subplot_1 = my_figure.add_subplot(1, 1, 1)
my_data_set = np.random.rand(number_of_data_points).cumsum()
subplot_1.plot(np.random.rand(number_of_data_points).cumsum())

number_of_ticks = 5
ticks = np.arange(0, number_of_data_points, number_of_data_points//number_of_ticks)
subplot_1.set_xticks(ticks)

labels = subplot_1.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation=45, fontsize='small')

subplot_1.set_title ("My First Ticked Plot")
subplot_1.set_xlabel ("Groups")

subplot_1.grid(True)
gridlines = subplot_1.get_xgridlines() + subplot_1.get_ygridlines()
for line in gridlines:
    line.set_linestyle(':')

plt.show()


# Line styles for grid lines
# <list>
#     <li>-    solid line</li>
#     <li>--   dashed line</li>
#     <li>-.   dash dot line</li>
#     <li>:    dotted</li>
# </list>
# 
# More information about lines available at: http://matplotlib.org/api/lines_api.html

# <h1>Plot Annotations</h1>

# In[33]:


number_of_data_points = 10

my_figure = plt.figure()
subplot_1 = my_figure.add_subplot(1, 1, 1)
subplot_1.plot(np.random.rand(number_of_data_points).cumsum())

subplot_1.text (1, 0.5, r'an equation: $E=mc^2$', fontsize=18, color='red')
subplot_1.text (1, 1.5, "Hello, Mountain Climbing!", family='monospace', fontsize=14, color='green')

# see: http://matplotlib.org/users/transforms_tutorial.html
# transform=subplot_1.transAxes; entire axis between zero and one
subplot_1.text(0.5, 0.5, "We are centered, now", transform=subplot_1.transAxes)

subplot_1.annotate('shoot arrow', xy=(2, 1), xytext=(3, 4),
            arrowprops=dict(facecolor='red', shrink=0.05))

plt.show()


# In[34]:


x = np.arange(0, 10, 0.005)
y = np.exp(-x/2.) * np.sin(2*np.pi*x)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)
ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)

plt.show()


# In[35]:



x = np.arange(0, 10, 0.005)
y = np.exp(-x/2.) * np.sin(2*np.pi*x)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)
ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)

xdata, ydata = 5, 0
xdisplay, ydisplay = ax.transData.transform_point((xdata, ydata))

bbox = dict(boxstyle="round", fc="0.8")
arrowprops = dict(
    arrowstyle = "->",
    connectionstyle = "angle,angleA=0,angleB=90,rad=10")

offset = 72
ax.annotate('data = (%.1f, %.1f)'%(xdata, ydata),
            (xdata, ydata), xytext=(-2*offset, offset), textcoords='offset points',
            bbox=bbox, arrowprops=arrowprops)


disp = ax.annotate('display = (%.1f, %.1f)'%(xdisplay, ydisplay),
            (xdisplay, ydisplay), xytext=(0.5*offset, -offset),
            xycoords='figure pixels',
            textcoords='offset points',
            bbox=bbox, arrowprops=arrowprops)


plt.show()


# In[36]:


fig = plt.figure()
for i, label in enumerate(('A', 'B', 'C', 'D')):
    ax = fig.add_subplot(2,2,i+1)
    ax.text(0.05, 0.95, label, transform=ax.transAxes,
      fontsize=16, fontweight='bold', va='top')

plt.show()


# ### Data Frame Plots
# documentation: http://pandas.pydata.org/pandas-docs/stable/visualization.html

# The plot method on Series and DataFrame is just a simple wrapper around plt.plot()
# 
# If the index consists of dates, it calls gcf().autofmt_xdate() to try to format the x-axis nicely as show in the plot window.

# In[37]:


ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
plt.show()  


# On DataFrame, plot() is a convenience to plot all of the columns, and include a legend within the plot.

# In[38]:


df = pd.DataFrame(np.random.randn(1000, 4), index=pd.date_range('1/1/2016', periods=1000), columns=list('ABCD'))
df = df.cumsum()
plt.figure()
df.plot()
plt.show()


# You can plot one column versus another using the x and y keywords in plot():

# In[40]:


df3 = pd.DataFrame(np.random.randn(1000, 2), columns=['B', 'C']).cumsum()
df3['A'] = pd.Series(list(range(len(df))))
df3.plot(x='A', y='B')
plt.show()


# In[41]:


df3.tail()


# ### Plots other than line plots
# Plotting methods allow for a handful of plot styles other than the default Line plot. These methods can be provided as the kind keyword argument to plot(). These include:
# 
# - ‘bar’ or ‘barh’ for bar plots
# - ‘hist’ for histogram
# - ‘box’ for boxplot
# - ‘kde’ or 'density' for density plots
# - ‘area’ for area plots
# - ‘scatter’ for scatter plots
# - ‘hexbin’ for hexagonal bin plots
# - ‘pie’ for pie plots
# 
# For example, a bar plot can be created the following way:

# In[43]:


plt.figure()
df.ix[5].plot(kind='bar')
plt.axhline(0, color='k')
plt.show()


# In[45]:


df.iloc[5]


# ### stack bar chart

# In[46]:


df2 = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
df2.plot.bar(stacked=True)
plt.show()


# ### horizontal bar chart

# In[47]:


df2.plot.barh(stacked=True)
plt.show()


# ### box plot

# In[48]:


df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
df.plot.box()
plt.show()


# ### area plot

# In[49]:


df = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
df.plot.area()
plt.show()


# ### Plotting with Missing Data
# Pandas tries to be pragmatic about plotting DataFrames or Series that contain missing data. Missing values are dropped, left out, or filled depending on the plot type.
# 
# | Plot Type      | NaN Handling            |   |
# |----------------|-------------------------|---|
# | Line           | Leave gaps at NaNs      |   |
# | Line (stacked) | Fill 0’s                |   |
# | Bar            | Fill 0’s                |   |
# | Scatter        | Drop NaNs               |   |
# | Histogram      | Drop NaNs (column-wise) |   |
# | Box            | Drop NaNs (column-wise) |   |
# | Area           | Fill 0’s                |   |
# | KDE            | Drop NaNs (column-wise) |   |
# | Hexbin         | Drop NaNs               |   |
# | Pie            | Fill 0’s                |   |
# 
# If any of these defaults are not what you want, or if you want to be explicit about how missing values are handled, consider using fillna() or dropna() before plotting.

# ### density plot

# In[51]:


ser = pd.Series(np.random.randn(1000))
ser.plot.kde()
plt.show()


# ### lag plot
# Lag plots are used to check if a data set or time series is random. Random data should not exhibit any structure in the lag plot. Non-random structure implies that the underlying data are not random.

# In[52]:


from pandas.tools.plotting import lag_plot
plt.figure()
data = pd.Series(0.1 * np.random.rand(1000) + 0.9 * np.sin(np.linspace(-99 * np.pi, 99 * np.pi, num=1000)))
lag_plot(data)
plt.show()


# ### matplotlib gallery
# documentation: http://matplotlib.org/gallery.html
