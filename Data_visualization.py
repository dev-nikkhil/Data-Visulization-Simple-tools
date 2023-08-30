#!/usr/bin/env python
# coding: utf-8

# # Data Visualization - Matplotlib & Seaborn

# In[1]:


# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Generate or load the dataset
x = np.linspace(0, 10, 100)  # 100 points between 0 and 10
y = np.sin(x)  


# * The np.linspace() function then divides the specified range (from 0 to 10) into 100 equally spaced points, and these points are stored in the array x. So, x will contain 100 values ranging from 0 to 10, with each value being evenly spaced.
# * This is often used in various applications like creating a range of x-values for plotting graphs, evaluating functions over a specified range, or generating data for analysis.
# * New array y that will store the sine values corresponding to each element in the array x. The np.sin() function in NumPy is used to compute the sine of an array of values. It takes an array (or a single value) as input and returns an array of the same shape with the sine values.

# In[2]:


# Create a basic line plot
plt.plot(x, y)


# In[3]:


# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Sine Curve Plot')


# In[4]:


# Add legend
plt.legend()


# In[5]:


# Customize appearance
plt.plot(x, y, label='Sine Curve', color='purple', linestyle='', marker='*')


# In[6]:


# Display the plot
plt.show()


# In[7]:


import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a basic line plot
plt.plot(x, y, label='Sine Curve')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Sine Curve Plot')

# Add legend
plt.legend()

# Customize appearance
plt.plot(x, y, label='Sine Curve', color='blue', linestyle='-', marker='o')

# Display the plot
plt.show()


# # Perform various types of plots using Matplotlib

# In[8]:


import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
x = np.linspace(0, 10, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)
y_exp = np.exp(x)
y_random = np.random.rand(100)


# * Just like with the sine function, np.cos() computes the cosine of an array of values and returns an array of the same shape with the cosine values. The resulting array y_cos will contain the cosine values corresponding to each element in the array x.
# 
# * The exponential function np.exp() calculates the value of e raised to the power of the input. The resulting array y_exp will contain the exponential values corresponding to each element in the array x.
# 
# *  y_random that contains 100 random numbers drawn from a uniform distribution between 0 and 1. The np.random.rand() function from NumPy generates random numbers in the specified shape. In this case, it generates an array of shape (100,), which means it will contain 100 random values.
# 

# In[9]:


# Line Plot
plt.figure(figsize=(20,10))
plt.plot(x, y_sin, label='Sine Curve', color='blue')
plt.plot(x, y_cos, label='Cosine Curve', color='red')
#x_ticks = x
#plt.xticks(x_ticks)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot')
plt.legend()
plt.grid(True)
plt.show()


# * The line plt.grid(True) is a Matplotlib function call that adds grid lines to the current plot. This function is used to enhance the readability of a plot by providing a grid background that helps align the data points and makes it easier to estimate values.

# In[10]:


# Scatter Plot
plt.figure(figsize=(10, 5))
plt.scatter(x, y_random, label='Random Data', color='green', marker='o')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot')
plt.legend()
plt.grid(True)
plt.show()


# In[11]:


# Bar Plot
categories = ['A', 'B', 'C', 'D', 'E']
values = [15, 24, 12, 32, 8]
plt.figure(figsize=(8, 5))
plt.bar(categories, values, color='purple')
plt.xlabel('Values')
plt.ylabel('Categories')
plt.title('Bar Plot')
plt.grid(True)
plt.show()


# In[12]:


# Histogram
plt.figure(figsize=(8, 5))
plt.hist(y_random, bins=20, color='orange', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.grid(True)
plt.show()


# In[13]:


# Pie Chart
labels = ['Apples', 'Bananas', 'Oranges', 'Grapes']
sizes = [35, 20, 25, 20]
colors = ['red', 'yellow', 'orange', 'purple']
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[14]:


# Subplots
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, y_sin, label='Sine Curve', color='blue')
plt.title('Sine Curve')

plt.subplot(2, 2, 2)
plt.plot(x, y_cos, label='Cosine Curve', color='red')
plt.title('Cosine Curve')

plt.subplot(2, 2, 3)
plt.scatter(x, y_random, label='Random Data', color='green', marker='o')
plt.title('Scatter Plot')

categories = ['A', 'B', 'C', 'D', 'E']
values = [15, 24, 12, 32, 8]

plt.subplot(2, 2, 4)
plt.bar(categories, values, color='purple')
plt.title('Bar Plot')

plt.tight_layout()
plt.show()


# # Data Visualization - Seaborn

# In[15]:


import numpy as np
import seaborn as sns


# In[16]:


# Generate sample data
x = np.linspace(0, 10, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)
y_random = np.random.rand(100)


# In[17]:


sns.set_style("whitegrid")  # Apply a style to the plots


# In[18]:


import matplotlib.pyplot as plt
# Line Plot
plt.figure(figsize=(10, 5))
sns.lineplot(x=x, y=y_sin, label='Sine Curve')
sns.lineplot(x=x, y=y_cos, label='Cosine Curve')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot')
plt.legend()
plt.show()


# In[19]:


# Scatter Plot
plt.figure(figsize=(10, 5))
sns.scatterplot(x=x, y=y_random, label='Random Data')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot')
plt.legend()
plt.show()


# In[20]:


# Bar Plot
categories = ['A', 'B', 'C', 'D', 'E']
values = [15, 24, 12, 32, 8]
plt.figure(figsize=(8, 5))
sns.barplot(x=categories, y=values)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Plot')
plt.show()


# In[21]:


# Histogram
plt.figure(figsize=(8, 5))
sns.histplot(y_random, bins=20, kde=True)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()


# In[22]:


# Pair Plot
data = sns.load_dataset("iris")
sns.pairplot(data, hue="species")
plt.show()


# # Difference between Matplotlib and Seaborn libraries
# ## 1. Abstraction Level:
#         Matplotlib - Low level
#         Seaborn - High level interface
# ## 2. Default Styles:
#         Matplotlib - lack some aesthetics; required additional styles
#         Seaborn - built in styles and color palettes; additional customization
# ## 3. Plot Types:
#         Matplotlib - basic (line & scatter) to complex (heatmaps & contour)
#         Seaborn - strong in statistical visualization (violin, pair, distribution)
# ## 4. Data Handling: 
#         Matplotlib - arrays and lists
#         Seaborn - pandas dataframes (convinent for data analysis and visualization)
# ## 5. Complexity:
#         Matplotlib - suitable for both simple and complex plots
#         Seaborn - designed to simplify the creation of complex plots

# # Working with matplotlib and seaborn using a dataset

# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[24]:


# Load the Iris dataset
iris = sns.load_dataset("iris")


# In[25]:


# Set style for Seaborn plots
sns.set(style="whitegrid")


# In[26]:


# Pair Plot using Seaborn
sns.pairplot(iris, hue="species", markers=["o", "s", "D"])
plt.title("Pair Plot of Iris Dataset")
plt.show()


# In[27]:


# Box Plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x="species", y="sepal_length", data=iris)
plt.title("Box Plot of Sepal Length by Species")
plt.show()


# In[28]:


# Violin Plot using Seaborn
plt.figure(figsize=(10, 6))
sns.violinplot(x="species", y="petal_length", data=iris)
plt.title("Violin Plot of Petal Length by Species")
plt.show()


# In[29]:


# Histogram using Matplotlib
plt.figure(figsize=(8, 6))
plt.hist(iris["sepal_width"], bins=15, color="skyblue", edgecolor="black")
plt.xlabel("Sepal Width")
plt.ylabel("Frequency")
plt.title("Histogram of Sepal Width")
plt.show()


# In[43]:


# Correlation Heatmap using Seaborn
correlation_matrix = iris.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

