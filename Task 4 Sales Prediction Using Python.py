#!/usr/bin/env python
# coding: utf-8

# ## Sales Predictio Using Pyhon

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("advertising.csv")
df


# ## Data Analysis

# In[3]:


df.info()


# In[4]:


print(df.columns)


# In[5]:


column_names = ['TV', 'Radio', 'Newspaper', 'Sales']

for column in column_names:
    column_data = df[column]
    print(f"Column: {column}")
    print(column_data)
    print()


# In[6]:


df.isnull().sum()


# In[7]:


df.count()


# In[8]:


print(df.describe)


# In[9]:


print(df.shape)


# In[10]:


print(df.dtypes)


# In[11]:


filtered_data = df[(df['Radio'] >= 3.7) & (df['Radio'] <= 10.8)]

print(filtered_data)


# In[12]:


filtered_data = df[(df['TV'] >= 180) & (df['TV'] <= 230)]

print(filtered_data)


# In[13]:


filtered_data = df[(df['Newspaper'] >= 40) & (df['Newspaper'] <= 60)]

print(filtered_data)


# In[14]:


filtered_data = df[(df['Sales'] >= 12) & (df['Sales'] <= 15)]

print(filtered_data)


# In[15]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[16]:


df.hist(bins=10, figsize=(10, 8))
plt.tight_layout()
plt.show()


# In[17]:


# Define the colors based on conditions
colors = ['red' if length >= 120 else 'yellow' for length in df['TV']]


# In[18]:


# Scatter plot with different colors
plt.scatter(df['TV'], df['Radio'], c=colors)
plt.xlabel('TV')
plt.ylabel('Radio')
plt.title('TV vs Radio')


# In[19]:


# Define the colors based on conditions
colors = ['red' if length >= 45 else 'yellow' for length in df['Newspaper']]


# In[20]:


# Scatter plot with different colors
plt.scatter(df['Newspaper'], df['Radio'], c=colors)
plt.xlabel('Newspaper')
plt.ylabel('Radio')
plt.title('Newspaper vs Radio')


# In[21]:


# Select the columns for correlation
columns = ['TV', 'Radio', 'Newspaper', 'Sales']


# In[22]:


# Calculate the coefficient matrix
correlation_matrix = df[columns].corr()


# In[23]:


# Display the coefficient matrix
print(correlation_matrix)


# In[24]:


# Plot the correlation matrix as a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[25]:


sns.countplot(x='TV', data=df, )
plt.show()


# In[26]:


#check Outliers
sns.heatmap(df.isnull(), yticklabels=False, annot=True)


# In[27]:


sns.scatterplot(x='TV', y='Sales',
                hue='TV', data=df, )


# In[28]:


sns.scatterplot(x='Newspaper', y='Sales',
                hue='Newspaper', data=df, )


# In[29]:


sns.pairplot(df, hue='TV', height=2)


# In[30]:


plt.figure(figsize=(10, 10))


# In[31]:


# Creating box plots for 'TV', 'Radio', and 'Newspaper' against 'Sales'
plt.subplot(2, 2, 1)
sns.boxplot(x='TV', y='Sales', data=df)

plt.subplot(2, 2, 2)
sns.boxplot(x='Radio', y='Sales', data=df)

plt.subplot(2, 2, 3)
sns.boxplot(x='Newspaper', y='Sales', data=df)

plt.tight_layout()
plt.show()


# In[32]:


# Distribution of a numerical column
sns.histplot(df['Sales'], kde=True)
plt.title('Distribution of Sales')
plt.show()


# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[34]:


# Splitting the data into features (X) and target variable (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']


# In[35]:


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[36]:


# Training the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[37]:


# Predicting on the test set
y_pred = lr.predict(X_test)


# In[38]:


# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)


# In[39]:


# Calculating accuracy
threshold = 0.1  # Define your threshold value here
accurate_predictions = (abs(y_test - y_pred) <= threshold).sum()
total_predictions = len(y_test)
accuracy = accurate_predictions / total_predictions


# In[40]:


print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)
print("Accuracy:", accuracy)


# In[41]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[42]:


# Splitting the data into features (X) and target variable (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']


# In[43]:


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[44]:


# Training the RandomForestRegressor model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)


# In[45]:


# Predicting on the test set
y_pred = rf.predict(X_test)


# In[46]:


# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)


# In[47]:


print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)

