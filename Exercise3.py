#!/usr/bin/env python
# coding: utf-8

# # Import the relevant libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import seaborn as sns
sns.set()


# # Loading the raw data. Preprocessing

# In[2]:


raw_csv_data = pd.read_csv('Internship_train.csv')


# In[3]:


df = raw_csv_data.copy()
df.head(15)


# In[4]:


pd.options.display.max_columns = None


# In[5]:


#display(df)


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


sns.displot(df['target'])


# # Feature selection

# In[9]:


#for i in range(0, 53):
#    feature = '' + str(i)
#    plt.scatter(df[feature], df['target'])
#    plt.xlabel(feature)
#    plt.ylabel('target')
#    plt.show()


# In[10]:


#Only feature '6' is meaningful for our analysis, all other seems random.
cleaned_df = df[['6', 'target']]
cleaned_df.head()


# In[11]:


cleaned_df.describe()


# In[12]:


sns.displot(df['6'])


# In[13]:


features = cleaned_df['6']
targets = cleaned_df['target']
X = np.array(features)
y = np.array(targets)


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
X_test_poly = poly.transform(X_test.reshape(-1, 1))


# # Regression model

# In[16]:


reg = LinearRegression()
reg.fit(X_train_poly, y_train)

y_pred = reg.predict(X_test_poly)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)


# # Predictions

# In[20]:


data_for_predictions = pd.read_csv('Internship_hidden_test.csv')


# In[21]:


data_for_predictions.head()


# In[22]:


df_for_predictions = data_for_predictions.copy()
df_for_predictions.describe()


# In[23]:


x_new = df_for_predictions[['6']]
x_new = np.array(x_new)
x_new_poly = poly.transform(x_new.reshape(-1, 1))
y_pred = reg.predict(x_new_poly)


# In[25]:


model_predictions = pd.DataFrame({'predictions' : y_pred})
df_predictions = pd.concat([df_for_predictions, model_predictions], axis = 1)
df_predictions


# In[26]:


df_predictions.to_csv('Internship_predictions.csv', index=False)


# In[ ]:




