
# coding: utf-8

# In[32]:


# Load dataset using pandas
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np

df = pd.read_csv("TrainTestData_Promotional Sales.csv")
df = df[df['Trx'] != '10000']

#df = df.drop(['Account'], axis=1)


# In[33]:


# Split Test Data and Train Data

test_df = df[df['Trx'] == '????']
train_df = df[df['Trx'] != '????']


# In[34]:


# Drop Month Column
train_df = train_df.drop(['Month'], axis=1)

test_df = test_df.drop(['Month'], axis=1)


# In[35]:


# Convert Trx to integer type

train_df['Trx'] = train_df['Trx'].astype(int)


# In[36]:


train_df['Trx'] = train_df['Trx'].apply(lambda x: np.log2(x + 1))


# In[37]:


# Separate Trx
y_train = train_df['Trx']
train_data = train_df.drop(['Trx'], axis=1)

y_test_predict = test_df['Trx']
test_data = test_df.drop(['Trx'], axis=1)


# In[38]:


test_data.columns


# In[39]:


int_columns = ['# Speaker Programs', '# Calls', '# PDE', '# Emails', '# Clicks', '# Samples']

train_data[int_columns] = train_data[int_columns].apply(lambda x: np.log2(x + 1))
test_data[int_columns] = test_data[int_columns].apply(lambda x: np.log2(x + 1))


# In[40]:


import matplotlib.pyplot as plt

train_data.hist()
plt.show()


# In[41]:


# import matplotlib.pyplot
# import pylab

# x = train_data['# Samples'].tolist()
# y = y_train.tolist()

# matplotlib.pyplot.scatter(x,y)

# matplotlib.pyplot.show()


# In[42]:


train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)


# In[43]:


# from sklearn.decomposition import PCA

# pca = PCA(n_components=3)
# pca.fit(train_data)

# train_data = pca.fit_transform(train_data)
# test_data = pca.fit_transform(test_data)


# In[44]:


# Split train test data from train dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data, y_train, test_size=0.20, random_state=42)


# In[45]:


# # Apply Standard Scaling
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# scaler.fit(X_train)
# scaler.transform(X_train)
# scaler.transform(X_test)


# In[46]:


# # Try Linear Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

poly = PolynomialFeatures(degree=3)
X_ = poly.fit_transform(X_train)
X_test_ = poly.fit_transform(X_test)

clf = LinearRegression()
#clf = MLPRegressor()
#clf = XGBRegressor()
#clf = RandomForestRegressor(random_state=42, n_estimators=50)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error
import math

print math.sqrt(mean_squared_error(y_test, y_pred))


# In[47]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [16,9]
plt.scatter(clf.predict(X_train), clf.predict(X_train) - y_train, c='b', s=40, alpha=0.5)
plt.scatter(clf.predict(X_test), clf.predict(X_test) - y_test, c='g', s=40)
plt.yticks(np.arange(-8000, 2000, 500))
plt.hlines(y=0, xmin=0, xmax=50)
plt.show()


# In[199]:


clf.feature_importances_


# In[133]:


predictions = clf.predict(test_data)

# predictions = np.asarray(predictions)
# np.savetxt("output.csv", predictions, delimiter=",")


# In[138]:


squarer = lambda t: 2 ** t
vfunc = np.vectorize(squarer)
exp_predictions = vfunc(predictions)


# In[139]:


def chunks(l, n):
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]
    
pred_array = list(chunks(exp_predictions, 6))


# In[140]:


np.savetxt("output.csv", np.asarray(pred_array), delimiter=",", header='A,B,C,D,E,F')

