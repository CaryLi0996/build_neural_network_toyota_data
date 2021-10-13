#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries and packages

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
import numpy as np

get_ipython().system('pip install dmba')
from dmba import classificationSummary
from dmba import regressionSummary
import pydotplus
from IPython.display import Image
import numbers
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Import dataset

df = pd.read_csv('ToyotaCorolla.csv')


# In[ ]:


# Only keep the important variables

predictors = ['Age_08_04', 'KM', 'Fuel_Type', 'HP', 'Automatic', 'Doors', 'Quarterly_Tax', 'Mfr_Guarantee', 'Guarantee_Period', 'Airco', 'Automatic_airco', 'CD_Player', 'Powered_Windows', 'Sport_Model','Tow_Bar', 'Price']


# In[ ]:


# Converts categorical data into dummy or indicator variables

pred_processed = pd.get_dummies(df[predictors], drop_first=True)


# In[ ]:


# Use transformer to scale the data to the range [0,1]
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(pred_processed)
print(scaled_data)
scaled_data = pd.DataFrame(scaled_data, columns = pred_processed.columns)
scaled_data.head()


# In[ ]:


outcome = 'Price'
predictors = [c for c in pred_processed.columns if c != outcome]

# partition data
X = scaled_data[predictors]
y = scaled_data[outcome]

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)
print('Train_X   :', train_X.shape)
print('Valid_X :', valid_X.shape)
print('Train_y   :', train_y.shape)
print('Valid_y :', valid_y.shape)


# In[ ]:


def RMS(clf):
  clf.fit(train_X, train_y)
  # training performance 
  trainPrediction = clf.predict(train_X)
  rms_train = mean_squared_error(train_y, trainPrediction, squared=False)
  print("RMS for training: ", rms_train)
  print("r2 score:", (r2_score(train_y, trainPrediction))*100)


  # validation performance
  validPrediction = clf.predict(valid_X)
  rms_valid = mean_squared_error(valid_y, validPrediction, squared=False)
  print("RMS for validation: ",rms_valid)
  print("r2 score:", (r2_score(valid_y, validPrediction))*100)


# In[ ]:


# Neural network model 1: single hidden layer with two nodes

clf1 = MLPRegressor(hidden_layer_sizes=(2), activation='logistic', solver='lbfgs',
                    random_state=1)

clf1.fit(train_X, train_y)

# RMS error for the training and validation data
RMS(clf1)


# In[ ]:


# Neural network model 2: single hidden layer with five nodes
clf2 = MLPRegressor(hidden_layer_sizes=(5), activation='logistic', solver='lbfgs',
                    random_state=1)

clf2.fit(train_X, train_y)

# RMS error for the training and validation data
RMS(clf2)


# In[ ]:


# Neural network model 3: two layers, five nodes in each layer
clf3 = MLPRegressor(hidden_layer_sizes=(5,5), activation='logistic', solver='lbfgs',
                    random_state=1)

clf3.fit(train_X, train_y)

# RMS error for the training and validation data
RMS(clf3)


# ### Semi-Conclusion: 
# 
# The RMS error for the training and validation data both increase as the number of layers and nodes increase, we can say this is because the model is overfitting the training data.

# In[ ]:


# Use GridSearchCV for hyperparameter tuning 
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

# Train neural network with 2 hidden nodes
clf = MLPRegressor(hidden_layer_sizes=(5), activation='logistic', solver='lbfgs',
                    random_state=1)
clf.fit(train_X, train_y)

param_grid = {
    'hidden_layer_sizes': [(2), (5), (5,5)], 
}

# apply grid search
gridSearch = GridSearchCV(MLPRegressor(activation='logistic', solver='lbfgs', random_state=1), 
                          param_grid, cv=5, n_jobs=-1, return_train_score=True)
gridSearch.fit(train_X, train_y)
print('Initial score: ', gridSearch.best_score_)
print('Initial parameters: ', gridSearch.best_params_)


# In[ ]:


# GridSearch CV to find the appropriate number of layers and nodes:

display=['param_hidden_layer_sizes', 'mean_test_score', 'std_test_score']
print(pd.DataFrame(gridSearch.cv_results_)[display])


# In[ ]:


pd.DataFrame(gridSearch.cv_results_)[display].plot(x='param_hidden_layer_sizes', 
                                                   y='mean_test_score', yerr='std_test_score')

plt.show()


# ### Summary:
# 
# Based on the hyperparameter tuning using GridSearchCV, 1 layer and 5 nodes performs the best since it has 
# 
# * lower RMS error than (2 layers, 5 nodes) 
# 
# * higher mean_test_score than (1 layer, 2 nodes).
