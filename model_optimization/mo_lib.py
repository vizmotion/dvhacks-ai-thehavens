# Library for model optimization
# 2018-11-30

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# get_RF_mse_sel: function to run random forest on the data input
# input:
#   data: pandas dataframe with the data
#   target: target feature to run the model against
#   features: list of features to develop the model
# output: mse

def get_RF_mse_sel(data,target,features):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2018)
    rf = RandomForestRegressor()
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    return mean_squared_error(y_test, y_pred, multioutput='raw_values')

# get_features: function to randomly select features to train the model
# input:
#   data: pandas dataframe with the data
#   size: number of features to select
# output: features

def get_features(data,size):
    features = np.unique(np.random.choice(data.columns.drop(target),size=size))
    return features

# get_min_RF_mse_bf: function to find set of features with the minimum mse using
# brute force - random sampling
# input:
#   data: pandas dataframe with the data
#   target: target feature to run the model against
#   n_iterations: how many iterations to perform

def get_min_RF_mse_bf(data,target,n_iterations):
    # select a very high minimum value
    mse_min = 10000
    for size in range(1,len(data.columns.drop(target))):
        for _ in range(n_iterations):
            # generate set of features
            features = np.unique(np.random.choice(data.columns.drop(target),size=size))
            # compute mse for the RF model
            mse = get_RF_mse_sel(data,target,features)
            # update minimum
            if mse < mse_min:
                mse_min = mse
                features_min = features
    return features_min, mse_min