import mo_lib
import pandas as import pd
import numpy as np

"""
script to run the model optimization
input:
   file_name to a csv file
   flag optimization approach: -o
       bf: brute force
       ga: genetic algorithm
       gr: greedy
"""

file_name = '/Users/hugocontreras/Documents/GitHub/dvhacks-ai-thehavens/data/winequality-red.csv'
data = pd.read_csv(file_name)
target = 'provided_by_user'

# finding the minimum mse by brute force
features_min, mse_min = mo_lib.get_min_RF_mse_bf(data,target,n_iterations=100)
