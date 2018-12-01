import mo_lib
import pandas as import pd
import numpy as np

data = pd.read_csv(file_name)
target = 'provided_by_user'

# funding the minimum mse by brute force
features_min, mse_min = mo_lib.get_min_RF_mse_bf(data,target,n_iterations=100)
