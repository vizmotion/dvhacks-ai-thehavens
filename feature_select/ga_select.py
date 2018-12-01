import pandas as pd
from random import random, seed
import cma

from model_optimization.mo_lib import get_mse_fi_mo


def select_features_cma(data, target):
    """
    DEPRECATED - This is a real-valued GA and doesn't work well for our purposes

    Select optimal set of features to predict target using CMA-ES

    :param data: dataframe containing data to be used for prediction
    :type data: Pandas Dataframe
    :param target: Label of target column
    :type target: string or other column label type

    """
    seed(0)
    initial_guess = [2*random()-1 for i in range(len(data.columns)-1)]
    print(f'initial guess - {initial_guess}')

    optimal_features = cma.fmin(get_mse_fi_mo, initial_guess, 0.01, args=(data, target, 'rf', False))
    # mse, feature_importances = get_mse_fi_mo(data, target, initial_guess, 'rf')
    # print(f'{mse}, {feature_importances}')

    print(f'{optimal_features}')



if __name__ == '__main__':
    """Load test data and run optimization"""
    df = pd.read_csv('../data/winequality-red.csv')
    select_features_cma(df, 'quality')

