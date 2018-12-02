import pandas as pd

from model_optimization.mo_lib import get_mse_fi_mo
from feature_select.ga import GeneticAlgorithmFeatureSelection

if __name__ == '__main__':
    """Load test data and run optimization"""
    # df = pd.read_csv('../data/winequality-red.csv')

    print('Loading data')
    df = pd.read_csv('../data/forest_cover/covtype.data')
    cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                  'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                  'Horizontal_Distance_To_Fire_Points']
    cols.extend([f'Wilderness_Area_{x}' for x in range(4)])
    cols.extend([f'Soil_Type_{x}' for x in range(40)])
    cols.append('Cover_Type')

    df.columns = cols

    # select_features_cma(df, 'quality')
    print('Starting GA')

    ga = GeneticAlgorithmFeatureSelection(get_mse_fi_mo, df, 'Cover_Type', fitness_args=['rf'], mu=0, sample_size=10000)

    ga.run_iterations(1000)

    print('Done!')