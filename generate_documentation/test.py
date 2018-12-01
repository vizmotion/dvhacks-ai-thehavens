import gd_lib
import pandas as pd

data = pd.read_csv('/Users/hugocontreras/Documents/GitHub/dvhacks-ai-thehavens/data/winequality-red.csv')

gd_lib.get_connected_components_documentation(data,th_corr=0.15)
