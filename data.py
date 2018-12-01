import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.preprocessing import LabelEncoder
#TODO: Code in generic data cleaning
np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000

def clean_data(df):

	for col in df.columns.values:
		# Fill NaNs
		# with median if possible
		try:
			df[col].fillna(df[col].median(), inplace=True)
		except TypeError:
			col_mode = df[col].mode()
			if len(col_mode) > 0:
				df[col].fillna(df[col].mode()[0], inplace=True)
			else:
				df[col].fillna(method='bfill', inplace=True)
				df[col].fillna(method='ffill', inplace=True)
			
		# Encode labels
		if str(df[col].values.dtype) == 'object':
			col_encoder = LabelEncoder().fit(df[col].values)
			df[col] = col_encoder.transform(df[col].values)
			
	return df


def correlation_matrix(df):
	return df.corr(method='pearson')


# Testing
if __name__ == '__main__':
	df = pd.read_csv('./data/winequality-red.csv')
	df = clean_data(df)
	print(correlation_matrix(df))
