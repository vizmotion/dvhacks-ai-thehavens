import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
#TODO: Code in generic data cleaning

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