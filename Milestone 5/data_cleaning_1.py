import pandas as pd
import numpy as np

raw_df = pd.read_csv('train_cimb.csv')
raw_df.describe()
raw_df.shape()

#target variable
raw_df['Return'] = raw_df['Close'].pct_change(1).shift(-1)
raw_df['target_cls'] = np.where(raw_df.Return > 0, 1, 0)
raw_df['target_rgs'] = raw_df['Return']
raw_df.tail()

# drop na
raw_df = raw_df.dropna()

# drop unwanted column
clean_df = raw_df.drop(['Date','target', 'return', 'Return', 'target_rgs'],axis=1)

# save
clean_df.to_csv('cimb_clean.csv', index=False)