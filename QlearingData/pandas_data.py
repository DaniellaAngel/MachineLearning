import numpy as np
import pandas as pd

df = pd.read_csv('data.csv')
df['Col_sum'] = df.apply(lambda x: x.sum(), axis=1)
# df.loc['Row_sum'] = df.apply(lambda x: x.sum())

print df

train_data = pd.DataFrame()
print train_data
for i in range(5):
	dt = df.iloc[:,[i]]
	# for item in dt:
	# 	print item
	print dt
	train_data[i] = dt
print train_data

# alist = [7,10,23,45,67,888,99]

# print df.iloc[3]

# counts = df[u'5'].value_counts

# print counts

# line = df[0:1]



