import numpy as np
import pandas as pd

datas = pd.read_csv("data.csv")
print datas
# print datas.index
# print datas.columns
# print datas.values
print datas['2'].values #get the value of '2' column
# print datas.T 
print datas[0:1].values[0] #get the value of first row

line = datas[0:1].values[0]
# to do the vote
def voteMost(pramas):
	candidate = -1
	count = 0
	for value in pramas:
	  if count == -1:
	    candidate = value
	  if candidate == value:
	    count += 1
	  else:
	    count -= 1
	return candidate
result1 = voteMost(line)
print result1

