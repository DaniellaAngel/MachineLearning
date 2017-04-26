# import numpy as np
# import pandas as pd

# datas = pd.read_csv("datas.csv")
# datas['Col_sum'] = datas.apply(lambda x: x.sum(), axis=1)
# # print datas
# # print datas.index
# # print datas.columns
# # print datas.values
# # print datas['2'].values #get the value of '2' column
# # print datas.T 
# print datas.iloc[:,99:100].values

# def produce_labels(pred_labels):
# 	labels = []
# 	for item in pred_labels:
# 		if item[0] > 0:
# 			labels.append(1)
# 		elif item[0] < 0:
# 			labels.append(-1)
# 	return labels, len(labels)


# pred_labels = datas.iloc[:,99:100].values
# final_labels,num_labels = produce_labels(pred_labels)
# print final_labels,num_labels


# print datas[0:1].values[0] #get the value of first row

# line = datas[0:1].values[0]
# # to do the vote
# def voteMost(pramas):
# 	candidate = -1
# 	count = 0
# 	for value in pramas:
# 	  if count == -1:
# 	    candidate = value
# 	  if candidate == value:
# 	    count += 1
# 	  else:
# 	    count -= 1
# 	return candidate
# result1 = voteMost(line)


# print result1

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pandas as pd


#########################################################################################################
#states   is the name of classifiers 1,2,...99----the head of DataFrame of datas
#actions  is
#rewords  is
#########################################################################################################

#read the data from *.csv file ----the libs of all the classifers and the right labels
datas = pd.read_csv("datas.csv")
labels = pd.read_csv("labels.csv")

#sum the rows of the array
def sum_rows(datas):
	return datas.apply(lambda x: x.sum(), axis=1)

datas_row_sum = sum_rows(datas)
print datas_row_sum.values

#process the result of adding all the rows,convert it to labels -1 or 1
def prediction_labels(pred_labels):
	labels = []
	for item in pred_labels:
		if item > 0:
			labels.append(1)
		elif item < 0:
			labels.append(-1)
	return np.asarray(labels), len(labels)

pred_labels,pred_labels_len = prediction_labels(datas_row_sum.values)
print pred_labels,pred_labels_len,type(pred_labels)
print labels.T.values[0],labels.values.shape[0],labels.values.shape[1],type(labels.T.values[0])

print pred_labels-labels.T.values[0]

#count the number of the same labels between pred_labels and  labels
def count_acc(pred_labels,labels):
	diff_labels = pred_labels - labels
	count = 0
	for item in diff_labels:
		if item == 0:
			count+=1
		else:
			count = count
	return count

print count_acc(pred_labels,labels.T.values[0])
print type(datas.values),datas.values.shape[0],datas.values.shape[1]

#load columns form datas
def load_col():
	pass

# class classifier_fakegame():
# 	def _init_(self):
# 		self.state = 0
# 		self.classifiers = datas.values
# 		self.num_classifiers = self.classifiers.shape[1]
# 		self.num_actions = ['0','1']#0---minus classifier 1---add classifier

# 	def getClassifier(self):
# 		self.state = np.random.randint(0,len(self.classifiers))
# 		return self.state

# 	def pullClassifier(self,action):
# 		classifier = self.classifiers[self.state,action]
# 		result = np.random.randn(1)
# 		if result > classifier:
# 			return 1
# 		else:
# 			return -1
