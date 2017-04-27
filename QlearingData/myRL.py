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
datas = pd.read_csv("data.csv")
labels = pd.read_csv("label.csv")
print len(datas.values)


#load columns form datas
classifiers = pd.DataFrame()
for x in xrange(1,10):
	state = np.random.randint(0,9)
	print "state",state
	classifiers[state] = datas.iloc[:,[state]]
	if x == 2:
		print state
		classifiers.pop(state)
	print classifiers
print "classifiers1",classifiers

def acc_pred(classifiers_,labels_):
	add_labels = classifiers_.apply(lambda x: x.sum(), axis=1).values
	pred_labels = []
	for item1 in add_labels:
		if item1 > 0:
			pred_labels.append(1)
		else:
			pred_labels.append(-1)
	diff_labels = pred_labels - labels_
	count = 0
	for item2 in diff_labels:
		if item2 == 0:
			count+=1
		else:
			count = count
	accuracy = float(count)/len(labels_)
	return accuracy
accuracy = acc_pred(classifiers,labels.values.T[0])
print "accuracy",accuracy

# class classifier_fakegame():
# 	def _init_(self):
# 		self.datas = pd.read_csv("data.csv")
# 		self.labels = pd.read_csv("label.csv")
# 		self.state = 0
# 		self.classifiers = pd.DataFrame()
# 		self.num_classifiers = self.classifiers.values.shape[1]
# 		self.num_actions = 2
	
# 	def getClassifier(self):
# 		self.state = np.random.randint(0,len(self.datas))
# 		return self.state

# 	def classifiers(self):
# 		self.classifiers[self.state] = self.datas.pop(self.state)
# 	#sum the rows of the array
# 	def sum_rows(self):
# 		return self.datas.apply(lambda x: x.sum(), axis=1)

# 	#process the result of adding all the rows,convert it to labels -1 or 1
# 	def prediction_labels(pred_labels):
# 		labels = []
# 		for item in pred_labels:
# 			if item > 0:
# 				labels.append(1)
# 			elif item < 0:
# 				labels.append(-1)
# 		return np.asarray(labels), len(labels)
# 	def count_acc(pred_labels,labels):
# 		diff_labels = pred_labels - labels
# 		count = 0
# 		for item in diff_labels:
# 			if item == 0:
# 				count+=1
# 			else:
# 				count = count
# 		return count
# 	def pullClassifier(self,action):

# 		result = np.random.randn(1)
# 		if result > classifier:
# 			return 1
# 		else:
# 			return -1


