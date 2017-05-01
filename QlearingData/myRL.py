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

# state = np.random.randint(0,9)
# if action == 1:
# 	classifiers[state] = datas.iloc[:,[state]]
# elif action == 0:
# 	classifiers.pop(state)

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
