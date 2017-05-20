import numpy as np
import pandas as pd
import random

dataset = pd.read_csv("datas.csv")
label = pd.read_csv("labels.csv")

def choose_diff(dataset):
	add_labels = dataset.apply(lambda x: x.sum(), axis=1).values
	diff = []
	count = 0
	val_length = len(dataset.columns.values)
	for item in add_labels:
		if abs(item) > 85:
			count += 1
		diff.append(item)
	print count
choose_diff(dataset)
print 11/10
# def pred_label(classifiers):
# 	# print "c",len(self.classifiers.values)
# 	if len(classifiers.values) == 0:
# 		print "initial classifiers"
# 		return
# 	else:
# 		add_labels = classifiers.apply(lambda x: x.sum(), axis=1).values
# 		pred_labels = []
# 		for item1 in add_labels:
# 			if item1 > 0:
# 				pred_labels.append(1)
# 			else:
# 				pred_labels.append(-1)
# 		# print "pred_labels",self.pred_labels
# 		return pred_labels

# def caculate_acc(pred_value,label):
# 	diff_labels = pred_value - label.values.T[0]
# 	count = 0
# 	for item2 in diff_labels:
# 		if item2 == 0:
# 			count+=1
# 		else:
# 			count = count
# 	accuracy = float(count)/len(label.values.T[0])

# 	return accuracy


# #83.3%
# # validate = dataset.iloc[:,[ 5,20,30,72,9,91,94,21,3,97,61,67,79,4,18,62,54,57,95,80,70,68,40,50,60,52,82,25,71,19,32,33,55,59,46,12]]
# arr = [1,58,68,81,29,40,51,92,93,57,39,6,87,67,53,90,42,27,52,54,44,45,47,76,88,63,56,66,86,14,98,80,95,2,5,41,64,72,75,83,89,70,22,18,9,60,13,85,33,24,77,84,61,4,73,26,50,65]
# validate = dataset.iloc[:,[1,58,68,81,29,40,51,92,93,57,39,6,87,67,53,90,42,27,52,54,44,45,47,76,88,63,56,66,86,14,98,80,95,2,5,41,64,72,75,83,89,70,22,18,9,60,13,85,33,24,77,84,61,4,73,26,50,65]]
# #83.42%
# # arr = [1,25,68,2,18,7,9,97,80,14,30,4,73,27,38,16,33,95,94,65,44,63,47,83,70,96,55,12,69,26,53,81,36,13,75]
#83.47%
# arr =  [63,22,91,20,90,52,26,4,61,70,86,17,78,49,97,96,67,39,11,64,71,33,3,6,76,14,73,66,80,85,84,31,19,47,82,45,27,41,74,1,25,32,8,98,59,99,5,56,30,18,9]
#83.52% 67
# arr = [68,58,81,9,3,15,46,61,40,24,86,8,93,63,50,74,14,94,5,87,83,79,65,17,98,53,13,67,37,54,89,29,42,82,60,28,6,95,56,18,33,71,75,35,90,34,12,76,20,99,26,47,62,78,85,69,73,11,92,10,55,49,91,57,64,2,52]
#0.8342 61
# arr = [17,9,31,5,69,21,23,74,61,76,35,12,90,59,26,15,79,41,8,24,68,80,20,98,86,30,85,22,18,56,2,45,83,52,62,44,73,40,19,64,81,87,14,84,47,34,67,16,39,7,60,42,3,88,49,97,78,32,91,51,33]
#0.8351 41
# arr = [60,48,83,98,6,64,20,9,40,21,26,44,69,76,35,66,52,29,68,82,1,87,81,4,18,70,90,27,31,47,28,72,22,2,19,42,45,3,58,91,30]
#0.8331 55
# arr =  [9,89,5,92,28,11,68,21,88,70,33,63,64,83,18,85,29,52,93,40,12,84,31,45,22,57,38,30,8,37,97,61,65,2,35,96,81,67,98,82,55,50,34,1,20,69,19,91,99,51,49,43,71,36,23]
#0.8339 43
# arr = [33,80,55,47,59,73,96,45,42,91,20,32,53,18,84,6,4,41,71,70,81,46,36,9,19,79,67,57,74,58,65,93,40,87,35,34,92,3,8,76,66,86,37]
#0.8347 33
# arr =  [62,87,76,18,86,24,80,98,53,38,12,16,52,81,63,34,19,65,89,1,35,33,41,36,20,37,70,60,39,3,99,40,5]
#0.8355 39
# arr = [55,69,23,79,47,42,76,96,32,52,59,75,73,16,81,85,80,20,33,87,37,82,13,83,4,70,41,5,98,74,60,45,93,28,3,97,56,91,18]

# result =[x-1 for x in arr]
# # print result
# validate = dataset.iloc[:,result]

# # print validate

# pred_labels_ = pred_label(validate)
# acc = caculate_acc(pred_labels_,label)
# print acc,len(arr)
# classifiers = pd.DataFrame()
# # s = np.array([np.random.randint(0,98),np.random.randint(0,98)])
# # s = [0,98]
# s = random.sample(range(99),2)
# classifiers[1] = dataset.iloc[:,1]
# print "classifiers1",classifiers
# print s
# # a = dataset.iloc[:,s]
# d = np.array([1])
# x = 2
# nd = np.hstack((d,)*x)
# classifiers[s+nd] = dataset.iloc[:,s]
# # print a
# # classifiers[s+nd] = dataset.iloc[:,s]
# print "classifiers2",classifiers
# classifiers.drop(s+nd,axis=1,inplace=True)
# # sumr = s+nd
# # classifiers.pop(sumr[0])
# # classifiers.pop(sumr[1])
# print "classifiers3",classifiers
# print "env classifier",classifiers.columns.values
# #print dataset.iloc[:,[93]]#have 0 value