import numpy as np
import pandas as pd
import random

dataset = pd.read_csv("datas.csv")
label = pd.read_csv("labels.csv")
def pred_label(classifiers):
	# print "c",len(self.classifiers.values)
	if len(classifiers.values) == 0:
		print "initial classifiers"
		return
	else:
		add_labels = classifiers.apply(lambda x: x.sum(), axis=1).values
		pred_labels = []
		for item1 in add_labels:
			if item1 > 0:
				pred_labels.append(1)
			else:
				pred_labels.append(-1)
		# print "pred_labels",self.pred_labels
		return pred_labels

def caculate_acc(pred_value,label):
	diff_labels = pred_value - label.values.T[0]
	count = 0
	for item2 in diff_labels:
		if item2 == 0:
			count+=1
		else:
			count = count
	accuracy = float(count)/len(label.values.T[0])

	return accuracy

# print dataset.columns.values
# result = [x-1 for x in np.array(dataset.columns.values)]
print np.array(dataset.columns.values)
for item in dataset.columns.values:
	print item-1

# #83.3%
# # validate = dataset.iloc[:,[ 5,20,30,72,9,91,94,21,3,97,61,67,79,4,18,62,54,57,95,80,70,68,40,50,60,52,82,25,71,19,32,33,55,59,46,12]]
# arr = [1,58,68,81,29,40,51,92,93,57,39,6,87,67,53,90,42,27,52,54,44,45,47,76,88,63,56,66,86,14,98,80,95,2,5,41,64,72,75,83,89,70,22,18,9,60,13,85,33,24,77,84,61,4,73,26,50,65]
# validate = dataset.iloc[:,[1,58,68,81,29,40,51,92,93,57,39,6,87,67,53,90,42,27,52,54,44,45,47,76,88,63,56,66,86,14,98,80,95,2,5,41,64,72,75,83,89,70,22,18,9,60,13,85,33,24,77,84,61,4,73,26,50,65]]
# #83.42%
# # arr = [1,25,68,2,18,7,9,97,80,14,30,4,73,27,38,16,33,95,94,65,44,63,47,83,70,96,55,12,69,26,53,81,36,13,75]
#84.47%
# arr =  [63,22,91,20,90,52,26,4,61,70,86,17,78,49,97,96,67,39,11,64,71,33,3,6,76,14,73,66,80,85,84,31,19,47,82,45,27,41,74,1,25,32,8,98,59,99,5,56,30,18,9]

# arr =  [38,17,89,83,45,8,13,73,53,79,47,43,40,26,34,52,98,91,56,39,25,77,21,61,24,55,20,30,32,5,51,63,31,65,85,3,76,81,87,22,92,82,10,97,70,48,35]
# result =[x-1 for x in arr]
# print result
# validate = dataset.iloc[:,result]

# print validate

# pred_labels_ = pred_label(validate)
# acc = caculate_acc(pred_labels_,label)
# print acc
