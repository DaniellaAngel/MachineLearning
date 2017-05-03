import numpy as np
import pandas as pd

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


#83.3%
# validate = dataset.iloc[:,[ 5,20,30,72,9,91,94,21,3,97,61,67,79,4,18,62,54,57,95,80,70,68,40,50,60,52,82,25,71,19,32,33,55,59,46,12]]
arr = [1,58,68,81,29,40,51,92,93,57,39,6,87,67,53,90,42,27,52,54,44,45,47,76,88,63,56,66,86,14,98,80,95,2,5,41,64,72,75,83,89,70,22,18,9,60,13,85,33,24,77,84,61,4,73,26,50,65]
validate = dataset.iloc[:,[1,58,68,81,29,40,51,92,93,57,39,6,87,67,53,90,42,27,52,54,44,45,47,76,88,63,56,66,86,14,98,80,95,2,5,41,64,72,75,83,89,70,22,18,9,60,13,85,33,24,77,84,61,4,73,26,50,65]]
#83.42%
# arr = [1,25,68,2,18,7,9,97,80,14,30,4,73,27,38,16,33,95,94,65,44,63,47,83,70,96,55,12,69,26,53,81,36,13,75]
# validate = dataset.iloc[:,[1,25,68,2,18,7,9,97,80,14,30,4,73,27,38,16,33,95,94,65,44,63,47,83,70,96,55,12,69,26,53,81,36,13,75]]
print validate

pred_labels_ = pred_label(validate)
acc = caculate_acc(pred_labels_,label)
print acc,len(arr)