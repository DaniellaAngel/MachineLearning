
import numpy as np
np.random.seed(1)
import pandas as pd
import time
import random

NUM = 99
GOAL = 0.84
DIFF = 3800 #the number of less diffrences

class Classifier(object):
	"""docstring for Classifer"""
	def __init__(self):
		super(Classifier, self).__init__()
		self.action_space = ['a','d'] #a----add;d-----delete
		self.n_actions = len(self.action_space)
		self.n_features = 2
		self.classifiers = pd.DataFrame()
		self.dataset = pd.read_csv("datas.csv")
		self.label = pd.read_csv("labels.csv")
		self.sate = np.random.randint(0,9)

	def build_classifier(self):
		self.classifiers = pd.DataFrame()

		return self.classifiers

	def pred_label(self):
		if len(self.classifiers.values) == 0:
			print "initial classifiers"
			self.reset()
			return
		else:
			self.add_labels = self.classifiers.apply(lambda x: x.sum(), axis=1).values
			self.pred_labels = []
			self.diff = []
			count = 0
			val_length = len(self.classifiers.columns.values)
			for item1 in self.add_labels:
				if abs(item1) > (val_length/10)*10:
					count += 1
				self.diff.append(item1)

				if item1 > 0:
					self.pred_labels.append(1)
				else:
					self.pred_labels.append(-1)

			return self.pred_labels,count

	def caculate_acc(self,pred_value):
		diff_labels = pred_value - self.label.values.T[0]
		count = 0
		for item2 in diff_labels:
			if item2 == 0:
				count+=1
			else:
				count = count
		accuracy = float(count)/len(self.label.values.T[0])

		return accuracy

	def reset(self):
		self.origin_state = 1
		self.classifiers = pd.DataFrame()
		self.classifiers[self.origin_state] = self.dataset.iloc[:,[self.origin_state]]

		return self.classifiers

	def step(self,action):
		s = random.sample(range(NUM),2)
		d = np.array([1])
		x = 2
		nd = np.hstack((d,)*x)

		# print "s=======>",s

		self.classifiers[s+nd] = self.dataset.iloc[:,s]

		if action ==0: #add
			self.classifiers[s+nd] = self.dataset.iloc[:,s]
		elif action ==1: #delete
			dropnN = random.sample(list(self.classifiers.columns.values),2)
			# print "dropnN",dropnN
			self.classifiers.drop(dropnN,axis=1,inplace=True)

		if len(self.classifiers.columns.values)%2 == 0:
			# print "this is =========>2"
			dropn2 = random.sample(list(self.classifiers.columns.values),1)
			# print "dropn2",dropn2
			self.classifiers.drop(dropn2,axis=1,inplace=True)

		# print "length of classifiers3==========>",len(self.classifiers.columns.values)		
		# print "env classifier values====3>",self.classifiers.columns.values

  		pred_labels, count_arr = self.pred_label()
  		accuracy = self.caculate_acc(pred_labels)
  		# print "env ,accuracy1========>",accuracy

		s_ = random.sample(range(NUM),2)
		# print "s_======>",s_
		self.classifiers[s_+nd] = self.dataset.iloc[:,s_]

		
		if len(self.classifiers.columns.values)%2 == 0:
			# print "this is =========>3"
			dropn3 = random.sample(list(self.classifiers.columns.values),1)
			self.classifiers.drop(dropn3,axis=1,inplace=True)

		# print "length of classifiers5==========",len(self.classifiers.columns.values)	
		# print "env s_ classifier values====5>",self.classifiers.columns.values

		pred_labels_, count_arr_ = self.pred_label()
  		accuracy_ = self.caculate_acc(pred_labels_)

  		# print "env accuracy========>",accuracy_

		if accuracy_ > GOAL:
			# and count_arr_< DIFF:
			reward = 1
			done = True
			print "env accuracy========>",accuracy_
			print "env final classifier values====>",self.classifiers.columns.values
			print "length of classifiers==========>",len(self.classifiers.columns.values)
		elif accuracy_ == GOAL:
			# and count_arr_< DIFF:
			reward = 1
			done = True
			print "env accuracy========>",accuracy_
			print "env final classifier values====>",self.classifiers.columns.values
			print "length of classifiers==========>",len(self.classifiers.columns.values)
		else:
			if accuracy < accuracy_:
				reward = 0
				done = False
			# if accuracy < accuracy_ or (count_arr_ < count_arr and count_arr_ > DIFF):
			# 	reward = 0
			# 	done = False
			# if accuracy < accuracy_ or (count_arr_ < count_arr and count_arr_ < DIFF):
			# 	reward = 0
			# 	done = False
			else:
				reward = -1
				done = False

		return s_, reward, done, accuracy_

	def render(self):
		time.sleep(0.1)
		pass
		
	
		












