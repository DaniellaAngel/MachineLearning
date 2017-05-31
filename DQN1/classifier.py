
import numpy as np
np.random.seed(1)
import pandas as pd
import time
import random

INITLEN = 11
STEPLEN = 9
DROPNUM = 3
REFERACC = 0.8326
GOALACC = 0.8360

class Classifier(object):
	"""docstring for Classifer"""
	def __init__(self):
		super(Classifier, self).__init__()
		self.action_space = ['a','d'] #a----add;d-----delete
		self.n_actions = len(self.action_space)
		self.n_features = 2
		self.dataset = pd.read_csv("datas.csv")
		self.label = pd.read_csv("labels.csv")
		self.classifiers = pd.DataFrame()
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
			for item1 in self.add_labels:
				if item1 > 0:
					self.pred_labels.append(1)
				else:
					self.pred_labels.append(-1)
			return self.pred_labels

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
		self.init_classifiers = np.array(random.sample(list(self.dataset.columns.values),INITLEN)).astype('int')
		self.initones = np.hstack((np.array([1]),)*INITLEN)
		self.classifiers[self.init_classifiers] = self.dataset.iloc[:,self.init_classifiers-self.initones]
		if len(self.classifiers.columns.values)%2 == 0:
			# print "this is =========>2"
			init_dropn = random.sample(list(self.classifiers.columns.values),1)
			# print "dropn2",dropn2
			self.classifiers.drop(init_dropn,axis=1,inplace=True)
		self.initpred_labels = self.pred_label()
  		self.init_accuracy = self.caculate_acc(self.initpred_labels)

  		self.init_state = np.array([len(self.classifiers.columns.values),self.init_accuracy])

  		print len(self.classifiers.columns.values),self.classifiers.columns.values,self.init_accuracy
		return self.init_state

	def step(self,action):
		# s = self.init_state
		# print "s=======>",s
		if action == 0: #add

			arr0 = np.array(random.sample(list(self.dataset.columns.values),STEPLEN)).astype('int')
			arrnd = np.hstack((np.array([1]),)*STEPLEN)
			self.classifiers[arr0] = self.dataset.iloc[:,arr0-arrnd]

			if len(self.classifiers.columns.values)%2 == 0:
				dropn0 = random.sample(list(self.classifiers.columns.values),1)
				self.classifiers.drop(dropn0,axis=1,inplace=True)

			pred_labels = self.pred_label()
  			accuracy = self.caculate_acc(pred_labels)

			s_ = np.array([len(self.classifiers.columns.values),accuracy])

		elif action == 1: #delete
			dropnN = random.sample(list(self.classifiers.columns.values),DROPNUM)
			self.classifiers.drop(dropnN,axis=1,inplace=True)

			if len(self.classifiers.columns.values)%2 == 0:
				dropn1 = random.sample(list(self.classifiers.columns.values),1)
				self.classifiers.drop(dropn1,axis=1,inplace=True)

			pred_labels = self.pred_label()
  			accuracy = self.caculate_acc(pred_labels)

			s_ = np.array([len(self.classifiers.columns.values),accuracy])

		print "s_=======>",s_
		clen = len(self.classifiers.columns.values)
		print "clen",clen
		if clen < 50 or accuracy < REFERACC:
			reward = -1
			done = True
		elif clen > 50 and accuracy > GOALACC:
			reward = 1
			done = True
			print "env accuracy========>",accuracy_
			print "env final classifier values====>",self.classifiers.columns.values
			print "length of classifiers5==========>",len(self.classifiers.columns.values)
		else:
			reward = 0
			done = False
		

		return s_, reward, done, accuracy, clen

	def render(self):
		time.sleep(0.1)
		pass
		
	
		












