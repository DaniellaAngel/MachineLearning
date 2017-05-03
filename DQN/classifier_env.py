
import numpy as np
np.random.seed(1)
import pandas as pd
import time

NUM = 99
GOAL = 0.8326

class Classifier(object):
	"""docstring for Classifer"""
	def __init__(self):
		super(Classifier, self).__init__()
		self.action_space = ['a','d'] #a----add;d-----delete
		self.n_actions = len(self.action_space)
		self.n_features = 1
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
		# time.sleep(0.5)
		self.origin_state = 1
		self.classifiers = pd.DataFrame()
		self.classifiers[self.origin_state] = self.dataset.iloc[:,[self.origin_state]]

		return self.classifiers

	def step(self,action):
		s = np.random.randint(1,NUM)
		self.classifiers[s] = self.dataset.iloc[:,[s]]
		if action ==0: #add
			self.classifiers[s] = self.dataset.iloc[:,[s]]
		elif action ==1: #delete
			self.classifiers.pop(s)
		print "env classifier",self.classifiers.columns.values
  		pred_labels = self.pred_label()
  		accuracy = self.caculate_acc(pred_labels)

		s_ = np.random.randint(1,NUM)
		self.classifiers[s_] = self.dataset.iloc[:,[s_]]
		print "env s_ classifier values",self.classifiers.columns.values
		pred_labels_ = self.pred_label()
  		accuracy_ = self.caculate_acc(pred_labels_)
  		print "env ,accuracy_",accuracy_

		if accuracy > GOAL:
			# GOAL = accuracy
			reward = 1
			done = True
		elif accuracy == GOAL:
			# reward = 0
			# done = False
			reward = 1
			done = True

		else:
			if accuracy < accuracy_:
				reward = 1
				done = False
			else:
				reward = -1
				done = False

		return s_, reward, done

	def render(self):
		time.sleep(0.1)
		pass
		
	
		












