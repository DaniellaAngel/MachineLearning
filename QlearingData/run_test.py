from classifier_env import Classifier
from Brain_RL import QLTable

def update():

	for episode in range(10):
		# initial observation
		print "run episode======================================================",episode
		observation = env.reset().columns.values
		print "run observation",observation
		while True:
			# fresh env
			env.render()
			# print "str(observation)",str(observation)
			# RL choose action based on observation
			action = RL.choose_action(str(observation))
			# print "run action",action
			# RL take action and get next observation and reward
			observation_, reward, done = env.step(action)
			# print "run observation_,reward,done",observation_, reward, done
			# RL learn from this transition
			RL.learn(str(observation), action, reward, str(observation_))

			# swap observation
			observation = observation_

			# break while loop when end of this episode
			if done==True:
			    break

if __name__ == "__main__":
	
	env = Classifier()
	print env.n_actions,env.pred_label()
	RL = QLTable(actions=list(range(env.n_actions)))
	update()
