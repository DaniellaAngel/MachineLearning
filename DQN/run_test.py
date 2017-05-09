import numpy as np

from classifier_env import Classifier
from Brain_RL import DeepQNetwork

def run_classifier():
	step = 0
	for episode in range(2):
		# initial observation
		print "run episode======================================================",episode
		observation = np.array([0,env.reset().columns.values[0]])
		print "run observation",observation
		while True:
			# fresh env
			env.render()
			# RL choose action based on observation
			action = RL.choose_action(observation)

			# RL take action and get next observation and reward
			observation_, reward, done ,acc= env.step(action)
			# print "observation_, reward, done",observation_, reward, done
			
			# RL learn from this transition
			# RL.store_transition(observation, action, reward, np.array([observation_]))
			RL.store_transition(observation, action, reward, observation_)

			if (step > 200) and (step % 5 == 0):
				RL.learn()
			# swap observation
			# observation = np.array([observation_])
			observation = observation_

			# break while loop when end of this episode
			if done==True:
			    break
			if step%100 == 0:
				print "##########################","episode=",episode,"step=",step,"the accuracy=",acc,"##########################"
			step += 1

	print "game over"

if __name__ == "__main__":
	env = Classifier()
	RL = DeepQNetwork(env.n_actions, env.n_features, learning_rate=0.01,reward_decay=0.9,e_greedy=0.9,replace_target_iter=200,memory_size=2000,output_graph=False)
	run_classifier()
	RL.plot_cost()
