

from classifier_env import Classifier
# from Brain_RL import DeepQNetwork
from doubleDQN import DoubleDQN

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


env = Classifier()
MEMORY_SIZE = 3000
ACTION_SPACE = 2

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):
    natural_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=2, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=False, sess=sess
    )

with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=2, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=False)

sess.run(tf.global_variables_initializer())

def train(RL):
    total_steps = 0
    observation = np.array([0,env.reset().columns.values[0]])
    print "observation",observation
    while True:
        # if total_steps - MEMORY_SIZE > 8000: env.render()

        action = RL.choose_action(observation)
        # print "action",action

        # f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # convert to [-2 ~ 2] float actions
        # print "action",f_action
        # observation_, reward, done, acc = env.step(np.array([f_action]))
        observation_, reward, done ,acc= env.step(action)
        # print observation_, reward, done ,acc

        # reward /= 10     # normalize to a range of (-1, 0). r = 0 when get upright
        # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
        # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.
        # print reward
        RL.store_transition(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE:   # learning
            RL.learn()

        if total_steps - MEMORY_SIZE > 400000:   # stop game
            break

        observation = observation_
        if total_steps%100 == 0:
            print "##########step=",total_steps,"the accuracy=",acc,"##########"
        total_steps += 1
    return RL.q
q_natural = train(natural_DQN)
q_double = train(double_DQN)

plt.plot(np.array(q_natural), c='r', label='natural')
plt.plot(np.array(q_double), c='b', label='double')
plt.legend(loc='best')
plt.ylabel('Q eval')
plt.xlabel('training steps')
plt.grid()
plt.show()