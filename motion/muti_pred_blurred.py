'''

Created by Daniella 2017/4/11

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np
import time

import convert2tfrecords as datafile

#############inputdata
train_imgs,train_labels_,test_imgs,test_labels_= datafile.read_tfrecord_fromfile()
# _train_labels = tf.one_hot(train_labels_, 4)
# _test_labels = tf.one_hot(test_labels_, 4)

##############################################################################
IMAGE_WIDTH = 256
IMAGE_HEIGHT = IMAGE_WIDTH
CHANNEL = 3
NUM_CLASS = 4
EPOCH = 10000
BATCH_SIZE = 4
##############################################################################

X = tf.placeholder(tf.float32,shape=[None,IMAGE_WIDTH,IMAGE_HEIGHT,CHANNEL])
Y = tf.placeholder(tf.float32,shape=[NUM_CLASS])


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.01)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# def max_pool(x):
	# return tf.nn.max_pool(x, ksize=[1,5,5,1], strides=[1,2,2,1], padding='SAME')
def avg_pool(x,ksize,strides):
	return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding='SAME')

############Define CNN Net
x_image = tf.reshape(X,[BATCH_SIZE, 256, 256, 3])
W_conv1 = weight_variable([5, 5, 3, 64])
b_conv1 = bias_variable([64])
scale1 = tf.Variable(tf.ones([64]))
beta1 =  tf.Variable(tf.ones([64]))
h_conv1_abs = tf.abs(conv2d(x_image, W_conv1) + b_conv1)
batch_mean1, batch_var1 = tf.nn.moments(h_conv1_abs,[0])
BN1 = tf.nn.batch_normalization(h_conv1_abs,batch_mean1,batch_var1,beta1,scale1,variance_epsilon=1e-3)
h_conv1 = tf.nn.tanh(BN1)
# h_conv1 = tf.nn.tanh(h_conv1_abs)
h_pool1 = avg_pool(h_conv1, [1,5,5,1], [1,2,2,1])

W_conv2 = weight_variable([5, 5, 64, 128])
b_conv2 = bias_variable([128])
scale2 = tf.Variable(tf.ones([128]))
beta2 =  tf.Variable(tf.ones([128]))
h_conv2_d = conv2d(h_pool1, W_conv2) + b_conv2
batch_mean2, batch_var2 = tf.nn.moments(h_conv2_d,[0])
BN2 = tf.nn.batch_normalization(h_conv2_d,batch_mean2,batch_var2,beta2,scale2,variance_epsilon=1e-3)
h_conv2 = tf.nn.tanh(BN2)
# h_conv2 = tf.nn.tanh(h_conv2_d)
h_pool2 = avg_pool(h_conv2, [1,5,5,1], [1,2,2,1])

W_conv3 = weight_variable([1, 1, 128, 256])
b_conv3 = bias_variable([256])
scale3 = tf.Variable(tf.ones([256]))
beta3 =  tf.Variable(tf.ones([256]))
h_conv3_d = conv2d(h_pool2, W_conv3) + b_conv3
batch_mean3, batch_var3 = tf.nn.moments(h_conv3_d,[0])
BN3 = tf.nn.batch_normalization(h_conv3_d,batch_mean3,batch_var3,beta3,scale3,variance_epsilon=1e-3)
h_conv3 = tf.nn.relu(BN3)
# h_conv3 = tf.nn.relu(h_conv3_d)
h_pool3 = avg_pool(h_conv3, [1,5,5,1], [1,2,2,1])

W_conv4 = weight_variable([1, 1, 256, 512])
b_conv4 = bias_variable([512])
scale4 = tf.Variable(tf.ones([512]))
beta4 =  tf.Variable(tf.ones([512]))
h_conv4_d = conv2d(h_pool3, W_conv4) + b_conv4
batch_mean4, batch_var4 = tf.nn.moments(h_conv4_d,[0])
BN4 = tf.nn.batch_normalization(h_conv4_d,batch_mean4,batch_var4,beta4,scale4,variance_epsilon=1e-3)
h_conv4 = tf.nn.relu(BN4)
# h_conv4 = tf.nn.relu(h_conv4_d)
h_pool4 = avg_pool(h_conv4, [1,5,5,1], [1,2,2,1])

W_conv5 = weight_variable([1, 1, 512, 1024])
b_conv5 = bias_variable([1024])
scale5 = tf.Variable(tf.ones([1024]))
beta5 =  tf.Variable(tf.ones([1024]))
h_conv5_d = conv2d(h_pool4, W_conv5) + b_conv5
batch_mean5, batch_var5 = tf.nn.moments(h_conv5_d,[0])
BN5 = tf.nn.batch_normalization(h_conv5_d,batch_mean5,batch_var5,beta5,scale5,variance_epsilon=1e-3)
h_conv5 = tf.nn.relu(BN5)
# h_conv5 = tf.nn.relu(h_conv5_d)
h_pool5 = avg_pool(h_conv5, [1,32,32,1], [1,32,32,1])
print('h_conv5',h_conv5.get_shape())

############the first fully connect
W_fc1 = weight_variable([1*1*1024,1024])
b_fc1 = bias_variable([1024])
h_pool5_flat = tf.reshape(h_pool5, [-1, 1*1*1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat,W_fc1) + b_fc1) 

############dropout layer
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

###########softmax layer
W_fc2 = weight_variable([1024, NUM_CLASS])
b_fc2 = bias_variable([NUM_CLASS])
output = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print(output.get_shape())
y_pridect = tf.nn.softmax(output)
print(y_pridect.get_shape())

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output,labels =Y))
###########learning rate decay
batch = tf.Variable(0,tf.float32)
starter_learning_rate = 1e-3
global_step = batch*BATCH_SIZE
decay_steps = 500
decay_rate = 0.9
learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,decay_steps,decay_rate,staircase=True)

# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(Y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

train_batch, train_label_batch = tf.train.batch([train_imgs, train_labels_],batch_size=BATCH_SIZE, capacity=32)
# train_batch, train_label_batch = tf.train.shuffle_batch([train_imgs, train_labels_],batch_size=batch_size, capacity=capacity,min_after_dequeue=min_after_dequeue)
test_batch,test_label_batch = tf.train.batch([test_imgs, test_labels_],batch_size=BATCH_SIZE, capacity=32)
# test_batch,test_label_batch = tf.train.shuffle_batch([test_imgs,test_labels_],batch_size=test_batch_size, capacity=capacity,min_after_dequeue=min_after_dequeue)


init_op = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init_op)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess = sess,coord=coord)
	try:
		print("start training==========================")
		for i in range(10):
			imgs, labels = sess.run([train_batch,train_label_batch])
			print("labels",labels)
			out_data,ypridect = sess.run([output,y_pridect],feed_dict = {X: imgs, Y: labels, keep_prob: 1.0})
			# train_step.run(feed_dict = {X: imgs, Y: labels, keep_prob: 0.2})
			# out_data,loss_val,train_acc= sess.run([output,cost,accuracy],feed_dict = {X: imgs, Y: labels, keep_prob: 1.0})
			print("out_data",out_data)
			print("ypridect",ypridect)
			# print("loss_val",loss_val)
			
			# if i%10 ==0:
			# 	train_acc = accuracy.eval(feed_dict = {X: imgs, Y: labels, keep_prob: 1.0})
			# 	print('step',i,'training accuracy',train_acc)
			# train_step.run(feed_dict = {X: imgs, Y: labels, keep_prob: 0.2})
		# print("start testing==========================")
		# for j in range(169):
		# 	if j % 10 == 0:
		# 		imgs_test, labels_test = sess.run([test_batch,test_label_batch])
				# print(labels_test)
				# test_acc = accuracy.eval(feed_dict = {X: imgs_test, Y: labels_test, keep_prob: 1.0 })
				# print('step',j,"test accuracy",test_acc)
	except tf.errors.OutOfRangeError:
		print('oops!')
	finally:
		print('finally')
		coord.request_stop()
	coord.join(threads)
	sess.close()