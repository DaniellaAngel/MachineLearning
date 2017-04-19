
import numpy as np
import time

import convert2tfrecords as datafile
import tensorflow as tf



training_iters = 10
# training_iters = 50000
batch_size = 8
display_step = 50
min_after_dequeue = 1000
capacity = min_after_dequeue + 3 * batch_size
# num_threads = 2
train_num_image = 10000
test_num_image = 10000

test_batch_size =10
test_iters = 10000
test_display_step = 50
merged = None

IMAGE_WIDTH=256
IMAGE_HEIGHT=256
CHANNEL=3


#with tf.name_scope('inputs'):
#traindata
X = tf.placeholder(tf.float32,shape=(batch_size,IMAGE_WIDTH,IMAGE_HEIGHT,CHANNEL),name='input_X')
#trainLabels
Y = tf.placeholder(tf.float32,shape=(batch_size,2),name='input_Y')
#testdata
Z = tf.placeholder(tf.float32,shape=(test_batch_size,IMAGE_WIDTH,IMAGE_HEIGHT,CHANNEL),name='input_Z')
#testlabels
W = tf.placeholder(tf.float32,shape=(test_batch_size,2),name='input_W')


conv1_weights = tf.Variable(tf.truncated_normal([5, 5, 3, 64], stddev=0.01),name="conv1_weights")
conv1_biases = tf.Variable(tf.zeros([64]),name = 'conv1_biases')
scale1 = tf.Variable(tf.ones([64]),name='scale1')
beta1 =  tf.Variable(tf.ones([64]),name='beta1')

conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=0.01),name="conv2_weights")
conv2_biases = tf.Variable(tf.zeros([128]),name ='conv2_biases')
scale2 = tf.Variable(tf.ones([128]),name='scale2')
beta2 =  tf.Variable(tf.ones([128]),name='beta2')

conv3_weights = tf.Variable(tf.truncated_normal([1, 1, 128, 256], stddev=0.01),name='conv3_weights')
conv3_biases = tf.Variable(tf.zeros([256]),name='conv3_biases')
scale3 = tf.Variable(tf.ones([256]),name='scale3')
beta3 =  tf.Variable(tf.ones([256]),name='beta3')

conv4_weights = tf.Variable(tf.truncated_normal([1, 1, 256, 512], stddev=0.01),name='conv4_weights')
conv4_biases = tf.Variable(tf.zeros([512]),name='conv4_biases')
scale4 = tf.Variable(tf.ones([512]),name='scale4')
beta4 =  tf.Variable(tf.ones([512]),name='beta4')

conv5_weights = tf.Variable(tf.truncated_normal([1, 1, 512, 1024], stddev=0.01),name='conv5_weights')
# conv5_biases = tf.Variable(tf.constant(0.1, shape=[128]))
conv5_biases = tf.Variable(tf.zeros([1024]),name='conv5_biases')
scale5 = tf.Variable(tf.ones([1024]),name = 'scale5')
beta5 =  tf.Variable(tf.ones([1024]),name = 'beta5')

fc_weights = tf.get_variable('fc_weights',shape=[1024, 512],initializer=tf.contrib.layers.xavier_initializer())
# fc_weights = tf.Variable(tf.truncated_normal([1 * 1 * 128, 256], stddev=0.01))
fc_biases = tf.Variable(tf.constant(0.1, shape=[512]),name='fc_biases')
# fc_biases = tf.get_variable('fc_biases',shape=[512],initializer=tf.contrib.layers.xavier_initializer())

out_weights = tf.get_variable('out_weights',shape=[512, 2],initializer=tf.contrib.layers.xavier_initializer())
# out_weights = tf.Variable(tf.truncated_normal([256, 2], stddev=0.1))
out_biases = tf.Variable(tf.constant(0.1, shape=[2]),name = 'out_biases')
# fc_biases = tf.get_variable('out_biases',shape=[2],initializer=tf.contrib.layers.xavier_initializer())


def convolutional_neural_network(data):


	print('input data',data.get_shape())
	#conv1
	conv1 = tf.nn.conv2d(data,filter =conv1_weights,strides=[1,1,1,1],padding="SAME")
	# tf.image_summary("conv1",conv1)
	addition = tf.add(conv1,conv1_biases)

	hidden = tf.abs(addition,name='none')
	#BN-batch normalization
	batch_mean1, batch_var1 = tf.nn.moments(hidden,[0])

	BN1 = tf.nn.batch_normalization(hidden,batch_mean1,batch_var1,beta1,scale1,variance_epsilon=1e-3)
	#tanh
	hidden = tf.tanh(BN1)
	# hidden = tf.tanh(hidden)
	#avg_pool
	hidden = tf.nn.avg_pool(hidden,ksize=[1,5,5,1],strides=[1,2,2,1],padding='SAME')
	print('conv1_model',hidden.get_shape())

	#conv2
	conv2 = tf.nn.conv2d(hidden, filter=conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
	addition = tf.add(conv2,conv2_biases)
	#BN
	batch_mean2, batch_var2 = tf.nn.moments(addition,[0])

	BN2 = tf.nn.batch_normalization(addition,batch_mean2,batch_var2,beta2,scale2,variance_epsilon=1e-3)
	#tanh
	hidden  = tf.tanh(BN2)
	# hidden  = tf.tanh(addition)
	#avg_pool
	hidden = tf.nn.avg_pool(hidden,ksize=[1,5,5,1],strides=[1,2,2,1],padding='SAME')
	print('conv2_model',hidden.get_shape())
	
	#conv3
	conv3= tf.nn.conv2d(hidden, filter=conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
	addition = tf.add(conv3,conv3_biases)
	#BN
	batch_mean3, batch_var3 = tf.nn.moments(addition,[0])

	BN3 = tf.nn.batch_normalization(addition,batch_mean3,batch_var3,beta3,scale3,variance_epsilon=1e-3)
	#relu
	hidden  = tf.nn.relu(BN3)
	# hidden  = tf.nn.relu(addition)
	#avg_pool
	hidden = tf.nn.avg_pool(hidden,ksize=[1,5,5,1],strides=[1,2,2,1],padding='SAME')
	print('conv3_model',hidden.get_shape())
	#conv4
	conv4= tf.nn.conv2d(hidden, filter=conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
	addition = tf.add(conv4,conv4_biases)
	#BN
	batch_mean4, batch_var4 = tf.nn.moments(addition,[0])
	
	BN4 = tf.nn.batch_normalization(addition,batch_mean4,batch_var4,beta4,scale4,variance_epsilon=1e-3)
	#relu
	hidden  = tf.nn.relu(BN4)
	# hidden  = tf.nn.relu(addition)
	#avg_pool
	hidden = tf.nn.avg_pool(hidden,ksize=[1,5,5,1],strides=[1,2,2,1],padding='SAME')
	print('conv4_model',hidden.get_shape())

	#conv5
	conv5= tf.nn.conv2d(hidden, filter=conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
	addition = tf.add(conv5,conv5_biases)
	#BN
	batch_mean5, batch_var5 = tf.nn.moments(addition,[0])
	
	BN5 = tf.nn.batch_normalization(addition,batch_mean5,batch_var5,beta5,scale5,variance_epsilon=1e-3)
	#relu
	hidden  = tf.nn.relu(BN5)
	# hidden  = tf.nn.relu(addition)
	#avg_pool
	hidden = tf.nn.avg_pool(hidden,ksize=[1,32,32,1],strides=[1,32,32,1],padding='SAME')
	print('conv5_model',hidden.get_shape())

	shape = hidden.get_shape().as_list()
	print('fc_model shape',hidden.get_shape());
	reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
	fc = tf.add(tf.matmul(reshape,fc_weights),fc_biases)
	output = tf.add(tf.matmul(fc,out_weights),out_biases)
	print('output',output.get_shape())

	return output

	
pred = convolutional_neural_network(X)
pred_test = convolutional_neural_network(Z)
train_prediction = tf.nn.softmax(pred,name='train_prediction')
test_prediction = tf.nn.softmax(pred_test,name = 'test_prediction')

# Define loss and optimizer
l2_loss = (tf.nn.l2_loss(fc_weights) + tf.nn.l2_loss(fc_biases) +tf.nn.l2_loss(out_weights) + tf.nn.l2_loss(out_biases))
# l2_loss = tf.nn.l2_loss(conv1_weights)+tf.nn.l2_loss(conv2_weights)+tf.nn.l2_loss(conv3_weights)+tf.nn.l2_loss(conv4_weights)+tf.nn.l2_loss(conv5_weights)+tf.nn.l2_loss(fc_weights)+tf.nn.l2_loss(out_weights)+tf.nn.l2_loss(conv1_biases)+tf.nn.l2_loss(conv2_biases)+tf.nn.l2_loss(conv3_biases)+tf.nn.l2_loss(conv4_biases)+tf.nn.l2_loss(conv5_biases)+tf.nn.l2_loss(fc_biases)+tf.nn.l2_loss(out_biases)
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, Y))+0.001*l2_loss
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, Y))
loss = -tf.reduce_sum(Y*tf.log(train_prediction))+0.001*l2_loss
# loss = -tf.reduce_sum(Y*tf.log(train_prediction))
# loss =tf.reduce_mean(-tf.reduce_sum(Y*tf.log(train_prediction)))
batch = tf.Variable(0,tf.float32)
starter_learning_rate = 0.001
global_step = batch*batch_size
decay_steps = 5000
decay_rate = 0.9
learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,decay_steps,decay_rate,staircase=True)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = batch)
# optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss,global_step = batch)
#adam
# optimizer = tf.train.AdamOptimizer(learning_rate =learning_rate).minimize(loss) 

# Initializing the variables
def error_rate_accuracy(predictions, labels):
	"""Return the error rate based on dense predictions and 1-hot labels."""
	return 100.0 *np.sum(np.argmax(predictions, 1)==np.argmax(labels, 1))/predictions.shape[0]
	# tf.scalar_summary("accuracy",accuracy)
	# return accuracy

if __name__ == '__main__':
	init_op = tf.global_variables_initializer()
	train_imgs,train_labels_,test_imgs,test_labels_= datafile.read_tfrecord_fromfile()
	
	print "test_labels_",test_labels_
	
	# train_batch, train_label_batch = tf.train.shuffle_batch([train_imgs, train_labels_],batch_size=batch_size, capacity=capacity,min_after_dequeue=min_after_dequeue)
	train_batch, train_label_batch = tf.train.batch([train_imgs, train_labels_],batch_size=batch_size, capacity=32)
	print "test_labels_",train_label_batch
	b=tf.one_hot(train_label_batch,2)
	print 'b',b
	# test_batch,test_label_batch = tf.train.shuffle_batch([test_imgs,test_labels_],batch_size=test_batch_size, capacity=capacity,min_after_dequeue=min_after_dequeue)
	test_batch,test_label_batch = tf.train.batch([test_imgs,test_labels_],batch_size=test_batch_size, capacity=32)
	start_time = time.time()
	with tf.Session() as sess:
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord,sess=sess)
		step = 0
		step_test =0
		# Keep training until reach max iterations
		print b.eval()
		try:
			#Training
			print('Start Training......')
			train_preds = []
			train_labels_all = []
			while step * batch_size < training_iters:
				if coord.should_stop():
					break
				images, labels = sess.run([train_batch,train_label_batch])
				print labels
				labels = datafile.convert_one_hot(labels)
				labels_ls = labels.tolist()
				train_labels_all += labels_ls
				_,predictions= sess.run([optimizer,train_prediction],feed_dict={X: images, Y: labels})
				# print('optimizer',sess.run(optimizer,feed_dict={X: images, Y: labels}))
				predictions_ls = predictions.tolist()
				train_preds +=predictions_ls
				# print('Every accuracy: %.1f%%' % error_rate_accuracy(predictions, labels))
				if step % display_step == 0:
					train_preds_m =np.array(train_preds)
					train_labels_all_m = np.array(train_labels_all)
					_loss,lr=sess.run([loss,learning_rate],feed_dict={X: images, Y: labels})
					# print('label_length',len(train_labels_all_m))
					print('learning rate: %.6f' % lr)
					print('train Iter'+str(step*batch_size),"Minibatch Loss="+"{:.6f}".format(_loss))
					print('The Minibatch accuracy: %.1f%%' % error_rate_accuracy(predictions, labels))
					print('Minibatch accuracy: %.1f%%' % error_rate_accuracy(train_preds_m, train_labels_all_m))
				step+=1
			train_preds =np.array(train_preds)
			train_labels_all = np.array(train_labels_all)
			
			print('Train accuracy: %.1f%%' % error_rate_accuracy(train_preds, train_labels_all))
			elapsed_time = time.time() - start_time
			print('Training time is:%.1f ms' %(1000 * elapsed_time))
			print('Optimization Finished!')

			# print('Start Testing......')
			# start_test_time = time.time()
			# test_preds=[]
			# test_labels_all=[]
			# while step_test * batch_size < test_iters:
			# 	if coord.should_stop():
			# 		break
			# 	test_images,test_labels = sess.run([test_batch,test_label_batch])
			# 	test_labels = datafile.convert_one_hot(test_labels)
			# 	# print('test labels',np.array(test_labels))
			# 	test_labels_ls = test_labels.tolist()
			# 	test_labels_all += test_labels_ls
			# 	test_pred = sess.run(test_prediction,feed_dict={Z: test_images})
			# 	test_pred_ls = test_pred.tolist()
			# 	test_preds+=test_pred_ls
			# 	if step_test % display_step == 0:
			# 		test_preds_m =np.array(test_preds)
			# 		test_labels_all_m = np.array(test_labels_all)
			# 		#print(test_preds_m)
			# 		# print(test_labels_all_m)
			# 		print('The Minibatch accuracy: %.1f%%' % error_rate_accuracy(test_pred, test_labels))
			# 		print('Test Minibatch accuracy: %.1f%%' % error_rate_accuracy(test_preds_m, test_labels_all_m))
			# 	step_test+=1
			# test_labels_all = np.array(test_labels_all)
			# test_preds = np.array(test_preds)
			
			# test_error = error_rate_accuracy(test_preds, test_labels_all)
			# print('Test accuracy: %.1f%%' % test_error)
			# test_elapsed_time = time.time() - start_test_time
			# print('Test time is:%.1f ms' %(1000 * test_elapsed_time))
		except tf.errors.OutOfRangeError:
			print('Epochs Complete!')
		else:
			pass
		finally:
			print('finally')
			coord.request_stop()
		coord.join(threads)
		sess.close()

