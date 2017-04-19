############################################################################################  
#!/usr/bin/python3.5  
# -*- coding: utf-8 -*-  
#Author  : huDandan  
#Date    : 2016.5.10  
#Function: image convert to tfrecords   
#############################################################################################  
  
import tensorflow as tf  
import numpy as np  
import os  
import os.path  
from PIL import Image
from skimage import data, io, filters
  
 
#Vtrain20_15_20_30 Vtest20_15_20_30
#Vtrain20_15_20_45 Vtest20_15_20_45
#Vtrain20_30_40_30 Vtest20_30_40_30
#Vtrain20_30_60_30 Vtest20_30_60_30

#########length#########
#Vtrain0_0_2_0 Vtest0_0_2_0
#Vtrain20_15_23_15 Vtest20_15_23_15
#Vtrain20_15_25_15 Vtest20_15_25_15

#########theta#########
#Vtrain20_15_20_17 Vtest20_15_20_17
#Vtrain20_15_20_20 Vtest20_15_20_20

#########muti_classes#########
#len
#train_len.csv
#test_len.csv
#theta
#train_theta.csv
#test_theta.csv

#########muti_classes_pred#########
####len
#train_len_pred.csv
#test_len_pred.csv
####theta
#train_theta_pred.csv
#test_theta_pred.csv

###############################################################################################  
train_file_train = 'train_len.csv'   
name_train='train_len.csv'     #train.tfrecords
train_file_test = 'test_len.csv' #  
name_test='test_len'     #train.tfrecords
output_directory='./tfrecords'  
resize_height=32 #  
resize_width=32 #  
###############################################################################################  
def _int64_feature(value):  
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))  
  
def _bytes_feature(value):  
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))  
  
def load_file(examples_list_file):  
    lines = np.genfromtxt(examples_list_file, delimiter=",", dtype=[('col1', 'S120'), ('col2', 'i8')])  
    examples = []  
    labels = []  
    for example, label in lines:  
        examples.append(example)  
        labels.append(label)  
    return np.asarray(examples), np.asarray(labels), len(lines)  
  
def extract_image(filename,  resize_height, resize_width):
    print('extract_filename',filename)

    image = io.imread(filename)

    print(image.size)
    print(image.shape)
    return image  
  
def transform2tfrecord(train_file, name, output_directory, resize_height, resize_width):  
    if not os.path.exists(output_directory) or os.path.isfile(output_directory):  
        os.makedirs(output_directory)  
    _examples, _labels, examples_num = load_file(train_file)  
    filename = output_directory + "/" + name + '.tfrecords'  
    writer = tf.python_io.TFRecordWriter(filename)
    current_path = os.getcwd() 
    for i, [example, label] in enumerate(zip(_examples, _labels)):  
        print('No.%d' % (i))  
        image = extract_image(example.decode(), resize_height, resize_width)
        label=int(label)  
        print('shape: %d, %d, %d, label: %d' % (image.shape[0], image.shape[1], image.shape[2], label))
        print(type(label))  
        image_raw = image.tostring()  
        example = tf.train.Example(features=tf.train.Features(feature={  
            'image_raw': _bytes_feature(image_raw),  
            'height': _int64_feature(image.shape[0]),  
            'width': _int64_feature(image.shape[1]),  
            'depth': _int64_feature(image.shape[2]),  
            'label': _int64_feature(label)  
        }))  
        writer.write(example.SerializeToString())  
    writer.close()  
  
def disp_tfrecords(tfrecord_list_file):  
    filename_queue = tf.train.string_input_producer([tfrecord_list_file])  
    reader = tf.TFRecordReader()  
    _, serialized_example = reader.read(filename_queue)  
    features = tf.parse_single_example(  
        serialized_example,  
 features={  
          'image_raw': tf.FixedLenFeature([], tf.string),  
          'height': tf.FixedLenFeature([], tf.int64),  
          'width': tf.FixedLenFeature([], tf.int64),  
          'depth': tf.FixedLenFeature([], tf.int64),  
          'label': tf.FixedLenFeature([], tf.int64)  
      }  
    )  
    image = tf.decode_raw(features['image_raw'], tf.uint8)  
    #print(repr(image))  
    height = features['height']  
    width = features['width']  
    depth = features['depth']  
    label = tf.cast(features['label'], tf.int32)  
    init_op = tf.global_variables_initializer()  
    resultImg=[]  
    resultLabel=[]  
    with tf.Session() as sess:  
        sess.run(init_op)  
        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  
        for i in range(4):  
            image_eval = image.eval()  
            resultLabel.append(label.eval())  
            image_eval_reshape = image_eval.reshape([height.eval(), width.eval(), depth.eval()])  
            resultImg.append(image_eval_reshape)  
            pilimg = Image.fromarray(np.asarray(image_eval_reshape))  
            pilimg.show()  
        coord.request_stop()  
        coord.join(threads)  
        sess.close()  
    return resultImg,resultLabel  
  
def read_tfrecord(filename_queuetemp):  
    filename_queue = tf.train.string_input_producer([filename_queuetemp])  
    reader = tf.TFRecordReader()  
    _, serialized_example = reader.read(filename_queue)  
    features = tf.parse_single_example(  
        serialized_example,  
        features={  
          'image_raw': tf.FixedLenFeature([], tf.string),
          'height': tf.FixedLenFeature([], tf.int64),  
          'width': tf.FixedLenFeature([], tf.int64),  
          'depth': tf.FixedLenFeature([], tf.int64),  
          'label': tf.FixedLenFeature([], tf.int64)  
      }  
    )  
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    # image  
    image=tf.reshape(image, [256, 256, 3])  
    # normalize  
    image = tf.cast(image, tf.float32) * (1. /255) - 0.5  
    # label  
    label = tf.cast(features['label'], tf.int32) 

    return image, label
    
def convert_one_hot(labels):
    one_hot_labels = []
    for num in labels:
        one_hot = [0,0]
        if num == 0:
            one_hot[0] = 1
        else:
            one_hot[1] = 1
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels).astype(np.int32)
    return labels 
  
def produce_tfrecords(): 
    transform2tfrecord(train_file_train, name_train, output_directory,  resize_height, resize_width) #convert traindata to tfrecords
    transform2tfrecord(train_file_test, name_test, output_directory,  resize_height, resize_width) #convert testdata to tfrecords     
    
    #img,label=disp_tfrecords(output_directory+'/'+name+'.tfrecords') 
def read_tfrecord_fromfile():   
    train_imgs,train_labels=read_tfrecord(output_directory+'/'+name_train+'.tfrecords') #read data from tfrecords
    test_imgs,test_labels=read_tfrecord(output_directory+'/'+name_test+'.tfrecords') #read data from tfrecords

    return train_imgs,train_labels,test_imgs,test_labels
  
if __name__ == '__main__':
    produce_tfrecords()
    # init_op = tf.global_variables_initializer()
    # train_imgs,train_labels_,test_imgs,test_labels=read_tfrecord_fromfile()
    # train_batch, train_label_batch = tf.train.batch([train_imgs, train_labels_],batch_size=4, capacity=32)
    # # Keep training until reach max iterations
    # with tf.Session() as sess:

    #     sess.run(init_op)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord,sess=sess)
    #     try:
    #         print('Start......')
    #         images,labels = sess.run([train_batch,train_label_batch])
    #         print(labels)
    #         print(type(images))
    #     except tf.errors.OutOfRangeError:
    #         print('Epochs Complete!')
    #     else:
    #         pass
    #     finally:
    #         print('finally')
    #         coord.request_stop()
    #     coord.join(threads)
    #     sess.close()
