#!/usr/bin/python
########################  Author : Sidharth Gulati  ########################  
########################  Affiliation : UCLA(EE)	########################  
########################  University ID : 104588717 ########################
from __future__ import print_function  
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow.python.platform import gfile
from PIL import Image
import time

IMAGE_TO_BUZ_ID_FILE = 'train_photo_to_biz_ids.csv'
BUZ_ID_TO_LABEL_FILE = 'train.csv'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_GEN_BATCH_SIZE = 300
NUM_CLASSES = 9
image_size = 227
height = 3

# File System Flags
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('graphDef_file', 'classify_image_graph_def.pb',
                           """Already trained Inception Graph Definition File name.""")
# Configuration required as per platform file system.
tf.app.flags.DEFINE_string('model_dir', '/home/ubuntu',
                           """Path to classify_image_graph_def.pb, """
                           """imagenet_synset_to_human_label_map.txt, and """
                           """imagenet_2012_challenge_label_map_proto.pbtxt.""")
def create_inception_graph():
	""""Creates a graph from saved GraphDef file and returns a Graph object.

  		Returns:
    	Graph holding the trained Inception network, and various tensors we'll be
    	manipulating.

    	Taken from tensorflow tutorial (https://github.com/tensorflow/tensorflow.git)
  	"""
	with tf.Session() as session:
		model_filename = os.path.join(FLAGS.model_dir, FLAGS.graphDef_file)
		with gfile.FastGFile(model_filename, 'rb') as f:
		  graph_def = tf.GraphDef()
		  graph_def.ParseFromString(f.read())
		  bottleneck_tensor = (tf.import_graph_def(graph_def, name='', return_elements= [BOTTLENECK_TENSOR_NAME]))
	return session.graph, bottleneck_tensor

def one_hot_encoding(labels):
	""""Creates a list of one hot encoding of image labels.

  		Returns:
    	List containing one hot encoding for the specified image label.
  	"""
	encoded_labels = [0]*NUM_CLASSES
	for label in labels:
		encoded_labels[label] = 1
	return encoded_labels


def prepare_batch_data(df_img_to_buz_id,df_buz_to_labels, images,img_dir):
	""""Creates a numpy array of batch images in the defined image size, given by, (image_size,image_size,height).

  		Returns:
    	Numpy ndarray containing images for the specified batch number
  	"""
	num_images = 0
	labels = []
	batch_image_arr = np.ndarray(shape=[len(images),image_size,image_size,height],dtype=np.float32)

	for img in images:
		business_id = df_img_to_buz_id.iloc[df_img_to_buz_id.index.values == img].business_id.values
		img_label = str((df_buz_to_labels.iloc[df_buz_to_labels.index.values == business_id])['labels'].values[0]).strip().split(' ')
		label_list = list(map(int, img_label))
		# One hot encoding of labels i.e if labels are 2,5,8 then labels[index] = [0,0,1,0,0,1,0,0,1,0]
		labels.append(one_hot_encoding(label_list))
		# Load Image 
		image_data = Image.open(img_dir+"/train_photos/"+str(img)+".jpg").resize((image_size,image_size), Image.BILINEAR)
		img_arr = np.array(image_data,np.float32).reshape(image_data.size[1],image_data.size[0],height)
		img_arr = (img_arr - np.mean(img_arr)) / np.std(img_arr)
		batch_image_arr[num_images,:,:,:] = img_arr
	return batch_image_arr, labels

def run_bottleneck_on_batch(image_data,bottleneck_tensor):

	"""Runs inference on a batch to extract the 'bottleneck' summary layer for images in the batch.

	Args:
	sess: Current active TensorFlow Session.
	image_data: Numpy array of image batch data.
	bottleneck_tensor: Layer before the final softmax.

	Returns:
	Numpy ndarray of bottleneck values.
	"""
	sess = tf.InteractiveSession()
	n_images = image_data.shape[0]
	bottleneck_list = []
	pool3 = sess.graph.get_tensor_by_name('pool_3:0')
	for i in range(n_images):
	    bottleneck_values = sess.run(pool3,{'DecodeJpeg:0': image_data[i,:]})
	    bottleneck_list.append(np.squeeze(bottleneck_values))
	return bottleneck_list
  


def generate_bottleneck_features(img_dir):
	"""Retrieves or calculates bottleneck values for an image dataset and saves them in batch files in bottleneck folder.
	Args:
    img_dir : Base directory where images and coressponding csv files are there.

  	"""
	df_img_to_buz_id = pd.read_csv(os.path.join(img_dir,IMAGE_TO_BUZ_ID_FILE), index_col =0, usecols = ['photo_id','business_id'])
	df_buz_to_labels = pd.read_csv(os.path.join(img_dir,BUZ_ID_TO_LABEL_FILE), index_col =0, usecols = ['business_id', 'labels'])
	images = df_img_to_buz_id.index.values
	images = np.random.permutation(images)
	num_images = images.shape[0]
	num_of_batches = (num_images / float(BOTTLENECK_GEN_BATCH_SIZE))
	graph, bottleneck_tensor = create_inception_graph()
	if isinstance(num_of_batches,float):
		num_of_batches = int(num_of_batches) + 1
	print("Total number of batches is = %i"%num_of_batches)
	if not os.path.isdir(os.path.join(os.getcwd(), 'bottleneck')):
		os.makedirs(os.path.join(os.getcwd(), 'img_label'))
		os.makedirs(os.path.join(os.getcwd(), 'bottleneck'))
		print("Bottleneck Values not found. Generating them.")
		for batch in range(num_of_batches):
			if batch == num_of_batches - 1:
				batch_images = images[(batch*BOTTLENECK_GEN_BATCH_SIZE):]
			else:
				batch_images = images[(batch*BOTTLENECK_GEN_BATCH_SIZE):(batch*BOTTLENECK_GEN_BATCH_SIZE + BOTTLENECK_GEN_BATCH_SIZE)]
			print("Preparing Image data for Batch %i having size %i" % (batch+1,batch_images.shape[0]))
			bottleneck_filename = 'bottleneck/img_bottleneck_batch_'
			label_filename = 'img_label/img_label_batch_'
			image_batch_arr, batch_labels = prepare_batch_data(df_img_to_buz_id,df_buz_to_labels, batch_images,img_dir)
			print("Genrating Bottleneck Values for Batch %i " % (batch+1))
			# Create bottleneck Values for the batch
			start = time.time()
			bottleneck_values = run_bottleneck_on_batch(image_batch_arr,bottleneck_tensor)
			print("Generation completed in ", time.time() - start,"seconds")
			bottleneck_filename = bottleneck_filename+str(batch+1)
			label_filename = label_filename + str(batch+1)
			print("Saving Bottleneck Values and labels for Batch %i with file name '%s' and '%s' resp." % (batch+1, bottleneck_filename+'.npy',
						label_filename+'.npy'))
			np.save(bottleneck_filename, bottleneck_values)
			np.save(label_filename, batch_labels)

generate_bottleneck_features(os.getcwd())