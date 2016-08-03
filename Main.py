from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
from sklearn.cross_validation import train_test_split
from datetime import datetime
from tensorflow.python.platform import gfile

BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
classes = ['invalid', 'valid']
n_epochs = 20
FLAGS = tf.app.flags.FLAGS
NUM_CLASSES = 9
BATCH_SIZE = 256
tf.app.flags.DEFINE_string('final_tensor_name', 'final_result',
                           """The name of the output classification layer in"""
                           """ the retrained graph.""")
tf.app.flags.DEFINE_integer('eval_step_interval', 10,
                            """How often to evaluate the training results.""")
tf.app.flags.DEFINE_integer('learning_rate', 0.01,
                            """Learning rate for the optimizer.""")
tf.app.flags.DEFINE_string('model_dir', '/home/ubuntu',
                           """Path to classify_image_graph_def.pb, """
                           """imagenet_synset_to_human_label_map.txt, and """
                           """imagenet_2012_challenge_label_map_proto.pbtxt.""")
def create_inception_graph():
	""""Creates a graph from saved GraphDef file and returns a Graph object.

	Returns:
	Graph holding the trained Inception network, and various tensors we'll be
	manipulating.
	"""
	with tf.Session() as sess:
		model_filename = os.path.join(FLAGS.model_dir, 'classify_image_graph_def.pb')
		with gfile.FastGFile(model_filename, 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			bottleneck_tensor = (tf.import_graph_def(graph_def, name='', return_elements=[BOTTLENECK_TENSOR_NAME]))
	return sess.graph, bottleneck_tensor

def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor):
	"""Adds a new softmax and fully-connected layer for training.

	We need to retrain the top layer to identify our new classes, so this function
	adds the right operations to the graph, along with some variables to hold the
	weights, and then sets up all the gradients for the backward pass.

	The set up for the softmax and fully-connected layers is based on:
	https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

	Args:
	class_count: Integer of how many categories of things we're trying to
	recognize.
	final_tensor_name: Name string for the new final node that produces results.
	bottleneck_tensor: The output of the main CNN graph.

	Returns:
	The tensors for the training and cross entropy results, and tensors for the
	bottleneck input and ground truth input.
	"""
	with tf.name_scope('input'):
		bottleneck_input = tf.placeholder_with_default(bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],name='BottleneckInputPlaceholder')

		ground_truth_input = tf.placeholder(tf.float32,[None, class_count],name='GroundTruthInput')

	# Organizing the following ops as `final_training_ops` so they're easier
	# to see in TensorBoard
	layer_name = 'final_training_ops'
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			layer_weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001), name='final_weights')
			
	with tf.name_scope('biases'):
		layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
	
	with tf.name_scope('Wx_plus_b'):
		logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
	

	final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
	tf.histogram_summary(final_tensor_name + '/activations', final_tensor)

	with tf.name_scope('cross_entropy'):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, ground_truth_input)
	with tf.name_scope('total'):
		cross_entropy_mean = tf.reduce_mean(cross_entropy)
	
	with tf.name_scope('train'):
		train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy_mean)

	return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
	final_tensor)

def add_evaluation_step(result_tensor, ground_truth_tensor):
	"""Inserts the operations we need to evaluate the accuracy of our results.

	Args:
	result_tensor: The new final node that produces results.
	ground_truth_tensor: The node we feed ground truth data
	into.

	Returns:
	Nothing.
	"""
	correct_prediction = tf.equal(tf.argmax(result_tensor, 1), tf.argmax(ground_truth_tensor, 1))
	evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return evaluation_step

def concatenate_values(filename, from_path, to_path):
	"""Concatenate the batch bottleneck and label values and then saves them to the disk.

	Returns:
	Concatenated values.
	"""
	print("Concatenating %s Values..."%from_path)
	if not os.path.isdir(os.path.join(os.getcwd(), to_path)):
		os.makedirs(os.path.join(os.getcwd(), to_path))
	files = os.listdir(os.path.join(os.getcwd(), from_path))
	iteration = 0
	concatenate_values = []
	for file in files:
		file_path = os.path.join(os.path.join(os.getcwd(), from_path), file)
		if iteration == 0:
			concatenate_values = np.load(file_path)
		else:
			concatenate_values = np.vstack((concatenate_values, np.load(file_path) ))
		iteration += 1
	save_path = os.path.join(os.getcwd(), to_path)
	np.save(os.path.join(save_path, filename), concatenate_values)
	return concatenate_values

def cache_concatenated_values():
	"""Cache the Concatenated bottleneck and label values in the memory for further use.

	Returns:
	Concatenated bottleneck and label values.
	"""
	concatenated_bottleneck_values = concatenate_values('conc_bottleneck_values','bottleneck','concatenated_values')
	concatenated_labels = concatenate_values('conc_labels','img_label','concatenated_values')
	return concatenated_bottleneck_values, concatenated_labels

def get_one_hot_encoded_per_label(label_list, label):
	 batch_labels = np.asarray(label_list)[:,label]
	 encoded_labels = (np.eye(len(classes))[batch_labels]).tolist()
	 return encoded_labels

def do_train(sess,bottleneck_values,labels, current_label):
	if (current_label == 0):
		file_name = 'data/'
		if not os.path.isdir(os.path.join(os.getcwd(), 'data')):
			os.makedirs(os.path.join(os.getcwd(), 'data'))
		# Bifurcating data in training, validation and testing datasets.
		shuffle = np.random.permutation(np.arange(len(bottleneck_values)))
		shuffled_bottleneck_values = bottleneck_values[shuffle]
		shfuffled_labels = labels[shuffle]
		X, X_test, y, y_test = train_test_split(shuffled_bottleneck_values, shfuffled_labels, test_size=0.30, random_state=42)
		X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=42)
		# Deleting unused variables for saving memory space.
		del X
		del y
		np.save(file_name+'X_train', X_train)
		np.save(file_name+'X_test', X_test)
		np.save(file_name+'X_valid', X_valid)
		np.save(file_name+'Y_train', y_train)
		np.save(file_name+'Y_test', y_test)
		np.save(file_name+'Y_valid', y_valid)
	else:
		X_train = np.load('data/X_train.npy')
		X_test = np.load('data/X_test.npy')
		X_valid = np.load('data/X_valid.npy')
		y_train = np.load('data/Y_train.npy')
		y_valid = np.load('data/Y_valid.npy')
		y_test = np.load('data/Y_test.npy')
	# Logging the shape of the dataset used.
	print("Training dataset shape = ", X_train.shape)
	print("Training labels shape = ", y_train.shape)
	print("Validation dataset shape =  ", X_valid.shape)
	print("Validation labels shape =  ", y_valid.shape)
	print("Testing dataset shape = ", X_test.shape)
	print("Testing labels shape = ", y_test.shape)

	X_valid = X_valid.tolist()
	X_test = X_test.tolist()
	# Set up the pre-trained graph.
	graph, bottleneck_tensor = create_inception_graph()
	bottleneck_tensor = bottleneck_tensor[0]
	# Add the new layer that we'll be training.
	(train_step, cross_entropy, bottleneck_input, ground_truth_input,
	final_tensor) = add_final_training_ops(len(classes),FLAGS.final_tensor_name,bottleneck_tensor)
	# Create the operations we need to evaluate the accuracy of our new layer.
	evaluation_step = add_evaluation_step(final_tensor, ground_truth_input)
	  # Set up all our weights to their initial default values.
	init = tf.initialize_all_variables()
	sess.run(init)
	num_steps = int((n_epochs * X_train.shape[0] / float(BATCH_SIZE)) + 1)
	label_list_iter = [current_label]
	for label in label_list_iter:
		filename_pred = "test_result_label/pred_test_res_class_"
		filename_actual = "actual_result_label/actual_test_res_class_"
		filename_valid = "valid_acc_results/valid_acc_label_"
		filename_train = "train_acc_results/train_acc_label_"
		filename_entropy = "entrpy_results/entropy_label_"
		if not os.path.isdir(os.path.join(os.getcwd(), 'test_result_label')):
			os.makedirs(os.path.join(os.getcwd(), 'test_result_label'))
		if not os.path.isdir(os.path.join(os.getcwd(), 'actual_result_label')):
			os.makedirs(os.path.join(os.getcwd(), 'actual_result_label'))
		if not os.path.isdir(os.path.join(os.getcwd(), 'valid_acc_results')):
			os.makedirs(os.path.join(os.getcwd(), 'valid_acc_results'))
		if not os.path.isdir(os.path.join(os.getcwd(), 'train_acc_results')):
			os.makedirs(os.path.join(os.getcwd(), 'train_acc_results'))
		if not os.path.isdir(os.path.join(os.getcwd(), 'entrpy_results')):
			os.makedirs(os.path.join(os.getcwd(), 'entrpy_results'))
		print("Label %i, Number of Epochs: %i" %(label,n_epochs))
		validation_accuracy_list = []
		training_accuracy_list = []
		cross_entropy_list = []
		for step in range(num_steps):
			offset = (step * BATCH_SIZE) % (y_train.shape[0] - BATCH_SIZE)
			train_batch = (X_train[offset:(offset + BATCH_SIZE)]).tolist()
			batch_labels = y_train[offset:(offset + BATCH_SIZE)]
			y_one_hot_train = get_one_hot_encoded_per_label(y_train[offset:(offset + BATCH_SIZE)], label)
			y_one_hot_valid = get_one_hot_encoded_per_label(y_valid, label)
			# Training step.
			sess.run([train_step],feed_dict={bottleneck_input: train_batch,
			    ground_truth_input: y_one_hot_train})
			is_last_step = (step + 1 == num_steps)
			if step % FLAGS.eval_step_interval == 0 or is_last_step:
				# Calculate the cross entropy and training accuracy.
				train_accuracy, cross_entropy_value = sess.run([evaluation_step, cross_entropy],
					feed_dict={bottleneck_input: train_batch,ground_truth_input: y_one_hot_train})
				 # Run a validation step and calculate validation accuracy.
				validation_accuracy = sess.run(evaluation_step,feed_dict={bottleneck_input: X_valid,ground_truth_input: y_one_hot_valid})
				print('%s: Step %d: Train accuracy = %.2f%%, Cross entropy = %f, Validation accuracy = %.1f%%' %
						 (datetime.now(), step, train_accuracy * 100, cross_entropy_value, validation_accuracy * 100))
				training_accuracy_list.append(train_accuracy)
				cross_entropy_list.append(cross_entropy_value)
				validation_accuracy_list.append(validation_accuracy)
		# Calculating test accuracy for label.
		y_one_hot_test = get_one_hot_encoded_per_label(y_test, label)
		probabilities, test_accuracy = sess.run([final_tensor, evaluation_step],feed_dict={bottleneck_input: 
			X_test,ground_truth_input: y_one_hot_test})
		print('%s: Label: %i. Final Test  accuracy = %.2f%%' %(datetime.now(), label,test_accuracy * 100))
		# Calculating the predicted class for the test dataset for corresponding label.
		prediction = np.argmax(probabilities,1)
		filename_pred = filename_pred + str(label)
		filename_actual = filename_actual + str(label)
		np.save(filename_pred, prediction)
		np.save(filename_actual,y_test[:,label])
		np.save(filename_train + str(label), training_accuracy_list)
		np.save(filename_valid + str(label), validation_accuracy_list)
		np.save(filename_entropy + str(label), cross_entropy_list)
	print("Completed.")

session = tf.Session()
current_label = 1
print("Concatenating and Caching values...")
bottleneck_values, labels = cache_concatenated_values()
print("Bottleneck Values shape = (%i,%i)" %(len(bottleneck_values),len(bottleneck_values[0])))
print("Labels shape = (%i,%i)" %(len(labels),len(labels[0])))
print("Values cached. Now performing training..")
do_train(session, bottleneck_values, labels, current_label)
