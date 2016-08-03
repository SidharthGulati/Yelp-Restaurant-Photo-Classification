########################  Author : Sidharth Gulati  ########################  
########################  Affiliation : UCLA(EE)	########################  
########################  University ID : 104588717 ######################## 
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

PRED_DIR = 'test_result_label'
ACTUAL_DIR = 'actual_result_label'
METRICS_FILE  = 'metrics_collected'
PLOTS_DIR = 'plots'
VALID_ACC_DIR = 'valid_acc_results'
TRAIN_ACC_DIR = 'train_acc_results'
ENTROPY_DIR = 'entrpy_results'
eval_step_interval = 10
NUM_CLASSES = 9

def concatenate_results(directory):
	files = os.listdir(os.path.join(os.getcwd(), directory))
	iteration = 0
	filepath = os.path.join(os.getcwd(), directory)
	for file in files:
		if iteration ==0:
			concatenate_values = np.load(os.path.join(filepath, file))
		else:
			concatenate_values = np.vstack((concatenate_values, np.load(os.path.join(filepath, file))))
		iteration += 1
	return concatenate_values.T

def get_metrics(pred, actual):
	f1_score_list = []
	precision_list = []
	recall_list = []
	for label in range(NUM_CLASSES):
		f1_score_list.append(metrics.f1_score( actual[:,label], pred[:,label]))
		precision_list.append(metrics.precision_score( actual[:,label], pred[:,label]))
		recall_list.append(metrics.recall_score( actual[:,label], pred[:,label]))
	return f1_score_list, precision_list, recall_list

def produce_metric_plots(f1_score,precision,recall,validation_metric,training_acc_metric,cross_entropy_metric ):
	# Produce Plots for Recall, Precision and F1 Score
	width=0.2
	ind=np.arange(NUM_CLASSES)
	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, f1_score, width, color='b')
	rects2= ax.bar(ind+width,precision,width,color='g')
	rects3=ax.bar(ind+2*width,recall,width,color='r')
	ax.set_ylabel('Scores')
	ax.set_title('F1-score/Precision/Recall')
	ax.set_xticks(ind + width)
	ax.set_xticklabels(('good_for_lunch', 'good_for_dinner', 'takes_res', 'outdoor_seat', 'rest_is_expensive','has_alcohol','has_table_service','amb_is_classy','good_for_kids'))
	ax.legend((rects1[0], rects2[0],rects3[0]), ('F1-score', 'Precision','Recall'))
	plt.show()
	filepath = os.path.join(os.getcwd(), PLOTS_DIR)
	fig.savefig(os.path.join(filepath, 'F1_precision_recall'))

	for label in range(NUM_CLASSES):
		# Produce Plots for Validation Accuracy
		valid_file = 'validation_acc_'
		x_ax = np.asarray(range(0,validation_metric.shape[0]*eval_step_interval, eval_step_interval))
		fig, ax = plt.subplots(nrows = 1, ncols = 1)
		ax.plot(x_ax, validation_metric[:,label]*100)
		plt.xlabel('Iterations')
		plt.ylabel('Validation Accuracy')
		plt.title('Validation Accuracy vs Number of Iterations')
		fig.savefig(os.path.join(filepath, valid_file + str(label)))
		plt.close(fig)

		# Produce Plots for Training Accuracy
		train_acc = 'training_acc'
		x_ax = np.asarray(range(0,training_acc_metric.shape[0]*eval_step_interval, eval_step_interval))
		fig, ax = plt.subplots(nrows = 1, ncols = 1)
		ax.plot(x_ax, training_acc_metric[:,label]*100)
		plt.xlabel('Iterations')
		plt.ylabel('Training Accuracy')
		plt.title('Training Accuracy vs Number of Iterations')
		fig.savefig(os.path.join(filepath, train_acc + str(label)))
		plt.close(fig)

		# Producing Plots for Cross Entropy metric.
		entropy_file = 'cross_entropy_'
		x_ax = np.asarray(range(0,validation_metric.shape[0]*eval_step_interval, eval_step_interval))
		fig, ax = plt.subplots(nrows = 1, ncols = 1)
		ax.plot(x_ax, validation_metric[:,label])
		plt.xlabel('Iterations')
		plt.ylabel('Cross Entropy')
		plt.title('Cross Entropy vs Number of Iterations')
		fig.savefig(os.path.join(filepath, entropy_file + str(label)))
		plt.close(fig)


# Load Different Metrics for evaluation.
if not os.path.isdir(os.path.join(os.getcwd(), PLOTS_DIR)):
	os.makedirs(os.path.join(os.getcwd(), PLOTS_DIR))
predictions = concatenate_results(PRED_DIR)
print(predictions)
actual = concatenate_results(ACTUAL_DIR)
print(actual)
validation_metric = concatenate_results(VALID_ACC_DIR)
training_acc_metric = concatenate_results(TRAIN_ACC_DIR)
cross_entropy_metric = concatenate_results(ENTROPY_DIR) 
f1_score, precision, recall = get_metrics(predictions, actual)
np.savez(METRICS_FILE, f1_score = f1_score, precision= precision, recall=recall,training_acc_metric=training_acc_metric,
			 cross_entropy_metric=cross_entropy_metric,validation_metric=validation_metric)
# Metrics score for all the labels combined.
print("F1 score = ", f1_score)
print("Precision = ", precision)
print("Recall", recall)
f1_overall=[]
precision_overall=[]
recall_overall=[]
for i in range(predictions.shape[0]):
	f1_overall.append(metrics.f1_score(actual[i,:],predictions[i,:]))
	precision_overall.append(metrics.precision_score(actual[i,:],predictions[i,:]))
	recall_overall.append(metrics.recall_score(actual[i,:],predictions[i,:]))
print("Mean F1-score", np.mean(f1_overall))
print("Mean Precision score", np.mean(precision_overall))
print("Mean recall score", np.mean(recall_overall))
# Produce Metrics plot and save it to directory.
produce_metric_plots(f1_score, precision, recall,validation_metric,training_acc_metric,cross_entropy_metric )