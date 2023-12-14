import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import h5py
import random
import math
import gpytorch
from sklearn.metrics import confusion_matrix, classification_report

class MulticlassCrossEntropyLoss(nn.Module):
	def __init__(self):
		super(MulticlassCrossEntropyLoss, self).__init__()
#		self.weights = weights
	def forward(self, y_pred, y_true):
		# Compute the cross entropy loss
		loss = - torch.sum( y_true * torch.log(y_pred), dim = 0 ).sum()
		return loss

def load_passbands_list(path, objid, path_enter):
	passbands_list = []
	hf = h5py.File(path, "r")
	for i in range(6):
		x = torch.tensor( np.array( hf.get("objid_" + str(objid) + "_passband_" + str(i)) ), dtype = torch.float32 )
		x = x[:,0:2]
		passbands_list.append( x )
	hf.close()
	training_set_metadata = pd.read_csv(path_enter + "training_set_metadata.csv")
	label = int(training_set_metadata.loc[training_set_metadata["object_id"] == objid,:]["target"])
	return passbands_list, label

def plot_passbands(data_obj):
	# Create a figure and axis
	fig, axs = plt.subplots(2, 3, figsize = (12, 6), constrained_layout = True)
	for j in range(len(data_obj)):
		# Sample data
		data = data_obj[j][:,1:2]
		# "x" axis values
		x_axis = data_obj[j][:,0:1]
		x_index, y_index = 0 if j <= 2 else 1, j if  j <= 2 else j - 3
		# Iterate over the data
		for i, value in enumerate(data):
			# Set the color based on the sign of the value
			color = 'green' if value >= 0 else 'red'
			# Plot the bar from 0 to the value
			axs[x_index,y_index].bar(x_axis[i], value, color=color)
		axs[x_index,y_index].axhline(0, color='black', linewidth=0.8)
		# Set y-axis limits
		min_value = min(data)
		max_value = max(data)
		axs[x_index,y_index].set_ylim(min_value if min_value < 0 else 0, max_value if max_value > 0 else 0)
		axs[x_index,y_index].set_xlabel("MJD")
		axs[x_index,y_index].set_title("Passband_" + str(j))
	# Show the plot
	plt.show()

def preprocess_data_obj(data_obj):
	# Get total tensor of times
	tensor_vector = [ obj[:,0:1] for obj in data_obj ]
	tensor_vector = torch.cat(tensor_vector, dim = 0)
	
	# Get times and sort it from minimum to maximum
	times, indices = torch.sort( tensor_vector.reshape(1,-1) )
	times = times.reshape(-1,1)
	
	# Generate 6 empty tensors of the shape of the tensor of times. Fill it with zeroes.
	passband_tensors = [ torch.cat( (times[:,0:1],torch.zeros(times.shape[0],1)), dim = 1 ) for i in range(6) ]
	
	for j in range(times.shape[0]):
		# Specify one time
		t = times[j]
		# Run over all passbands to check where this value of time is located
		for i in range(len(data_obj)):
			sumation = torch.sum( data_obj[i][:,0:1] == t )
			if float(sumation) == 1:
				passband_index = i
				break
		passband_tensors[passband_index][j,1] = data_obj[passband_index][torch.nonzero( torch.eq( data_obj[passband_index], t ) ).squeeze()[0],1]
	return torch.stack(passband_tensors, dim = 0).permute(2, 1, 0)

def get_sets(path_enter):
	training_set_metadata = pd.read_csv(path_enter + "training_set_metadata.csv")
	object_ids = np.unique(training_set_metadata["object_id"]).tolist()
	all_objects, labels = [], []
	for i in tqdm( range( len(object_ids) ) ):
		data_obj, label = load_passbands_list(path = path_enter + "Data_as_h5/obj_passbands.h5", objid = object_ids[i], path_enter = path_enter)
		preprocessed_data_obj = preprocess_data_obj(data_obj)
		all_objects.append( preprocessed_data_obj[1:2,:,:] )
		labels.append( label )
	return all_objects, labels

def split_sets(all_objects, labels):
	random.seed(666)
	integer_list = list( range(len(all_objects)) )
	random.shuffle(integer_list)
	## Define the size of each of the three new lists
	size_training = int(len(integer_list) * 0.75)
	size_validation = int(len(integer_list) * 0.05)
	size_test = int(len(integer_list) * 0.2)
	## Split the shuffled list into three new lists
	list_training = integer_list[:size_training]
	list_validation = integer_list[size_training:size_training+size_validation]
	list_test = integer_list[size_training+size_validation:]
	
	X_training, Y_training = [all_objects[i] for i in list_training], [labels[i] for i in list_training]
	X_val, Y_val = [all_objects[i] for i in list_validation], [labels[i] for i in list_validation]
	X_test, Y_test = [all_objects[i] for i in list_test], [labels[i] for i in list_test]
	
	return X_training, Y_training, X_val, Y_val, X_test, Y_test

def plot_results(model, path):
	# PLOT OF THE LOSSES
	fig, ax = plt.subplots(1, 1, figsize = (6, 3), constrained_layout = True)
	## Plot of physical losses
	ax.semilogy( range(len(model.loss_training_hist)), model.loss_training_hist, "k-", label = "Training loss" )
	ax.semilogy( range(len(model.loss_validation_hist)), model.loss_validation_hist, "b--", label = "Val. loss" )
	ax.legend()
	ax.set_xlabel("Epoch")
	plt.savefig(path + "Losses_vs_epoch.png")
	plt.close()
	# PLOT OF METRICS
	fig, ax = plt.subplots(1, 1, figsize = (6, 3), constrained_layout = True)
	## Plot of physical losses
	ax.semilogy( range(len(model.accuracy_training)), model.accuracy_training, "b-", label = "Training Accuracy" )
	ax.semilogy( range(len(model.accuracy_val)), model.accuracy_val, "b--", label = "Val. Accuracy" )
	ax.legend()
	ax.set_xlabel("Epoch")
	plt.savefig(path + "Metrics_vs_epoch.png")
	plt.close()

#def save_results(model, path):
#	hf = h5py.File("", "w")

def train_GP_and_eval(model, data_obj, passband, new_time, likelihood, num_iterations = 1000):
	# Set the model to training mode and initialize hyperparameters
	model.train()
	likelihood.train()
	# Define the optimizer and loss function
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.05)
	# "Loss" for GPs - the marginal log likelihood
	mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
	# Optimize the model
	loss_hist = []
	for i in range(num_iterations):
		optimizer.zero_grad()
		output = model(data_obj[passband][:,0:1].flatten())
		loss = -mll(output, data_obj[passband][:,1:2].flatten())
		loss.backward()
		loss_hist.append( float( loss.detach().cpu().numpy() ) )
		optimizer.step()
	# Get into evaluation (predictive posterior) mode
	model.eval()
	likelihood.eval()
	# Make predictions by feeding model through likelihood
	with torch.no_grad(), gpytorch.settings.fast_pred_var():
		test_x = new_time
		observed_pred = likelihood( model(test_x) )
	observed_pred = observed_pred.mean
	return observed_pred

# Define a function to plot a 2D confussion matrix
def plot_cm(model, cm_percent, dataset):
	# Plot the heatmap
	plt.imshow(cm_percent, cmap="Blues")
	# Add colorbar
	plt.colorbar()
	# Add axis labels
	plt.xlabel("Predicted Label")
	plt.ylabel("True Label")
	plt.xticks(np.arange(len(model.labeling_order)), model.labeling_order, rotation=45)
	plt.yticks(np.arange(len(model.labeling_order)), model.labeling_order)
	# Add title
	plt.title("Confusion Matrix " + dataset)
	# Add numerical values on top of the cells
	for i in range(cm_percent.shape[0]):
		for j in range(cm_percent.shape[1]):
			plt.text(j, i, round(cm_percent[i, j], 3), ha='center', va='center', color='black')
	# Save the plot
	plt.savefig("Images/confusion_matrix_" + dataset + ".png")
	plt.close("all")

# Define a function to plot a classification report from sklearn
def plot_report(report, dataset):
	df = pd.DataFrame(report).transpose()
	
	plt.figure(figsize=(8, 6))
	plt.subplots_adjust(left=0.2, top=0.8, bottom=0.2)
	plt.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center', colColours=['lightgray']*len(df.columns), bbox=[0, 0, 1, 1])
	plt.axis('off')  # Hide axis
	plt.title('Classification Report')
	
	plt.savefig("Images/classification_report_" + dataset + ".png")
	plt.close("all")

# Define a function to compute several metrics
## model: model to use
## X: Input tensor
## Y: Targets tensor
## bs, nb: batch size and number of batches, respectively.
def compute_metrics_single_input(model, X, Y, bs, nb):
	logits_pred_list, y_pred_list, y_true_list = [], [], []
	real_labeling = torch.cat( [ one_hot_encode(model = model, label_ground = Y[i]) for i in range(Y.shape[0]) ], dim = 0 ).to(model.device)
	for i in tqdm(range(nb)):
#		logits_pred = model( X[i*bs:(i+1)*bs] )
		y_pred = torch.argmax( model( X[i*bs:(i+1)*bs] ), dim = 1 ).view(-1,1)
		y_true = torch.argmax( real_labeling[i*bs:(i+1)*bs], dim = 1 ).view(-1,1)
#		logits_pred_list.append( logits_pred )
		y_pred_list.append( y_pred )
		y_true_list.append( y_true )
#	logits_pred_list = torch.cat( logits_pred_list, dim = 0 )
	y_pred_list = torch.cat( y_pred_list, dim = 0 )
	y_true_list = torch.cat( y_true_list, dim = 0 )
	
	# Metrics
	y_pred_numpy, y_true_numpy = y_pred_list.view(1,-1).detach().cpu().numpy()[0], y_true_list.view(1,-1).detach().cpu().numpy()[0]
	labeling_order_string = [str(label) for label in model.labeling_order]
	report = classification_report(y_true_numpy.tolist(), y_pred_numpy.tolist(), target_names = labeling_order_string, output_dict = True)
	## Confusion matrix
	cm = confusion_matrix(y_true_numpy, y_pred_numpy)
	## Raw accuracy
	accuracy = (y_pred_list == y_true_list).sum() / y_pred_list.shape[0]
	return report, cm, accuracy

# Define a function to one-hot encode the labeling
def one_hot_encode(model, label_ground):
	num_classes = len(model.labeling_order)
	label_index = model.labeling_order.index(label_ground)
	one_hot = torch.zeros(1, num_classes)
	one_hot[0, label_index] = 1
	return one_hot
















































