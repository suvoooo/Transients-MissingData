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
import utils

# Gaussian Interpolator model
# Define the GPyTorch ExactGP model
class ExactGPModel(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, likelihood):
		super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = gpytorch.kernels.ScaleKernel(
			gpytorch.kernels.MaternKernel() + gpytorch.kernels.RBFKernel() + gpytorch.kernels.RQKernel()
		)
	def forward(self, x):
		mean = self.mean_module(x)
		covar = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean, covar)

# Convolutional Model Classifier
class ConvClassifier(nn.Module):
	def __init__(self, input_dim, n_classes, num_layers, weights_tensor, smoothing_window_sizes, downsampling_coeffs, DTYPE, device, labeling_order = (6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95)):
		super(ConvClassifier, self).__init__()
		self.input_dim = input_dim
		self.n_classes = n_classes
		self.num_layers = num_layers
		self.weights_tensor = weights_tensor
		self.DTYPE = DTYPE
		self.device = device
		self.labeling_order = labeling_order
		self.smoothing_window_sizes = smoothing_window_sizes
		self.downsampling_coeffs = downsampling_coeffs
		# Define criterion (loss function)
		self.criterion = nn.CrossEntropyLoss()
		# Define lists to save losses
		self.loss_training_hist, self.loss_validation_hist = [], []
		# Neural architecture
		## Smoothing part
		self.conv_layers = []
		for i in range( len(self.smoothing_window_sizes) ):
			conv_layer = nn.Conv1d(in_channels = self.input_dim, out_channels = self.input_dim, kernel_size = self.smoothing_window_sizes[i], stride = 1, padding = self.smoothing_window_sizes[i] // 2, bias = False)
			conv_layer.weight.data.fill_(1 / self.smoothing_window_sizes[i])
			self.conv_layers.append( conv_layer.to(device) )
		### Register the parameters of the convolutionals
		self.conv_layers_params = nn.ModuleList( self.conv_layers )
		## After the process of transformation, define 3 convolutional layers for each branch.
		self.conv_layers_identity = nn.Conv1d(in_channels = self.input_dim, out_channels = 16, kernel_size = 3, stride = 1 ).to(device)
		self.conv_layers_smoothed = nn.Conv1d(in_channels = self.input_dim * len(self.downsampling_coeffs), out_channels = 16, kernel_size = 3, stride = 1 ).to(device)
		self.conv_layers_downsampled = nn.Conv1d(in_channels = self.input_dim, out_channels = 16, kernel_size = 3, stride = 1 ).to(device)
		## Finally, create a list of convolutionals after the transformation and concatenation
		self.conv_layers_final, self.maxpoolings1d_list = [], []
		for i in range(self.num_layers):
			self.conv_layers_final.append( nn.Conv1d(in_channels = 16, out_channels = 16, kernel_size = 1, stride = 5).to(device) )
			self.maxpoolings1d_list.append( nn.MaxPool1d(kernel_size = 2, stride = 1).to(device) )
		self.conv_layers_final_params = nn.ModuleList( self.conv_layers_final )
		# Add a set of fully connected layers in order to decrease the dimensionality
		self.fc_1 = nn.Linear(0, 256).to(device)
		self.fc_2 = nn.Linear(256, 128).to(device)
		self.fc_3 = nn.Linear(128, 64).to(device)
		# At the end we have a Linear layer to map to (1,n_classes)
		self.fc_out = nn.Linear(64, self.n_classes).to(device)
		# Define lists to save losses
		self.loss_training_hist, self.loss_validation_hist = [], []
		# Define lists to save metrics
		self.accuracy_training, self.accuracy_val = [], []
		# Define criterion
		self.criterion = utils.MulticlassCrossEntropyLoss()
	def one_hot_encode(self, label_ground):
		num_classes = len(self.labeling_order)
		label_index = self.labeling_order.index(label_ground)
		one_hot = torch.zeros(1, num_classes)
		one_hot[0, label_index] = 1
		return one_hot
	def forward(self, x):
		# Forward method will consist of a first part of transformation of the input:
		## Identity mapping: return the original input
		identity_mapping = x
		## Smoothing original series with a moving average with various windows sizes
		smoothed_outputs = []
		for conv_layer in self.conv_layers:
			time_series = x.permute(0, 2, 1)
			time_series = nn.ReLU()( conv_layer(time_series) )
			smoothed_series = time_series.permute(0, 2, 1)
			smoothed_outputs.append( smoothed_series )
		smoothed_outputs = torch.cat( smoothed_outputs, dim = 2 )
		## Downsampling of the original series
		downsampled_outputs = []
		for factor in self.downsampling_coeffs:
			time_series = x.permute(0, 2, 1)
			downsampled_ts = F.avg_pool1d(time_series, factor)
			downsampled_ts = downsampled_ts.permute(0, 2, 1)
			downsampled_outputs.append( downsampled_ts )
		downsampled_outputs = torch.cat(downsampled_outputs, dim = 1)
		# After the transformation:
		identity_mapping = nn.ReLU()( self.conv_layers_identity( identity_mapping.permute(0, 2, 1) ).permute(0, 2, 1) )
		smoothed_outputs = nn.ReLU()( self.conv_layers_smoothed( smoothed_outputs.permute(0, 2, 1) ).permute(0, 2, 1) )
		downsampled_outputs = nn.ReLU()( self.conv_layers_downsampled( downsampled_outputs.permute(0, 2, 1) ).permute(0, 2, 1) )
		# Concatenate all outputs
		concatenated = torch.cat( (identity_mapping, smoothed_outputs, downsampled_outputs), dim = 1 )
		for i in range( len(self.conv_layers_final) ):
			concatenated = nn.ReLU()( self.conv_layers_final[i]( concatenated.permute(0, 2, 1) ).permute(0, 2, 1) )
			concatenated = self.maxpoolings1d_list[i]( concatenated.permute(0, 2, 1) ).permute(0, 2, 1)
		concatenated = torch.reshape( concatenated, ( identity_mapping.shape[0], -1 ) )
		self.fc_1 = nn.Linear(concatenated.shape[1], 256).to(self.device)
		concatenated = nn.ReLU()( self.fc_1(concatenated) )
		concatenated = nn.ReLU()( self.fc_2(concatenated) )
		concatenated = nn.ReLU()( self.fc_3(concatenated) )
		concatenated = self.fc_out(concatenated)
		logits = F.softmax(concatenated, dim = 1)
		return logits
	def compute_metrics(self, X, Y, bs, nb):
		y_pred_list, y_true_list = [], []
		real_labeling = torch.cat( [ self.one_hot_encode(label_ground = Y[i] ) for i in range(Y.shape[0]) ], dim = 0 ).to(self.device)
		for i in tqdm(range(nb)):
			y_pred = torch.argmax( self( X[i*bs:(i+1)*bs,1,:].to(self.device) ), dim = 1 ).view(-1,1)
			y_true = torch.argmax( real_labeling[i*bs:(i+1)*bs], dim = 1 ).view(-1,1)
			y_pred_list.append( y_pred )
			y_true_list.append( y_true )
		y_pred_list = torch.cat( y_pred_list, dim = 0 )
		y_true_list = torch.cat( y_true_list, dim = 0 )
		# Metrics
		accuracy = (y_pred_list == y_true_list).sum() / y_pred_list.shape[0]
		return accuracy
	def train_step(self, X, Y, optimizer):
		real_labeling = torch.cat( [ self.one_hot_encode(label_ground = Y[i] ) for i in range(Y.shape[0]) ], dim = 0 ).to(self.device)
		output = self( X )
		loss = self.criterion(output, real_labeling)
		loss.backward()
		torch.nn.utils.clip_grad_value_(self.parameters(), 0.5)
		torch.nn.utils.clip_grad_value_(self.parameters(), -0.5)
		optimizer.step()
		optimizer.zero_grad(set_to_none = True)
		return loss

class ConvClassifier_2(nn.Module):
	def __init__(self, input_dim, n_classes, num_layers, weights_tensor, smoothing_window_sizes, downsampling_coeffs, DTYPE, device, labeling_order = (6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95)):
		super(ConvClassifier, self).__init__()
		self.input_dim = input_dim
		self.n_classes = n_classes
		self.num_layers = num_layers
		self.weights_tensor = weights_tensor
		self.DTYPE = DTYPE
		self.device = device
		self.labeling_order = labeling_order
		self.smoothing_window_sizes = smoothing_window_sizes
		self.downsampling_coeffs = downsampling_coeffs
		# Define criterion (loss function)
		self.criterion = nn.CrossEntropyLoss()
		# Define lists to save losses
		self.loss_training_hist, self.loss_validation_hist = [], []
		# Neural architecture
		self.conv1 = nn.Conv1d(in_channels=6, out_channels=32, kernel_size=3)
		self.relu1 = nn.ReLU()
		self.maxpool1 = nn.MaxPool1d(kernel_size=2)
		
		self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
		self.relu2 = nn.ReLU()
		self.maxpool2 = nn.MaxPool1d(kernel_size=2)
		
		self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
		self.relu3 = nn.ReLU()
		self.maxpool3 = nn.MaxPool1d(kernel_size=2)
		
		self.flatten = nn.Flatten()
		
		self.fc1 = nn.Linear(128 * 249, 256)
		self.relu4 = nn.ReLU()
		
		self.fc2 = nn.Linear(256, num_classes)
	def one_hot_encode(self, label_ground):
		num_classes = len(self.labeling_order)
		label_index = self.labeling_order.index(label_ground)
		one_hot = torch.zeros(1, num_classes)
		one_hot[0, label_index] = 1
		return one_hot
	def forward(self, x):
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)
		
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)
		
		x = self.conv3(x)
		x = self.relu3(x)
		x = self.maxpool3(x)
		
		x = self.flatten(x)
		
		x = self.fc1(x)
		x = self.relu4(x)
		
		x = self.fc2(x)
		
		return F.softmax(x, dim = 1)
	def train_step(self, X, Y, optimizer):
		real_labeling = torch.cat( [ self.one_hot_encode(label_ground = Y[i] ) for i in range(Y.shape[0]) ], dim = 0 ).to(self.device)
		output = self( X )
		loss = self.criterion(output, real_labeling)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad(set_to_none = True)
		return loss

# TransformerClassifier
class TransformerClassifier(nn.Module):
	def __init__(self, input_dim, n_classes, d_model, nhead, num_layers, weights_tensor, DTYPE, device, labeling_order = (6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95)):
		super(TransformerClassifier, self).__init__()
		self.input_dim = input_dim
		self.n_classes = n_classes
		self.d_model = d_model
		self.nhead = nhead
		self.num_layers = num_layers
		self.labeling_order = labeling_order
		self.weights_tensor = weights_tensor
		self.DTYPE = DTYPE
		self.device = device
		
		self.encoder = nn.Linear(input_dim, d_model).to(device)
		self.positional_encoding = self._generate_positional_encoding(self.d_model).to(device)
		self.transformer = nn.Transformer(
			d_model=d_model,
			nhead=nhead,
			num_encoder_layers=num_layers,
			num_decoder_layers=num_layers,
		).to(device)
		self.fc = nn.Linear(d_model, n_classes).to(device)
		# Define lists to save losses
		self.loss_training_hist, self.loss_validation_hist = [], []
		# Define lists to save metrics
		self.accuracy_training, self.accuracy_val = [], []
		# Define criterion
#		self.criterion = utils.MulticlassCrossEntropyLoss()
		self.criterion = nn.CrossEntropyLoss()
	def _generate_positional_encoding(self, d_model, max_len=5000):
		position = torch.arange(0, max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
		pe = torch.zeros(max_len, 1, d_model)
		pe[:, 0, 0::2] = torch.sin(position * div_term)
		pe[:, 0, 1::2] = torch.cos(position * div_term)
		
		return pe
	def one_hot_encode(self, label_ground):
		num_classes = len(self.labeling_order)
		label_index = self.labeling_order.index(label_ground)
		one_hot = torch.zeros(1, num_classes)
		one_hot[0, label_index] = 1
		return one_hot
	def forward(self, x):
		batch_size = x.size(0)
		x = self.encoder(x)
		x = x + self.positional_encoding[:x.size(0), :]
		x = self.transformer(x,x)
		x = self.fc(x)
		x = x.mean(dim = 1, keepdim = False).squeeze(0)  # Average pooling over the time dimension
		return F.softmax(x, dim=1)
	def compute_metrics(self, X, Y, bs, nb):
		y_pred_list, y_true_list = [], []
		real_labeling = torch.cat( [ self.one_hot_encode(label_ground = Y[i] ) for i in range(Y.shape[0]) ], dim = 0 ).to(self.device)
		for i in tqdm(range(nb)):
			y_pred = torch.argmax( self( X[i*bs:(i+1)*bs,1,:] ), dim = 1 ).view(-1,1)
			y_true = torch.argmax( real_labeling[i*bs:(i+1)*bs], dim = 1 ).view(-1,1)
			y_pred_list.append( y_pred )
			y_true_list.append( y_true )
		y_pred_list = torch.cat( y_pred_list, dim = 0 )
		y_true_list = torch.cat( y_true_list, dim = 0 )
		# Metrics
		accuracy = (y_pred_list == y_true_list).sum() / y_pred_list.shape[0]
		return accuracy
	def train_step(self, X, Y, optimizer):
		real_labeling = torch.cat( [ self.one_hot_encode(label_ground = Y[i] ) for i in range(Y.shape[0]) ], dim = 0 ).to(self.device)
		output = self(X)
		loss = self.criterion(output, real_labeling)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad(set_to_none = True)
		return loss

# Define a recurrent model to classify
class LSTMClassifier(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes, weights_tensor, DTYPE, device, labeling_order = (6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95)):
		super(LSTMClassifier, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.labeling_order = labeling_order
		self.weights_tensor = weights_tensor
		self.DTYPE = DTYPE
		self.device = device
		# Define lists to save losses
		self.loss_training_hist, self.loss_validation_hist = [], []
		# Define lists to save metrics
		self.accuracy_training, self.accuracy_val = [], []
		# Define criterion
#		self.criterion = utils.MulticlassCrossEntropyLoss()
		self.criterion = nn.CrossEntropyLoss()
		
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, num_classes)
	def one_hot_encode(self, label_ground):
		num_classes = len(self.labeling_order)
		label_index = self.labeling_order.index(label_ground)
		one_hot = torch.zeros(1, num_classes)
		one_hot[0, label_index] = 1
		return one_hot
	def forward(self, x):
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
		out, _ = self.lstm(x, (h0, c0))
		out = self.fc(out[:, -1, :])
		return F.softmax(out, dim=1)
	def compute_metrics(self, X, Y, bs, nb):
		y_pred_list, y_true_list = [], []
		real_labeling = torch.cat( [ self.one_hot_encode(label_ground = Y[i] ) for i in range(Y.shape[0]) ], dim = 0 ).to(self.device)
		for i in tqdm(range(nb)):
			y_pred = torch.argmax( self( X[i*bs:(i+1)*bs,1,:].to(self.device) ), dim = 1 ).view(-1,1)
			y_true = torch.argmax( real_labeling[i*bs:(i+1)*bs], dim = 1 ).view(-1,1)
			y_pred_list.append( y_pred )
			y_true_list.append( y_true )
		y_pred_list = torch.cat( y_pred_list, dim = 0 )
		y_true_list = torch.cat( y_true_list, dim = 0 )
		# Metrics
		accuracy = (y_pred_list == y_true_list).sum() / y_pred_list.shape[0]
		return accuracy
	def train_step(self, X, Y, optimizer):
		real_labeling = torch.cat( [ self.one_hot_encode(label_ground = Y[i] ) for i in range(Y.shape[0]) ], dim = 0 ).to(self.device)
		output = self( X )
		loss = self.criterion(output, real_labeling)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad(set_to_none = True)
		return loss
















































