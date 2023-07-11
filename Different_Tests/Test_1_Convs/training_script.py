import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import torch.nn.functional as F
import math
import random
import h5py
# Import utils
sys.path.insert(1, "../../")
import utils, models

print("Is PyTorch using GPU?", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

print("Loading data...")
# Import data
hf = h5py.File("../../Data/GP_galactic_data.h5", "r")

X_training = torch.tensor( np.array(hf.get("X_training")), dtype = DTYPE )
Y_training = torch.tensor( np.array(hf.get("Y_training")), dtype = DTYPE )

X_test = torch.tensor( np.array(hf.get("X_val")), dtype = DTYPE )
Y_test = torch.tensor( np.array(hf.get("Y_val")), dtype = DTYPE )

X_val = torch.tensor( np.array(hf.get("X_test")), dtype = DTYPE )
Y_val = torch.tensor( np.array(hf.get("Y_test")), dtype = DTYPE )

hf.close()

print("Training shape: ", X_training.shape, Y_training.shape)
print("Validation shape: ", X_val.shape, Y_val.shape)
print("Test shape: ", X_test.shape, Y_test.shape)

# Define batch_size and compute n_batches
bs, bs_val = 5, 5
n_b, n_b_val = int( X_training.shape[0] / bs ), int( X_val.shape[0] / bs_val )

print( "batch_size: ", bs, ", n_batches: ", n_b )
labeling_order = torch.unique(Y_training, return_counts = False).type(torch.int).detach().cpu().numpy().tolist()
print( "labeling_order: ", labeling_order )

# Compute weights from frequency of classes in the training set
unique_training, counts_training = torch.unique( Y_training, return_counts = True )

inverse_freqs =  1 / counts_training
weights_tensor = ( inverse_freqs / torch.sum(inverse_freqs) ).to(device)

# Define the model hyperparameters
input_dim = 6  # Number of input features
n_classes = 14
num_layers = 3
smoothing_window_sizes = [11, 51, 101]
downsampling_coeffs = [5, 10, 50]
n_classes = len(labeling_order)

model = models.LSTMClassifier(input_size = input_dim, hidden_size = 5, num_layers = num_layers, num_classes = n_classes, weights_tensor = weights_tensor, DTYPE = DTYPE, device = device, labeling_order = labeling_order).to(device)

# Define learning rate and optimizers
lr_w = 1e-4
optimizer = torch.optim.Adam( model.parameters(), lr = lr_w )
print("Optimizer's state_dict of the global NN:")
for var_name in optimizer.state_dict():
	print(var_name, "\t", optimizer.state_dict()[var_name])

# Training bucle
epochs = 101
print("Starting optimization...")
pbar = tqdm(range(epochs))

for epoch in pbar:
	pbar_batch = tqdm(range(n_b))
	# Training
	loss_training_batch, loss_val_batch = [], []
	for b in pbar_batch:
		loss_training = model.train_step( X = X_training[b*bs:(b+1)*bs,1,:,:].to(device), Y = Y_training[b*bs:(b+1)*bs].to(device), optimizer = optimizer )
		loss_training_batch.append( loss_training )
		pbar_batch.set_postfix({ 'loss_training_batch' : float( torch.mean( torch.stack(loss_training_batch), dim = 0 ).detach().cpu().numpy() ) })
	model.loss_training_hist.append( float( torch.mean( torch.stack(loss_training_batch), dim = 0 ).detach().cpu().numpy() ) )
	## Save metrics
	accuracy_training = model.compute_metrics( X = X_training, Y = Y_training, bs = bs, nb = n_b )
	model.accuracy_training.append( float(accuracy_training.detach().cpu().numpy()) )
	
	# Validation
	## Save metrics
	accuracy_val = model.compute_metrics( X = X_val, Y = Y_val, bs = bs_val, nb = n_b_val )
	model.accuracy_val.append( float(accuracy_val.detach().cpu().numpy()) )
	## Compute loss on validation set
	real_label_val = torch.cat( [ model.one_hot_encode(label_ground = Y_val[i] ) for i in range(Y_val.shape[0]) ], dim = 0 ).to(device)
	loss_val = []
	for i in range( n_b_val ):
		loss_val.append( float( model.criterion( model( X_val[i*bs_val:(i+1)*bs_val,1,:,:].to(device) ), real_label_val[i*bs_val:(i+1)*bs_val,:].to(device) ).detach().cpu().numpy() ) )
	model.loss_validation_hist.append( np.array(loss_val).mean() )
	pbar.set_postfix({ 'loss_training_hist' : model.loss_training_hist[len(model.loss_training_hist)-1], 'loss_validation_hist' : model.loss_validation_hist[len(model.loss_validation_hist)-1] })
	if epoch % 2:
		utils.plot_results(model = model, path = "Images/")
		torch.save( model.state_dict(), "Models_Data/Model_Saved/model_epoch_" + str(epoch) + ".pt" )


































































