import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import utils, models
import torch.nn.functional as F
import math
import random
import gpytorch
import h5py

print("Is PyTorch using GPU?", torch.cuda.is_available())
device = torch.device("cpu")

training_set = pd.read_csv("Data/training_set.csv")
training_set_metadata = pd.read_csv("Data/training_set_metadata.csv")

print(training_set.shape, training_set_metadata.shape)

# First, extract minimum and maximum times from all the entire data

training_set = pd.read_csv("Data/training_set.csv")
training_set_metadata = pd.read_csv("Data/training_set_metadata.csv")

print(training_set.shape, training_set_metadata.shape)

obj_ids = training_set_metadata["object_id"].values.tolist()

min_times, max_times = [], []

for i in tqdm(range( len(obj_ids) ) ):
	data_obj, label = utils.load_passbands_list(path = "Data/Data_as_h5/obj_passbands.h5", objid = obj_ids[i], path_enter = "Data/")
	min_times.append( float( torch.min( torch.cat( [ obj[:,0:1] for obj in data_obj ], dim = 0 ).flatten() ) ) )
	max_times.append( float( torch.max( torch.cat( [ obj[:,0:1] for obj in data_obj ], dim = 0 ).flatten() ) ) )

general_min, general_max = np.min(min_times), np.max(max_times)

print("Total minimum time: ", general_min)
print("Total maximum time: ", general_max)

new_time = torch.linspace( general_min, general_max, 5000 )

# ============================================================================================================================
all_tensors, all_labels, all_obj_ids = [], [], []
for i in tqdm( range( len(obj_ids) ) ):
	# Specify an object
	data_obj, label = utils.load_passbands_list(path = "Data/Data_as_h5/obj_passbands.h5", objid = obj_ids[i], path_enter = "Data/")
	all_labels.append( label )
	all_obj_ids.append( obj_ids[i] )
	# Now, go through all the passbands and adjust the models
	pred_passbands = []
	for j in range(0, 6):
		# Initialize likelihood and model
		likelihood = gpytorch.likelihoods.GaussianLikelihood()
		model = models.ExactGPModel(train_x = data_obj[j][:,0:1].flatten(), train_y = data_obj[j][:,1:2].flatten(), likelihood = likelihood)
		observed_pred = utils.train_GP_and_eval(model = model, data_obj = data_obj, passband = j, new_time = new_time, likelihood = likelihood, num_iterations = 100)
		pred_passbands.append( observed_pred )
	pred_passbands = torch.stack( pred_passbands, dim = 1 ).unsqueeze(0)
	all_tensors.append( pred_passbands )
all_tensors = torch.cat(all_tensors, dim = 0)
all_labels = torch.tensor(all_labels).view(-1,1)
all_obj_ids = torch.tensor(all_obj_ids).view(-1,1)

hf = h5py.File("Data/Data_as_h5/entire_interpolated_objects.h5", "w")

hf.create_dataset( "entire_objects_tensor", data = all_tensors.detach().cpu().numpy() )
hf.create_dataset( "all_labels", data = all_labels.detach().cpu().numpy() )
hf.create_dataset( "all_obj_ids", data = all_obj_ids.detach().cpu().numpy() )

hf.close()




























































































