import sys
import os
import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import spectrogram
from scipy.signal.windows import gaussian

print("Is PyTorch using GPU?", torch.cuda.is_available())
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

print("Loading data...")
# Import data
hf = h5py.File("Data/GP_entire_data.h5", "r")
print(hf.keys())

X_training = torch.tensor( np.array(hf.get("X_training")), dtype = DTYPE ).to(device)
Y_training = torch.tensor( np.array(hf.get("Y_training")), dtype = DTYPE ).to(device)
specz_training = torch.tensor( np.array(hf.get("specz_training")), dtype = DTYPE ).to(device)
photoz_training = torch.tensor( np.array(hf.get("photoz_training")), dtype = DTYPE ).to(device)

X_test = torch.tensor( np.array(hf.get("X_val")), dtype = DTYPE ).to(device)
Y_test = torch.tensor( np.array(hf.get("Y_val")), dtype = DTYPE ).to(device)
specz_val = torch.tensor( np.array(hf.get("specz_val")), dtype = DTYPE ).to(device)
photoz_val = torch.tensor( np.array(hf.get("photoz_val")), dtype = DTYPE ).to(device)

X_val = torch.tensor( np.array(hf.get("X_test")), dtype = DTYPE ).to(device)
Y_val = torch.tensor( np.array(hf.get("Y_test")), dtype = DTYPE ).to(device)
specz_test = torch.tensor( np.array(hf.get("specz_test")), dtype = DTYPE ).to(device)
photoz_test = torch.tensor( np.array(hf.get("photoz_test")), dtype = DTYPE ).to(device)

hf.close()

print("Training shape: ", X_training.shape, Y_training.shape)
print("Validation shape: ", X_val.shape, Y_val.shape)
print("Test shape: ", X_test.shape, Y_test.shape)

X_training_FFT = X_training[:,1,:,:].permute((0,2,1))
X_training_FFT = torch.abs( torch.fft.fft(X_training_FFT, dim = -1) ).permute((0,2,1)).unsqueeze(1).detach().cpu().numpy()

X_val_FFT = X_val[:,1,:,:].permute((0,2,1))
X_val_FFT = torch.abs( torch.fft.fft(X_val_FFT, dim = -1) ).permute((0,2,1)).unsqueeze(1).detach().cpu().numpy()

X_test_FFT = X_test[:,1,:,:].permute((0,2,1))
X_test_FFT = torch.abs( torch.fft.fft(X_test_FFT, dim = -1) ).permute((0,2,1)).unsqueeze(1).detach().cpu().numpy()

# Export datasets
hf = h5py.File("Data/GP_entire_data_FFT_only.h5", "w")

hf.create_dataset( "X_training_FFT", data = X_training_FFT )
hf.create_dataset( "X_val_FFT", data = X_val_FFT )
hf.create_dataset( "X_test_FFT", data = X_test_FFT )

hf.close()



























































































