# IMPORT MODULES ========================================================================================
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import george
import numpy as np
import pandas as pd
from astropy.table import Table, vstack
import scipy.optimize as op
from functools import partial
from tqdm import tqdm
import h5py
import sys

# FUNCTIONS =============================================================================================
def remap_filters(df):  # maybe not in snmachine (raise issue/channel)
	"""Function to remap integer filters to the corresponding lsst filters and
	also to set filter name syntax to what snmachine already recognizes
	df: pandas.dataframe
	    Dataframe of lightcurve observations
	"""
	df.rename({'passband': 'filter'}, axis='columns', inplace=True)
	filter_replace = {0: 'lsstu', 1: 'lsstg', 2: 'lsstr', 3: 'lssti',
	                  4: 'lsstz', 5: 'lssty'}
	df['filter'].replace(to_replace=filter_replace, inplace=True)
	return df

def remap_filters_inverse(df):
	filter_replace = {"lsstu": 0, "lsstg": 1, "lsstr": 2, "lssti": 3,
	                  "lsstz": 4, "lssty": 5}
	df['filter'].replace(to_replace=filter_replace, inplace=True)
	return df

def fit_2d_gp(obj_data, return_kernel=False, **kwargs):
	"""Fit a 2D Gaussian process.
	If required, predict the GP at evenly spaced points along a light curve.
	Parameters
	----------
	obj_data : pandas.core.frame.DataFrame or astropy.table.Table
	    Time, flux and flux error of the data (specific filter of an object).
	return_kernel : Bool, default = False
	    Whether to return the used kernel.
	kwargs : dict
	    Additional keyword arguments that are ignored at the moment. We allow
	    additional keyword arguments so that the various functions that
	    call this one can be called with the same arguments.
	Returns
	-------
	kernel: george.gp.GP.kernel, optional
	    The kernel used to fit the GP.
	gp_predict : functools.partial of george.gp.GP
	    The GP instance that was used to fit the object.
	"""
	guess_length_scale = 20.0  # a parameter of the Matern32Kernel
	obj_times = obj_data.mjd.astype(float)
	obj_flux = obj_data.flux.astype(float)
	obj_flux_error = obj_data.flux_error.astype(float)
	obj_wavelengths = obj_data['filter'].map(pb_wavelengths)
	def neg_log_like(p):  # Objective function: negative log-likelihood
		gp.set_parameter_vector(p)
		loglike = gp.log_likelihood(obj_flux, quiet=True)
		return -loglike if np.isfinite(loglike) else 1e25
	def grad_neg_log_like(p):  # Gradient of the objective function.
		gp.set_parameter_vector(p)
		return -gp.grad_log_likelihood(obj_flux, quiet=True)
	# Use the highest signal-to-noise observation to estimate the scale. We
	# include an error floor so that in the case of very high
	# signal-to-noise observations we pick the maximum flux value.
	signal_to_noises = np.abs(obj_flux) / np.sqrt(
		obj_flux_error ** 2 + (1e-2 * np.max(obj_flux)) ** 2
	)
	scale = np.abs(obj_flux[signal_to_noises.idxmax()])
	kernel = (0.5 * scale) ** 2 * george.kernels.Matern32Kernel([
		guess_length_scale ** 2, 6000 ** 2], ndim=2)
	kernel.freeze_parameter("k2:metric:log_M_1_1")
	gp = george.GP(kernel)
	default_gp_param = gp.get_parameter_vector()
	x_data = np.vstack([obj_times, obj_wavelengths]).T
	gp.compute(x_data, obj_flux_error)
	bounds = [(0, np.log(1000 ** 2))]
	bounds = [(default_gp_param[0] - 10, default_gp_param[0] + 10)] + bounds
	results = op.minimize(neg_log_like, gp.get_parameter_vector(),
	                      jac=grad_neg_log_like, method="L-BFGS-B",
	                      bounds=bounds, tol=1e-6)
	if results.success:
		gp.set_parameter_vector(results.x)
	else:
		# Fit failed. Print out a warning, and use the initial guesses for fit
		# parameters.
		obj = obj_data['object_id'][0]
		print("GP fit failed for {}! Using guessed GP parameters.".format(obj))
		gp.set_parameter_vector(default_gp_param)
	gp_predict = partial(gp.predict, obj_flux)
	if return_kernel:
		return kernel, gp_predict
	else:
		return gp_predict


def predict_2d_gp(gp_predict, gp_times, gp_wavelengths):
	"""Outputs the predictions of a Gaussian Process.
	Parameters
	----------
	gp_predict : functools.partial of george.gp.GP
	    The GP instance that was used to fit the object.
	gp_times : numpy.ndarray
	    Times to evaluate the Gaussian Process at.
	gp_wavelengths : numpy.ndarray
	    Wavelengths to evaluate the Gaussian Process at.
	Returns
	-------
	obj_gps : pandas.core.frame.DataFrame, optional
	    Time, flux and flux error of the fitted Gaussian Process.
	"""
	unique_wavelengths = np.unique(gp_wavelengths)
	number_gp = len(gp_times)
	obj_gps = []
	for wavelength in unique_wavelengths:
		gp_wavelengths = np.ones(number_gp) * wavelength
		pred_x_data = np.vstack([gp_times, gp_wavelengths]).T
		pb_pred, pb_pred_var = gp_predict(pred_x_data, return_var=True)
		# stack the GP results in a array momentarily
		obj_gp_pb_array = np.column_stack((gp_times, pb_pred, np.sqrt(pb_pred_var)))
		obj_gp_pb = Table(
			[
				obj_gp_pb_array[:, 0],
				obj_gp_pb_array[:, 1],
				obj_gp_pb_array[:, 2],
				[wavelength] * number_gp,
			],
			names=["mjd", "flux", "flux_error", "filter"],
		)
		if len(obj_gps) == 0:  # initialize the table for 1st passband
			obj_gps = obj_gp_pb
		else:  # add more entries to the table
			obj_gps = vstack((obj_gps, obj_gp_pb))
	obj_gps = obj_gps.to_pandas()
	return obj_gps

# REST OF THE SCRIPT ====================================================================================

# Amount of time partitions to sample in order to do the Gaussian interpolation
number_gp = 5000

colours = {
	'lsstu': '#9a0eea', 
	'lsstg': '#75bbfd', 
	'lsstr': '#76ff7b',
	'lssti': '#fdde6c', 
	'lsstz': '#f97306', 
	'lssty': '#e50000'
}

# Central passbands wavelengths
pb_wavelengths = {"lsstu": 3685., "lsstg": 4802., "lsstr": 6231.,
                  "lssti": 7542., "lsstz": 8690., "lssty": 9736.}

# Load data
data = pd.read_csv("Data/training_set.csv")
metadata = pd.read_csv("Data/training_set_metadata.csv")

## FILTER IF NECESSARY TO GALACTIC OR EXTRAGALACTIC DATA
metadata_galactic = metadata.loc[metadata["hostgal_photoz"] == 0,:]
metadata_extragalactic = metadata.loc[metadata["hostgal_photoz"] != 0,:]

id_objs_galactic = metadata_galactic["object_id"].values.tolist()
id_objs_extragalactic = metadata_extragalactic["object_id"].values.tolist()

# Filter galactic objects
data = data[data["object_id"].isin(id_objs_extragalactic)]
metadata = metadata_extragalactic

# Rename filters
data = remap_filters(df = data)
data.rename({'flux_err': 'flux_error'}, axis='columns', inplace=True)  # snmachine and PLAsTiCC uses a different denomination

# Extract unique filters
filters = data['filter']
filters = list(np.unique(filters))

# Extract total minimum and maximum times so all the objects can be sampled in the same temporal domain
general_min, general_max = np.min(data["mjd"]), np.max(data["mjd"])

print("Total minimum time: ", general_min)
print("Total maximum time: ", general_max)

# Extract all unique object ids
unique_obj_ids = np.unique(data["object_id"]).tolist()
# Define passbands
unique_passbands = list( range(6) )

data_features, labels_list = [], []
for i in tqdm( range( len(unique_obj_ids) ) ):
#for i in tqdm( range( 50 ) ):
	# Specify an object id and filter the data
	obs_single = data[data["object_id"] == unique_obj_ids[i]]
	# Save the label
	obs_metadata = metadata[metadata["object_id"] == unique_obj_ids[i]]
	labels_list.append( int(obs_metadata["target"].values) )
	# Fit the interpolation model
	gp_predict = fit_2d_gp(obs_single)
	# Generate the sampling of times
	gp_times = np.linspace( general_min, general_max, number_gp )
	# Extract wavelenghts
	gp_wavelengths = np.vectorize(pb_wavelengths.get)(filters)
	# Predict over the sampling
	obj_gps = predict_2d_gp( gp_predict, gp_times, gp_wavelengths )
	inverse_pb_wavelengths = {v: k for k, v in pb_wavelengths.items()}
	obj_gps['filter'] = obj_gps['filter'].map(inverse_pb_wavelengths)
	
	# Get passbands as integer from filters
	obj_gps = remap_filters_inverse(df = obj_gps)
	
	mjd_time_list, flux_list, flux_error_list = [], [], []
	for passband in unique_passbands:
		obj_gps_passband = obj_gps.loc[obj_gps["filter"] == passband,:].values
		mjd_time = np.transpose( obj_gps_passband[:,0:1] )
		flux = np.transpose( obj_gps_passband[:,1:2] )
		flux_error = np.transpose( obj_gps_passband[:,2:3] )
		
		mjd_time_list.append( mjd_time )
		flux_list.append( flux )
		flux_error_list.append( flux_error )
	mjd_time_list = np.stack( mjd_time_list, axis = 2 )
	flux_list = np.stack( flux_list, axis = 2 )
	flux_error_list = np.stack( flux_error_list, axis = 2 )
	
	features_list = np.expand_dims( np.concatenate( [ mjd_time_list, flux_list, flux_error_list ], axis = 0 ), axis = 0)
	data_features.append( features_list )

data_features = np.concatenate( data_features, axis = 0 )
labels_list = np.array(labels_list).reshape( (-1,1) )

# Split datasets into training, validation and test sets
random.seed(666)
integer_list = list( range( data_features.shape[0] ) )
random.shuffle(integer_list)
## Define the size of each of the three new lists
size_training = int(len(integer_list) * 0.75)
size_validation = int(len(integer_list) * 0.05)
size_test = int(len(integer_list) * 0.2)

X_training, Y_training = data_features[:size_training], labels_list[:size_training]
X_val, Y_val = data_features[size_training:size_training+size_validation], labels_list[size_training:size_training+size_validation]
X_test, Y_test = data_features[size_training+size_validation:], labels_list[size_training+size_validation:]

print(X_training.shape, Y_training.shape)
print(X_val.shape, Y_val.shape)
print(X_test.shape, Y_test.shape)

# Export datasets
hf = h5py.File("Data/GP_extragalactic_data.h5", "w")

hf.create_dataset( "X_training", data = X_training )
hf.create_dataset( "Y_training", data = Y_training )

hf.create_dataset( "X_val", data = X_val )
hf.create_dataset( "Y_val", data = Y_val )

hf.create_dataset( "X_test", data = X_test )
hf.create_dataset( "Y_test", data = Y_test )

hf.close()









































































































