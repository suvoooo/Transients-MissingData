'''
stes: 
1. fit the lcs with analtical model gaussian rise + exp-decay
2. obtain the flux ratios 
3. using the flux ratios, obtain Black-Body temperature (assume it doesn't change as we 
collect data over time). 
'''

import gc

import numpy as np 
import pandas as pd 
pd.options.mode.chained_assignment = None

from astropy.table import Table, vstack
import astropy.constants as const
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
import matplotlib.pyplot as plt



#from scipy.optimize import curve_fit#
# instead of curve fit we will use min chi square
from scipy.optimize import minimize #
import scipy.optimize as opt
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.integrate import simpson


#### include lsst throughputs
# throughput data from lsst github (https://github.com/lsst/throughputs/tree/main/baseline)
lsst_throughputs = {}
## load all the throughput files 

lsst_bands = ["u", "g", "r", "i", "z", "y"]

for band in lsst_bands:
    f_name = f"./filter_{band}.dat"
    
    # contains lambda in nm and corresponding throughput
    wl_nm = []
    throughput = []
    
    with open(f_name, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            values = line.split()
            wl_nm.append(float(values[0]))
            throughput.append(float(values[1]))
            
    wl_nm_arr = np.array(wl_nm)
    throughput_arr = np.array(throughput)
    
    wl_cm_arr = wl_nm_arr * 1e-7
    lsst_throughputs[band] = (wl_cm_arr, throughput_arr)


### let's try to plot 

colors_6_dict = {'u':'violet', 'g':'green', 
                 'r':'red', 'i':'indigo', 
                 'z':'darkslategray', 'y':'yellow'}



plt.figure(figsize=(9, 6))    
plt.xlabel("Wavelength (nm)")
plt.ylabel("Throughput (0-1)")
plt.title("LSST Filter Transmission Curves")
for band in lsst_bands:
    wavelength_cm, transmission = lsst_throughputs[band]
    wavelength_nm = wavelength_cm * 1e7  # Convert cm back to nm for plotting

    plt.plot(wavelength_nm, transmission, 
             label=f"{band}-band", color=colors_6_dict[band])


# Plot all bands 

plt.legend()
# plt.show()
plt.savefig('./lsst-throughputs-sixbands.png', dpi=200)

#########################
# read plasticc data
#########################

gc.collect()
#read csv
training = pd.read_csv('../Train-Data/training_set.csv')

meta_training = pd.read_csv('../Train-Data/training_set_metadata.csv')

print ('check shapes: ', training.shape, meta_training.shape)
print ('\n')
print (training.head(5))

# meta_training.head(5)

#############################
# select tde only dataframe, class=15
#############################

meta_training_tde = meta_training.loc[meta_training['target']==15]
print (meta_training_tde.shape)

# meta_training_tde.head(5)

my_map = {0:'u', 1:'g', 2:'r', 3:'i', 4:'z', 5:'y'}
training['passband_n'] = training['passband'].map(my_map)
# training.head(3)

########################
# data processing (rescale to start at 0 MJD)
########################

def rescale_time(input_df, obj_id): # used here
    obs_check = input_df[input_df['object_id']==obj_id]
    obs_time_max = obs_check['mjd'].iloc[-1]
    obs_time_min = obs_check['mjd'].iloc[0]
    obs_check['scaled_mjd'] = obs_check['mjd'] - obs_time_min
    return obs_check, obs_time_max, obs_time_min


#########################
# create a dictionary of days, flux and flux error 
#########################

colors_6_dict = {'u':'violet', 'g':'green', 'r':'red', 'i':'indigo', 
                 'z':'darkslategray', 'y':'yellow'}
marker_list = ['o', 'p', 's', 'd', '*', '^']
unique_passband = np.unique(training['passband_n'])
colors_6_list = ['violet', 'green', 'red', 'indigo', 
                 'darkslategray', 'yellow']

def return_flux_w_err(df, num_bands, scaled=False, mag_scaled=False,):
    '''
    for a particular category this function returns:
    mjds as x, flux as y, flux_err as y_err 
    '''
    x_all_bands = {}
    y_all_bands = {}
    y_err_all_bands = {}
    tot_bands = len(colors_6_list)
    
    for band in range(num_bands):
        sample = df[df['passband']==band]
        if scaled:
            phase=sample['scaled_mjd']
        else:
            phase=sample['mjd']
        
        # Only consider positive flux values
        positive_flux_sample = sample[sample['flux'] > 0]
        
        if mag_scaled:
            y_all_bands[colors_6_list[band]] = -2.5 * np.log10(positive_flux_sample['flux'])
            x_all_bands[colors_6_list[band]] = positive_flux_sample[phase]
            y_err_all_bands[colors_6_list[band]] = positive_flux_sample['flux_err']
            
        else:    
            x_all_bands[colors_6_list[band]] = phase    
            y_all_bands[colors_6_list[band]] = sample['flux']
            y_err_all_bands[colors_6_list[band]] = sample['flux_err']
            
        
        
    return x_all_bands, y_all_bands, y_err_all_bands


#######################
# tdescore model
########################

def tdescore_model_w_offset(t, t_peak, sigma, tau, A, offsets):
    """
    Lightcurve model for multiple bands with shared parameters but unique offsets.
    
    Params:
    - t: mjd arr (time arr)
    - t_peak: peak time (shared across all bands)
    - sigma: Width of gaussian (shared again).
    - tau: Exp. decay constant (shared agin).
    - offsets: Array of offsets for each band.
    - A: starting amp (could be from any bands) 
    
    Returns:
    - A dictionary of model fluxes for each band.
    """
    base_curve_rise = np.exp(-((t - t_peak) ** 2) / (2 * sigma ** 2))
    base_curve_decay = np.exp(-(t - t_peak) / tau)  
    # exp decay only when t>t_p
    base_curve = np.where(t < t_peak, base_curve_rise, base_curve_decay)

    return {band: (A + offset) * base_curve for band, offset in offsets.items()}


# Chi-square function # from fleet github
#!!!!!!!!!!!!!!!!!!!!!!!!!!!
#! Not used anymore, we use multi-band fitter
#!!!!!!!!!!!!!!!!!!!!!!!!!!!

def calc_chi2(ydata, ymod, n_parameters, fl_err):
    '''
    Calc reduced chi squared (check for d.o.f) 
    of a data set and a model while fitting some number of parameters.
    '''
    fl_err[fl_err <= 0] = np.nan  # Handling zero or negative uncertainties
    chisq = np.nansum(((ydata - ymod) / fl_err) ** 2.0)
    nu = ydata.size - n_parameters - 1.0
    if nu > 0:
        return chisq / nu
    else:
        return chisq
    
    


def combined_chi2(params, data_dict):
    """
    total chi^2 for fitting several bands with shared params but individual offsets.
    
    Parameters:
    - params: Array containing [t_peak, sigma, tau, A, offset1, offset2, ..., offsetN]
    - data_dict: Dictionary containing {band: (t, flux, flux_err)} for each band
    
    Returns:
    - Total chi-square value across all bands.
    """
    t_peak, sigma, tau, A = params[:4]
    offsets = params[4:] # this is now only 2
    
    chi2_total = 0
    offset_dict = {band: offset for band, offset in zip(data_dict.keys(), offsets)}
    for band, (t, flux, flux_err) in data_dict.items(): # loop over the dict and calc chi^2 sep
        model_flux = tdescore_model_w_offset(t, t_peak, sigma, tau, A, offset_dict)[band]
        chi2 = np.nansum(((flux - model_flux) / flux_err) ** 2)
        chi2_total += chi2
    
    return chi2_total


###########################
# include the black body function
###########################

# Constants
h = const.h.cgs.value  # Planck's constant in erg s
c = const.c.cgs.value  # Speed of light in cm/s
k_B = const.k_B.cgs.value  # Boltzmann constant in erg/K

print ('planck constant: ', h)
print ('\n')
print ('sp of light: ', c)
print ('\n')
print ('Boltzman Constant: ', k_B)

def planck_lambda(wl_cm, T):
    """Planck function in wv-length  (erg/s/cm^2/cm/sr)"""
    return (2 * h * c**2 / wl_cm**5) / (np.exp(h * c / (wl_cm * k_B * T)) - 1)

def compute_blackbody_flux(T, band):
    """Integrate Planck funct over LSST filter throughputs"""
    wl_cm, throughput = lsst_throughputs[band]  
    # Get throughput for this band
    B_lambda = planck_lambda(wl_cm, T)
    
    
    # integrating (Planck function * filter throughput) over wavelength
    flux = simpson(B_lambda * throughput, wl_cm)
    
    return flux


def compute_theoretical_flux_ratios(T):
    """Compute theoretical blackbody flux ratios at temperature T"""
    F_g = compute_blackbody_flux(T, 'g')
    F_r = compute_blackbody_flux(T, 'r')
    F_i = compute_blackbody_flux(T, 'i')
    F_z = compute_blackbody_flux(T, 'z')
    F_y = compute_blackbody_flux(T, 'y')
    F_u = compute_blackbody_flux(T, 'u')

    return {
        'g/r': F_g / F_r,
        'r/i': F_r / F_i,
        'i/z': F_i / F_z,
        'z/y': F_z / F_y,
        'u/g': F_u / F_g,
    } # choosing combinations should depend on the available data


def temperature_fit_loss(T):
    """Loss function: minimize diff. between obs. and theoretical flux ratios"""
    theoretical_ratios = compute_theoretical_flux_ratios(T)
    loss = sum((theoretical_ratios[key] - flux_ratios[key])**2 for key in flux_ratios)
    return loss

# Find best-fit blackbody temperature
T_initial = 12000  # Initial guess in Kelvin

#############################
# get the tde obj_id from meta and loop over training data
#############################

fit_results = [] # store results in a list and then in a dataframe

for obj_id in meta_training_tde['object_id']:
    obs_tr_tde_scaled, last_obs, first_obs = rescale_time(training, obj_id)
    obs_tr_tde_scaled = obs_tr_tde_scaled.loc[obs_tr_tde_scaled['flux']>0]
    (x_dates_dict_scaled, y_dict, 
     y_err_dict) = return_flux_w_err(obs_tr_tde_scaled, 6, 
                                     scaled=True, mag_scaled=False)
    
    (g_dates, g_fl, g_fl_err) = (x_dates_dict_scaled['green'], y_dict['green'], y_err_dict['green'])
    
    (r_dates, r_fl, r_fl_err) = (x_dates_dict_scaled['red'], y_dict['red'], y_err_dict['red'])
    
    (i_dates, i_fl, i_fl_err) = (x_dates_dict_scaled['indigo'], y_dict['indigo'], y_err_dict['indigo'])
    
    (u_dates, u_fl, u_fl_err) = (x_dates_dict_scaled['violet'], y_dict['violet'], y_err_dict['violet']) 
    
    (z_dates, z_fl, z_fl_err) = (x_dates_dict_scaled['darkslategray'], y_dict['darkslategray'], 
                                 y_err_dict['darkslategray'])
    
    (y_dates, y_fl, y_fl_err) = (x_dates_dict_scaled['yellow'], y_dict['yellow'], y_err_dict['yellow'])
    
    g_dates_arr = g_dates.to_numpy()
    g_fl_arr = g_fl.to_numpy()
    g_fl_err_arr = g_fl_err.to_numpy()

    r_dates_arr = r_dates.to_numpy()
    r_fl_arr = r_fl.to_numpy()
    r_fl_err_arr = r_fl_err.to_numpy()

    i_dates_arr = i_dates.to_numpy()
    i_fl_arr = i_fl.to_numpy()
    i_fl_err_arr = i_fl_err.to_numpy()


    u_dates_arr = u_dates.to_numpy()
    u_fl_arr = u_fl.to_numpy()
    u_fl_err_arr = u_fl_err.to_numpy()


    z_dates_arr = z_dates.to_numpy()
    z_fl_arr = z_fl.to_numpy()
    z_fl_err_arr = z_fl_err.to_numpy()

    y_dates_arr = y_dates.to_numpy()
    y_fl_arr = y_fl.to_numpy()
    y_fl_err_arr = y_fl_err.to_numpy()

    ### prepare the data for combined fit

    data_dict = {'green': (g_dates_arr, g_fl_arr, g_fl_err_arr), 
                 'red': (r_dates_arr, r_fl_arr, r_fl_err_arr), 
                 'indigo': (i_dates_arr, i_fl_arr, i_fl_err_arr), 
                 'violet': (u_dates_arr, u_fl_arr, u_fl_err_arr), 
                 'darkslategray': (z_dates_arr, z_fl_arr, z_fl_err_arr), 
                 'yellow': (y_dates_arr, y_fl_arr, y_fl_err_arr)}

    # Init. guess for the parameters: [t_peak, sigma, tau, offset_green, offset_red]
    t_peak_initial = g_dates_arr[np.argmax(g_fl_arr)]
    initial_guess = [t_peak_initial, 2, 5, np.max(g_fl_arr), 1, 1, 1, 1, 1, 1]

    # Set bounds for the parameters (modify as needed)
    bounds = [(t_peak_initial-10, t_peak_initial+10),  # t_peak 
              (0.001, 300),  # sigma 
              (0.001, 200),  # tau 
              (np.max(g_fl_arr)-200, np.max(g_fl_arr)+200), 
              (-100, 100),   #offset_green 
              (-100, 100),   #offset_red 
              (-100, 100),   #offset indigo 
              (-100, 100),   #offset u 
              (-100, 100),  #offset z 
              (-100, 100),  #offset y
              ]

    # Minimize the chi-square for all bands
    result = minimize(combined_chi2, initial_guess, args=(data_dict,), bounds=bounds)

    # Extract the best-fit parameters
    # Extract the best-fit parameters
    (t_peak_fit, sigma_fit, tau_fit, A_fit, offset_green_fit, 
     offset_red_fit, offset_i_fit, offset_u_fit, offset_z_fit, offset_y_fit) = result.x

    print(f"Fitted parameters:")


    print(f"t_peak = {t_peak_fit}")
    print(f"sigma = {sigma_fit}")
    print(f"tau = {tau_fit}")
    print(f"offset (green) = {offset_green_fit}")
    print(f"offset (red) = {offset_red_fit}")
    print(f"offset (i) = {offset_i_fit}")
    print(f"offset (u) = {offset_u_fit}")
    print(f"offset (z) = {offset_z_fit}")
    print(f"offset (y) = {offset_y_fit}")
    print(f"A = {A_fit}")
    print ('chi squared: %2.3f'%(result.fun))



    sampled_dates_arr_g = np.linspace(np.min(g_dates_arr), np.max(g_dates_arr), num=200)
    sampled_dates_arr_r = np.linspace(np.min(r_dates_arr), np.max(r_dates_arr), num=200)
    sampled_dates_arr_i = np.linspace(np.min(i_dates_arr), np.max(i_dates_arr), num=200)
    sampled_dates_arr_u = np.linspace(np.min(u_dates_arr), np.max(u_dates_arr), num=200)
    sampled_dates_arr_z = np.linspace(np.min(z_dates_arr), np.max(z_dates_arr), num=200)
    sampled_dates_arr_y = np.linspace(np.min(y_dates_arr), np.max(y_dates_arr), num=200)



    g_fit_flux = tdescore_model_w_offset(sampled_dates_arr_g, t_peak_fit, sigma_fit, tau_fit, 
                                        A_fit, {'green': offset_green_fit})['green']

    r_fit_flux = tdescore_model_w_offset(sampled_dates_arr_r, t_peak_fit, 
                                        sigma_fit, tau_fit, A_fit, {'red': offset_red_fit})['red']

    i_fit_flux = tdescore_model_w_offset(sampled_dates_arr_i, t_peak_fit, 
                                        sigma_fit, tau_fit, A_fit, {'indigo': offset_i_fit})['indigo']

    u_fit_flux = tdescore_model_w_offset(sampled_dates_arr_u, t_peak_fit, 
                                        sigma_fit, tau_fit, A_fit, {'violet': offset_u_fit})['violet']

    z_fit_flux = tdescore_model_w_offset(sampled_dates_arr_z, t_peak_fit, 
                                        sigma_fit, tau_fit, A_fit, {'darkslategray': offset_z_fit})['darkslategray']

    y_fit_flux = tdescore_model_w_offset(sampled_dates_arr_y, t_peak_fit, 
                                        sigma_fit, tau_fit, A_fit, {'yellow': offset_y_fit})['yellow']
    
    # Compute flux ratios from offsets
    flux_ratios = {'g/r': (A_fit + offset_green_fit) / (A_fit + offset_red_fit), 
                   'r/i': (A_fit + offset_red_fit) / (A_fit + offset_i_fit), 
                   'i/z': (A_fit + offset_i_fit) / (A_fit + offset_z_fit), 
                   'z/y': (A_fit + offset_z_fit) / (A_fit + offset_y_fit), 
                   'u/g': (A_fit + offset_u_fit) / (A_fit + offset_green_fit),
                   }

    # print computed ratios (for check)
    for key, value in flux_ratios.items():
        print(f"{key} flux ratio: {value:.3f}")


        

    # get the redshift for the obj (didn't use)
    # expectation: we won't have z at early stage
    obj_meta = meta_training_tde.loc[meta_training_tde.object_id==obj_id]
    obj_specz = obj_meta['hostgal_specz'].to_numpy()

    T_fit = minimize(temperature_fit_loss, T_initial, bounds=[(3000, 50000)]).x[0]
    # fixed temp

    # Append results to the list
    fit_results.append({"object_id": obj_id, "t_peak": t_peak_fit, "sigma": sigma_fit, 
                        "tau": tau_fit, "A": A_fit, 
                        "offset_green": offset_green_fit, 
                        "offset_red": offset_red_fit, 
                        "offset_indigo": offset_i_fit, 
                        "offset_violet": offset_u_fit, 
                        "offset_z": offset_z_fit, 
                        "offset_y": offset_y_fit, 
                        "chi_squared": result.fun, 
                        "T_fit": T_fit, 
                        "redshift": obj_specz, 
                        **flux_ratios  # Unpack flux ratios directly into the dictionary 
                        })

# Convert results to a DataFrame
fit_results_df = pd.DataFrame(fit_results)

print (fit_results_df.head(5))

fit_results_df.to_csv('./TDE_TDEScore_BB_FlRatio.csv', index=False)



