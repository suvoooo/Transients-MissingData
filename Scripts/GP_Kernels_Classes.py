import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table, vstack

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic

### Build Our Own GP Class Including Data Processing, Fit and Predict


colors_6_dict = {'u':'violet', 'g':'green', 'r':'red', 'i':'indigo', 'z':'darkslategray', 'y':'yellow'}
marker_list = ['o', 'p', 's', 'd', '*', '^']

pb_wavelengths = {"u": 3685., "g": 4802., "r": 6231.,
                  "i": 7542., "z": 8690., "y": 9736.}
inverse_pb_wavelengths = {v: k for k, v in pb_wavelengths.items()}

unique_passband = np.array(['g', 'i', 'r', 'u', 'y', 'z'])
gp_wavelengths = np.vectorize(pb_wavelengths.get)(unique_passband)

unique_wavelengths = np.unique(gp_wavelengths)

print ('unique_wavelengths: ', unique_wavelengths, unique_passband.shape)

############################################
# GP class using Sklearn
############################################

class GP_for_transient():
    def __init__(self, input_df:pd.DataFrame, obj_id:int, ):
        self.input_df=input_df
        self.obj_id=obj_id
    
    def shift_time(self):
        obs_check = self.input_df[self.input_df['object_id'] == self.obj_id]
        obs_time = obs_check['mjd']
        obs_check_detected_time = obs_time[obs_check['detected']==1]
        is_obs_transient = (obs_time > obs_check_detected_time.iloc[0] - 50) & (obs_time < obs_check_detected_time.iloc[-1] + 50)
        obs_transient = obs_check[is_obs_transient] # apply the mask

        obs_transient['mjd'] -= min(obs_transient['mjd'])

        adf_check = adf = pd.DataFrame(data=[], columns=self.input_df.columns)
        adf_check = np.vstack((adf, obs_transient))

        obs_transient = pd.DataFrame(data=adf_check, columns=obs_transient.columns)
        return obs_transient
    
    def time_wavelength_data(self):
        transient_df = self.shift_time()
        obs_transient_wavelengths = transient_df['passband_n'].map(pb_wavelengths)

        obs_transient_times = transient_df.mjd.astype(float)

        x_data = np.vstack([obs_transient_times, obs_transient_wavelengths]).T
    
        obs_tr_fl = transient_df.flux.astype(float)
    
        obs_tr_flerr = transient_df.flux_err.astype(float)

        signal_to_noise = np.abs(obs_tr_fl) / np.sqrt(obs_tr_flerr ** 2 + (1e-2 * np.max(obs_tr_fl)) ** 2)
    
        scale = np.abs(obs_tr_fl[signal_to_noise.idxmax()])
    
        return x_data, obs_tr_fl, obs_tr_flerr, scale
    
    def matern32_kernel_obj(self):
        _, _, _, scale = self.time_wavelength_data()

        kernels = (0.5 * scale) ** 2 * Matern(length_scale=3, 
                                              length_scale_bounds=(0, np.log(6000**2)), nu=1.5)


        gp_obj = GaussianProcessRegressor(kernel=kernels)
    
        return gp_obj
    
    def rat_quadratic_obj(self):
        _, _, _, scale = self.time_wavelength_data()
        kernels = (0.5*scale)**2 * RationalQuadratic(length_scale=3.0, 
                                                     length_scale_bounds=(0, np.log(6000**2)), alpha=1.5)
        
        gp_obj1 = GaussianProcessRegressor(kernel=kernels)
        return gp_obj1
        
    def radial_basis_obj(self):
        _, _, _, scale = self.time_wavelength_data()
        kernels = (0.5*scale)**2 * RBF(length_scale=3.0, length_scale_bounds=(0, np.log(6000**2)),)
        
        gp_obj1 = GaussianProcessRegressor(kernel=kernels)
        return gp_obj1

    def gp_fit_predict(self, kernel='matern'):
        transient_df = self.shift_time()
        filled_gp_times = np.linspace(min(transient_df['mjd']), max(transient_df['mjd']), 100)
        num_gp = len(filled_gp_times)
        if kernel=='matern':
            print ('Using Mater32 Kernel')
            gp = self.matern32_kernel_obj()
        elif kernel=='quadratic':
            print ('using Quadratic Kernel')
            gp = self.rat_quadratic_obj()
        elif kernel=='RBF':
            print ('using RBF kernel')
            gp = self.radial_basis_obj()        
        else: print ('error')    
        x_data, fl_data, _, _ = self.time_wavelength_data()
        gp.fit(x_data, fl_data)

        obj_gps = []
        for wl in unique_wavelengths:
            gp_wavelengths = np.ones(num_gp) * wl
            pred_x_data = np.vstack([filled_gp_times, gp_wavelengths]).T
            y_mean_fl, y_std_fl = gp.predict(pred_x_data, return_std=True)
            obj_gp_pb_array = np.column_stack((filled_gp_times, y_mean_fl, y_std_fl))
            obj_gp_pb = Table([obj_gp_pb_array[:, 0], 
                               obj_gp_pb_array[:, 1], 
                               obj_gp_pb_array[:, 2], 
                               [wl] * num_gp,], names=["mjd", "flux", "flux_err", "filter"],)
            if len(obj_gps)==0:
                obj_gps = obj_gp_pb
            else:
                obj_gps = vstack((obj_gps, obj_gp_pb))
        
        obj_gps = obj_gps.to_pandas()
    
        return obj_gps
    
    def pred_gp_fl_final(self, kernel='matern'):

        if kernel=='matern':

            obj_gps_pred = self.gp_fit_predict(kernel='matern')
        elif kernel=='quadratic':
            obj_gps_pred = self.gp_fit_predict(kernel='quadratic')        
        elif kernel=='RBF':
            obj_gps_pred = self.gp_fit_predict(kernel='RBF')   

        obj_gps_pred['passband_n'] = obj_gps_pred['filter'].map(inverse_pb_wavelengths)

        my_map_r = {'u':0, 'g':1, 'r':2, 'i':3, 'z':4, 'y':5}

        obj_gps_pred['passband'] = obj_gps_pred['passband_n'].map(my_map_r)
        return obj_gps_pred
        
        
        
#################################################

##############################
# plotting routines
###############################
colors_6_list = ['violet', 'green', 'red', 'indigo', 'darkslategray', 'yellow']
def return_flux_w_err(df):
    '''
    for a particular category this function returns:
    mjds as x, flux as y, flux_err as y_err 
    '''
    x_all_bands = {}
    y_all_bands = {}
    y_err_all_bands = {}
    for band in range(len(colors_6_list)):
        sample = df[df['passband']==band]

        phase=sample['mjd']
        x_all_bands[colors_6_list[band]] = phase
        y_all_bands[colors_6_list[band]] = sample['flux']
        y_err_all_bands[colors_6_list[band]] = sample['flux_err']
    return x_all_bands, y_all_bands, y_err_all_bands

def plot_real_data_and_model(obj_data, obj_id, obj_model=None, number_col=2, show_legend=True):
    """Plots real data and model fluxes at the corresponding mjd"""
    passbands = np.unique(obj_data['passband_n'])
    fig = plt.figure(figsize=(9, 4))
    for pb in passbands:
        obj_data_pb = obj_data[obj_data['passband_n'] == pb] # obj LC in that passband
        if obj_model is not None:
            obj_model_pb = obj_model[obj_model['passband_n'] == pb]
            model_flux = obj_model_pb['flux']
            plt.plot(obj_model_pb['mjd'], model_flux, color=colors_6_dict[pb], 
                     alpha=.7, label='')
            try:
                model_flux_error = obj_model_pb['flux_err']
                plt.fill_between(x=obj_model_pb['mjd'], 
                                 y1=model_flux-(model_flux_error), 
                                 y2=model_flux+(model_flux_error), 
                                 color=colors_6_dict[pb], alpha=0.3, label=None)
            except:
                pass
        plt.errorbar(obj_data_pb['mjd'], obj_data_pb['flux'], obj_data_pb['flux_err'], 
                     fmt='o', color=colors_6_dict[pb], label=pb[-1])
    plt.xlabel('Time (days)')
    plt.ylabel('Flux units')
    if show_legend:
        plt.legend(ncol=number_col, handletextpad=.3, borderaxespad=.3, 
                   labelspacing=.2, borderpad=.3, columnspacing=.4)
    plt.title('ObjID_%d_AllBands'%(obj_id))    
    plt.show()
    
def plot_4_bands_w_GP(dates, dates_gp, fl, fl_gp, fl_err, fl_err_gp, id):
    '''
    dates, fl, fl_err are all dictionaries 
    keys: 6 bands based on the colors_6_list = ['violet', 'green', 'red', 'indigo', 'darkslategray', 'yellow']
    values: dates: mjds, fl: photon flux, fl_err: corresponding error
    id: integer; object id
    change the selected color list below to see other bands
    '''

    color_list_selected = ['red', 'green', 'indigo', 'yellow']
    plot_colors = ['crimson', 'seagreen', 'indigo', 'gold']
    alpha_list = [0.2, 0.2, 0.3, 0.5]
    fig, axs = plt.subplots(4, 1, figsize=(10, 7))
    for x, y, z, c in zip(axs, color_list_selected, alpha_list, plot_colors):
        x.plot(dates_gp[y], fl_gp[y], 
               color=c, ls='--', )
        x.errorbar(dates[y], fl[y], fl_err[y], fmt='s', 
                   color=c, label='Org.')
        x.fill_between(dates_gp[y], 
                       y1=fl_gp[y] - (fl_err_gp[y]), y2=fl_gp[y] + (fl_err_gp[y]), 
                       color=c, alpha=z, label='GP')
    axs[0].legend(fontsize=11)
    axs[3].set_xlabel('MJD', fontsize=12)
    fig.text(0.02, 0.5, 'Flux', va='center', rotation='vertical', fontsize=14)
    plt.tight_layout()    
    plt.suptitle('ObjID_%d_RGIY'%(id), fontsize=11)
    plt.tight_layout()
    plt.show()
    
    
    
#####################################            
