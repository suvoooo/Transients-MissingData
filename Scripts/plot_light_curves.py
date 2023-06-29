import os, gc, itertools
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.manifold import TSNE

from astropy.coordinates import SkyCoord
from astropy import units as u

############################################

gc.collect()
#read csv
training = pd.read_csv('../Train-Data/training_set.csv')
meta_training = pd.read_csv('../Train-Data/training_set_metadata.csv')
print ('check shapes: training & meta-training', training.shape, meta_training.shape)

### numbers to pass-band
my_map = {0:'u', 1:'g', 2:'r', 3:'i', 4:'z', 5:'y'}
training['passband_n'] = training['passband'].map(my_map)

######################################

## merge the flux data set based on object ids
merged = training.merge(meta_training, on = "object_id")
merged = merged.sort_values(by='object_id', ascending=True)
print ('check the shape of the merged dataframe: ', merged.shape)
# merged.head(5)

### group objects based on pass band and object id 

groups_obj_pb = training.groupby(['object_id', 'passband'])



times = groups_obj_pb.apply(lambda block: block['mjd'].values).reset_index().rename(columns={0: 'seq'})
print (times.head(3))

print ('\n')

object_list_grpby=times.groupby('object_id').apply(lambda x: x['object_id'].unique()[0]).tolist()
print (len(object_list_grpby), object_list_grpby[5:10])

############################
### plotting routine
############################

colors_6_list = ['violet', 'green', 'red', 'indigo', 'darkslategray', 'yellow']
passband_list = ['u', 'g', 'r', 'i', 'z', 'y']


def plot_one_object(df, grpd_obj, cat, obj_id, split=False, frequency=0):
    
    if split:
        fig, axes = plt.subplots(2, 3, sharex=True, sharey=False, figsize=(12, 8))
    else:
        fig, ax=plt.subplots(figsize=(10, 6))
    for band in range(len(colors_6_list)):
        if split:
            ax = axes[band // 3, band % 3]
        sample = df[(df['object_id'] == grpd_obj) & (df['passband']==band)]
        if frequency:
            phase=(sample['mjd'] * frequency ) % 1
        else:
            phase=sample['mjd']
        ax.errorbar(x=phase,y=sample['flux'], yerr=sample['flux_err'], 
                    c = colors_6_list[band],fmt='o',alpha=0.7, label='%s'%(passband_list[band]))
        ax.set_xlabel('mjd', fontsize=13)
        ax.set_ylabel('Flux', fontsize=13)
        ax.set_title('Category: %d, Obj_Id: %d'%(cat, obj_id))
        ax.legend(fontsize=10, loc='best')
    fig.savefig('./LC-6bands-CatId%d-ObjId-%d.png'%(cat, obj_id), dpi=200)
	
#############################################
#
#############################################
##

print ('select category id: eg: 6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95')

category_id = int(input('select from the list above: ' ))

## you can change the index[2] to select another object under the same category
iobj=meta_training[(meta_training['target']==category_id) & (meta_training['ddf']==1)]['object_id'].index[2]

# ddf: A Boolean flag to identify the object as coming from the DDF survey area 
# the DDF fields have significantly smaller uncertainties
print ('check what object-id: ', iobj)
## call the plotting routine created above
plot_one_object(training, grpd_obj=object_list_grpby[iobj], cat=category_id, obj_id=iobj, split=False)
