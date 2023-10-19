'''
helper to go through all the fits (focusing on metadata) file in a folder.
create a compact dataframe containing all the relevant info from the header. 
Add and extract more as you need. 

Elasticc2 data was released with fits file for different classes in different folders. 
So a single dataframe created by this helper will contain distribution of parameters for that class. 
'''

#########

import os, re
import matplotlib.pyplot as plt

import sncosmo
import numpy as np
import pandas as pd

def read_fits_meta(phot_file, head_file):
    '''
    phot_file = photon file containing the time series; days and flux.
    head_file = header file containing metadata.
    output: dataframe for list of observations within 
    '''
    elasticc_fits = sncosmo.read_snana_fits(phot_file=phot_file, 
                                            head_file=head_file)
    ### this is a list of all the objects within that fits
    SNID = []
    RA = []
    DEC = []
    HOSTGAL_PHOTOZ = []
    HOSTGAL_PHOTOZ_ERR = []
    HOSTGAL_SPECZ = []
    HOSTGAL_SPECZ_ERR = []
    HOSTGAL_RA = []
    HOSTGAL_DEC = []
    HOSTGAL_SNSEP = []
    HOSTGAL_DDLR = []
    HOSTGAL_LOGMASS = []

    print ('check how many objects: ', len(elasticc_fits))

    for i in range(len(elasticc_fits)):
        snid = elasticc_fits[i].meta['SNID']
        SNID.append(snid)
        ra, dec = elasticc_fits[i].meta['RA'], elasticc_fits[i].meta['DEC']
        host_photoz, photoz_err =  (elasticc_fits[i].meta['HOSTGAL_PHOTOZ'], 
                                    elasticc_fits[i].meta['HOSTGAL_PHOTOZ_ERR'])
        RA.append(ra)
        DEC.append(dec)
        HOSTGAL_PHOTOZ.append(host_photoz)
        HOSTGAL_PHOTOZ_ERR.append(photoz_err)
        host_specz, specz_err = (elasticc_fits[i].meta['HOSTGAL_SPECZ'], 
                                 elasticc_fits[i].meta['HOSTGAL_SPECZ_ERR'])
        HOSTGAL_SPECZ.append(host_specz)
        HOSTGAL_SPECZ_ERR.append(specz_err)
        host_ra, host_dec = elasticc_fits[i].meta['HOSTGAL_RA'], elasticc_fits[i].meta['HOSTGAL_DEC']
        HOSTGAL_RA.append(host_ra)
        HOSTGAL_DEC.append(host_dec)
        snsep, ddlr = elasticc_fits[i].meta['HOSTGAL_SNSEP'], elasticc_fits[i].meta['HOSTGAL_DDLR']
        HOSTGAL_SNSEP.append(snsep)
        HOSTGAL_DDLR.append(ddlr)
        logmass = elasticc_fits[i].meta['HOSTGAL_LOGMASS']
        HOSTGAL_LOGMASS.append(logmass)
    data = {'SNID': SNID, 'RA': RA, 'DEC':DEC, 
            'HOSTGAL_PHOTOZ': HOSTGAL_PHOTOZ, 
            'HOSTGAL_PHOTOZ_ERR': HOSTGAL_PHOTOZ_ERR, 
            'HOSTGAL_SPECZ': HOSTGAL_PHOTOZ, 
            'HOSTGAL_SPECZ_ERR': HOSTGAL_PHOTOZ_ERR, 
            'HOSTGAL_RA': HOSTGAL_RA, 'HOSTGAL_DEC': HOSTGAL_DEC, 
            'HOSTGAL_SNSEP': HOSTGAL_SNSEP, 'HOSTGAL_DDLR': HOSTGAL_DDLR, 
            'HOSTGAL_LOGMASS': HOSTGAL_LOGMASS}
    df = pd.DataFrame(data)
    return df     



def read_all_matched_fits_files(directory_path):
    # Initialize an empty list to store dataframes from matched pairs of files
    dfs = []

    # List all FITS files in the directory
    fits_files = [file for file in os.listdir(directory_path) if file.endswith(".FITS")]

    # Create a regular expression pattern to match the common part of the filenames
    pattern = r'ELASTICC2_TRAIN_02_NONIaMODEL0-(\d{4})'
    # pattern matching for head and phot files is necessary for reading the data

    for fits_file in fits_files:
        # Check if the filename matches the pattern
        match = re.match(pattern, fits_file)
        if match:
            # Extract the number part from the filename
            number = match.group(1)

            # Construct the full paths to the PHOT and HEAD FITS files
            phot_file_path = os.path.join(directory_path, 
                                          f'ELASTICC2_TRAIN_02_NONIaMODEL0-{number}_PHOT.FITS')
            head_file_path = os.path.join(directory_path, 
                                          f'ELASTICC2_TRAIN_02_NONIaMODEL0-{number}_HEAD.FITS')

            # Read the FITS files and extract the data
            df = read_fits_meta(phot_file=phot_file_path, head_file=head_file_path)

            # Append the dataframe to the list
            dfs.append(df)

    # Concatenate all dataframes into a single big dataframe
    big_df = pd.concat(dfs, ignore_index=True)

    return big_df

# Directory containing FITS files
fits_directory = '/d8/CAC/sbhattacharyya/Downloads/Elasticc2/Data/ELASTICC2_TRAIN_02_SNII-Templates'
tr_class = fits_directory.split('_')[-1] # extract the transient class from the path name
csv_path = fits_directory + '/Elasticc2_all_Meta_%s.csv'%(tr_class)

# Call the function to read all matched pairs of FITS files in the directory
big_dataframe = read_all_matched_fits_files(fits_directory)
big_dataframe.to_csv(csv_path, index=False)