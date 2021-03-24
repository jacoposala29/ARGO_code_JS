# _____________________________________________________________________________
# This script creates a mask for Hur and NoHur Argo profiles. Then it creates 
# two separate .csv files: one for Hurricanes Argo profiles and the other for 
# Non-Hurricanes Argo profiles. 
# _____________________________________________________________________________

import h5py # Binary data format library
import numpy as np # Library to compute most math operations
import pandas as pd # Library to handle dataframes - super useful
import pickle as pkl# Library useful to write output files
from tools import create_ArgoProfileID  # Create an ID code for a specific Argo profile 

# Import specific functions written by Donata in file TC_and_TS_define_param.py
from TC_and_TS_define_param import (
        window_size, min_num_obs, # Returns 8, 20
        var2use, # Temperature, Salinity
        folder2use,
        )

df = pkl.load(open(str(folder2use) + '/Outputs/ArgoProfileDF_NoHur.pkl', 'rb')) # Unpack file using pickle and load non-hurricanes profiles
NoHurs = set(df['ProfileID'].values) # Set a list of profiles withous hurricanes found from the list 'df'  

path = str(folder2use) + '/Outputs/gridArgoProfFiltered_'+window_size+'_'+min_num_obs+'_'+var2use+'.mat' # Directory 

f = h5py.File(path, 'r') # File opened using HDF5
float_ids =         np.array(f['profFloatIDAggrSel']).flatten().astype(int) #  # ID of Argo float
cycle_nums =        np.array(f['profCycleNumberAggrSel']).flatten() # # Number of Argo cycle
profile_ids =       list(
    create_ArgoProfileID(f, c) #  # Create and ID for each profile and each Argo float
    for f, c in zip(float_ids, cycle_nums)) 
mask = np.array([pid in NoHurs for pid in profile_ids], dtype=int) # Total elements: 269779. '1' elements (Hurs): 241221. '0' elements (NoHurs): 28558. 

np.savetxt(str(folder2use) + '/Outputs/NoHurMask.csv', mask) # Save NoHur mask file 
np.savetxt(str(folder2use) + '/Outputs/HurricaneProfileMask.csv', 1-mask) # Save Hur mask file

