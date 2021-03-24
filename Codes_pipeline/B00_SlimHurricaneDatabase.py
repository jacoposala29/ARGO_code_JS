# _____________________________________________________________________________
# This script reads metadata (lat, lon, time, profile ID) from Argo .mat files 
# and stores them in a pandas dataframe, which is much smaller and manageable 
# than the original .mat files
# _____________________________________________________________________________

# Import libraries needed for the program to run
import h5py          # Library to read some files (for example .mat)
import numpy as np   # Library to compute most math operations
import pandas as pd  # Library to handle dataframes - super useful
import pickle as pkl # Library useful to write output files
import scipy.io

# Import additional specific functions written by Donata in file tools.py
from tools import (
        conform_lons,         # Convert longitude expressed from [20, 380] - as Argo does - to longitudes expressed from [-180, 180].
        create_ArgoProfileID, # Create an ID code for a specific Argo profile
        matlab_to_datetime,   # Convert Matlab days since year zero to Python datetime object
        )

# Import specific variables written by Donata in file TC_and_TS_define_param.py
from TC_and_TS_define_param import (
        year_pairs, # Returns ((2007, 2008), (2009, 2010))
        folder2use, case2use
        )

# Directory where data are stored
data_dir  = '../Inputs/'

# Initialize list where outputs will be saved
df_lst = []
# Loop across year pairs
for (sy, ey) in year_pairs:
    # Specify file with Argo data for the current year pair
    path = data_dir + f'Argo_data_aggr_{sy}_{ey}.mat'
    # Open Argo data file
    if case2use == 'noML':
        f = h5py.File(path, 'r') 
    else:
        f = scipy.io.loadmat(path)
    # Read the following data from the Argo file
    # Longitude
    lons =              conform_lons(np.array(f['profLongAggr'])).flatten()
    # Latitude
    lats =              np.array(f['profLatAggr']).flatten()
    # Dates
    dates =             np.array(f['profJulDayAggr']).flatten()
    # Convert Matlab-formatted dates into a Python datetime object
    datetimes =         np.array([matlab_to_datetime(x) for x in dates])
    # ID of Argo float
    float_ids =         np.array(f['profFloatIDAggr']).flatten().astype(int)
    # Number of Argo cycle
    cycle_nums =        np.array(f['profCycleNumberAggr']).flatten()
    # Create and ID for each profile and each Argo float
    profile_ids =       np.array(list(create_ArgoProfileID(f, c)
                                      for f, c in zip(float_ids, cycle_nums)))
    # Create pandas dataframe with these data saved in it, and append it to the empty list created before
    df_lst.append(pd.DataFrame({
        'ProfileID':        profile_ids,
        'Longitude':        lons,
        'Latitude':         lats,
        'ArgoDate':         dates,
        'Timestamp':        datetimes,
        'FloatID':          float_ids,
        'CycleNum':         cycle_nums,
        }))

# Concatenate all the pandas dataframes in the list into a single dataframe
df = pd.concat(df_lst)
# Reset the index of the concatenated pandas dataframe (it has no physical importance for us)
df = df.reset_index(drop=True)
# Save the single pandas dataframe in an output file
pkl.dump(df, open(str(folder2use) + '/Outputs/ArgoProfileDF.pkl', 'wb'))
