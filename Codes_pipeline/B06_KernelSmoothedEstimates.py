from itertools import product
import numpy as np
import os
import pandas as pd
import pickle as pkl
import sys
from TC_and_TS_define_param import depth_layers, folder2use

sys.path.append('./implementations/')
from Regressors import KernelSmoother
from implementation_tools import (
    grid,
    variable_diff,
)

# Set global variables
output_dir = str(folder2use) + '/Outputs/'
DEPTH_IDX = depth_layers
STAGE = ('adj', 'raw', 'mf')
h = 0.2 #bandwidth

# Read in dataframe from B05
df_all = pkl.load(open(str(folder2use) + '/Outputs/HurricaneAdjRawVariableDF.pkl', 'rb'))

if DEPTH_IDX > 1:
    #_____________________________________________________________________________
    # Calculate differences 50 dbar - 10 dbar
    diff_raw_before = np.zeros(df_all.shape[0])
    for i in np.arange(0,df_all.shape[0]):
        diff_raw_before[i] = df_all.raw_before_variable.iloc[i][4] - df_all.raw_before_variable.iloc[i][0]
    
    DFS = [
        (df_all, 'all_combined'),
        (df_all[df_all['wind'] >= 64], 'all_hurricanes'),
        (df_all[df_all['wind'] <  64], 'all_tstd'),
        (df_all[(diff_raw_before < 0)], 'decreasing01_combined'),
        (df_all[(diff_raw_before > 0)], 'increasing01_combined'),
        (df_all[(diff_raw_before < 0) & (df_all['wind'] >= 64)], 'decreasing01_hurricanes'),
        (df_all[(diff_raw_before > 0) & (df_all['wind'] >= 64)], 'increasing01_hurricanes'),
        (df_all[(diff_raw_before < 0) & (df_all['wind'] < 64)], 'decreasing01_tstd'),
        (df_all[(diff_raw_before > 0) & (df_all['wind'] < 64)], 'increasing01_tstd'),
        ]
else:
    DFS = [
        (df_all, 'all_combined'),
        (df_all[df_all['wind'] >= 64], 'all_hurricanes'),
        (df_all[df_all['wind'] <  64], 'all_tstd'),
        ]

# Create grid for output
bounds_x1 = (-8, +8) # will be x-axis in 2d plots (space, in degrees)
bounds_x2 = (-2, +20)  # will be y-axis in 2d plots (time, in days)
shape = (100, 400) 
# Grid over which the 2d plot will be made, and estimates calculated in this script
test_X = grid(bounds_x1, bounds_x2, *shape).copy()
n_test = test_X.shape[0]
# Save grid
pkl.dump(test_X, open(f'{output_dir}/test_X.pkl', 'wb'))


# Loop across stages ('adj', 'raw', 'mf') and categories (TSTD, hurricanes, combined)
for stage, (df, subset) in product(STAGE, DFS):
    fn_out = f'{output_dir}/KernelSmoothedMatx_{h}_{stage}_{subset}.pkl'
    #if os.path.exists(fn_out):
    #    continue
    estimates = np.zeros((n_test, DEPTH_IDX))
    # Loop across depth levels
    for depth_idx in range(DEPTH_IDX):
        print(stage, subset, depth_idx)
        ks = KernelSmoother(h=h)
        # Creates space-time training set from observations
        # standard_signed_angle is distance from TS track in degrees
        # hurricane_dtd is delta time in days
        train_X = np.array(df[['standard_signed_angle', 'hurricane_dtd']]).copy()
        if stage == 'mf':
            raw = variable_diff(df, 'raw', depth_idx) # raw after - raw before
            adj = variable_diff(df, 'adj', depth_idx) # adj after - adj before
            train_y = (raw - adj).copy()
        else:
            # Calculates post-pre TS temperature/salinity for the depth we're considering now
            train_y = variable_diff(df, stage, depth_idx) # (raw after - raw before) or (adj after - adj before)
        # Fit model on observations of post-pre TS differences of Temp/Salinity
        ks.fit(train_X, train_y)
        # Use the fitted model to predict values on a regular grid
        estimates[:, depth_idx] = ks.predict(test_X)
    # Writes output
    pkl.dump(estimates, open(fn_out, 'wb'))
    
    
    