#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:11:18 2021

@author: jacoposala
"""
from itertools import product
import argparse
import numpy as np
import os
import pandas as pd
import pickle as pkl
import sys
sys.path.append('./implementations/')

from Regressors import ThinPlateSpline
from implementation_tools import (
    grid,
    variable_diff,
)
from TC_and_TS_define_param import grid_lower, grid_upper, grid_stride, folder2use

parser = argparse.ArgumentParser(
        description='Plot thin plate spline estimates (three panel)')
parser.add_argument('--integrated', dest='mode', action='store_const',
                    const='integrated', default='gridded',
                    help='process assuming integrated heat content (default: gridded)')
args = parser.parse_args()

data_dir = folder2use + '/Outputs'
results_dir = folder2use + '/Outputs'
window_size_gp = 5
stage = 'adj'

# Set dimensions of the grid
grid_start = int(grid_lower)
grid_end = int(grid_upper)
grid_stride = int(grid_stride)
DEPTH_IDX = int((grid_end - grid_start)/ grid_stride+1)

# Define various cases
if DEPTH_IDX > 1:
    sub_list = (
        'all_combined',
        'all_hurricanes',
        'all_tstd',
        'increasing_combined',
        'decreasing_combined',
        'increasing_hurricanes',
        'decreasing_hurricanes',
        'increasing_tstd',
        'decreasing_tstd',
    )
else: # single level
    sub_list = (
    'all_combined',
    'all_hurricanes',
    'all_tstd',
)

# Loop through all cases
for sub in sub_list:
    print(sub)
    # Load output from B10 (Argo profile data)
    df = pkl.load(open(f'{data_dir}/HurricaneVariableCovDF_Subset{sub}.pkl', 'rb'))

    X = np.array(df[['standard_signed_angle', 'hurricane_dtd']]).copy()

    bounds_x1 = (-8, +8)
    bounds_x2 = (-2, +20)
    train_knots = grid(bounds_x1, bounds_x2, 33, 45)
    n_param = train_knots.shape[0] + 3
    shape = (100, 400)
    test_X = grid(bounds_x1, bounds_x2, *shape)

    # Define lambda values to test
    LAMB_ = np.linspace(0.05, 10.0, 200) 
    # Add first value to array and rename it...
    LAMB = np.zeros(len(LAMB_)+1)
    LAMB[0] = 0.01
    LAMB[1:] = LAMB_
    
    y = variable_diff(df, stage, 0)
    # Initialize array for LOOCV metrics
    LOOCV = np.zeros((len(LAMB), DEPTH_IDX, 4, len(y)))
    
    # Loop through depths
    for depth_idx in range(DEPTH_IDX):
        print(depth_idx)
        
        y = variable_diff(df, stage, depth_idx)
        # I don't think these two lines are needed...
        #S = df['var'].apply(lambda x: x[depth_idx]).values
        #W = 1 / S
        # Load output from B10 (block covariance script)
        block_S = pkl.load(open(f'{results_dir}/BlockCovmat_{window_size_gp}_'
                                    f'{(depth_idx+1)*10:03d}{sub}.pkl', 'rb'))
        block_W = pkl.load(open(f'{results_dir}/BlockPremat_{window_size_gp}_'
                                    f'{(depth_idx+1)*10:03d}{sub}.pkl', 'rb'))
    
        # Loop through lambda values
        for idx, lamb in enumerate(LAMB):
            print(lamb)
            # Perform the thin plate spline
            tps_block = ThinPlateSpline(lamb=lamb, knots=train_knots)
            tps_block.fit(X, y, W=block_W)
            ret = tps_block.predict(X, sd='diag', S=block_S, k=2, diag_H=True)
            
            # Calculate metrics needed to compute the LOOCV score
            var_inv = (1/ret[1])**2
            norm_var_inv = var_inv / var_inv.sum()
            resid = (y - ret[0])
            diag_H = ret[3]
            
            # Calculate LOOCV score (eq 44 in paper)
            LOOCV[idx, depth_idx, 0, :] = norm_var_inv * (
                    (resid / (1 - diag_H)) ** 2)
            # Store other parameters from the thin plate spline...
            LOOCV[idx, depth_idx, 1, :] = ret[0]
            LOOCV[idx, depth_idx, 2, :] = ret[1]
            LOOCV[idx, depth_idx, 3, :] = ret[3]
    
    # Save lambda values and LOOCV scores
    loocv_data = (LAMB, LOOCV)
    pkl.dump(loocv_data, open(f'{results_dir}/B32_LOOCV_Data_{sub}.pkl', 'wb'))

