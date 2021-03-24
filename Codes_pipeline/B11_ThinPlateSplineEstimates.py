from itertools import product
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

# Set global variables
output_dir = folder2use + '/Outputs'
lamb = 0.5
data_dir = folder2use + '/Outputs'
results_dir = folder2use + '/Outputs'
window_size_gp = 5
stage = 'adj'

# Set dimensions of the grid
#DEPTH_IDX = 1
grid_start = int(grid_lower)
grid_end = int(grid_upper)
grid_stride = int(grid_stride)
DEPTH_IDX = int((grid_end - grid_start)/ grid_stride+1)

#sub = sys.argv[1]
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
else: 
    sub_list = (
    'all_combined',
    'all_hurricanes',
    'all_tstd',
)


for sub in sub_list:
    print(sub)
    df = pkl.load(open(f'{data_dir}/HurricaneVariableCovDF_Subset{sub}.pkl', 'rb'))

    X = np.array(df[['standard_signed_angle', 'hurricane_dtd']]).copy()

    bounds_x1 = (-8, +8)
    bounds_x2 = (-2, +20)
    train_knots = grid(bounds_x1, bounds_x2, 17, 23)
    n_param = train_knots.shape[0] + 3
    shape = (100, 400)
    test_X = grid(bounds_x1, bounds_x2, *shape)
    n_test = test_X.shape[0]

    PREDS_NOREW = np.zeros((n_test, DEPTH_IDX))
    PREDS_DIAG  = np.zeros((n_test, DEPTH_IDX))
    PREDS_BLOCK = np.zeros((n_test, DEPTH_IDX))
    STDEV_DIAG  = np.zeros((n_test, DEPTH_IDX))
    STDEV_BLOCK = np.zeros((n_test, DEPTH_IDX))
    MASK_DIAG  = np.zeros((n_test, DEPTH_IDX))
    MASK_BLOCK = np.zeros((n_test, DEPTH_IDX))
    THETA_BLOCK = np.zeros((n_param, DEPTH_IDX))
    BASIS_BLOCK = np.zeros((n_test, n_param)) # Invariant to depth
    COV_THETA_BLOCK = np.zeros((n_param, n_param, DEPTH_IDX))

    for depth_idx in range(DEPTH_IDX):
        print(depth_idx)
        y = variable_diff(df, stage, depth_idx)
        S = df['var'].apply(lambda x: x[depth_idx]).values
        W = 1 / S
        # Load B10 outputs
        block_S = pkl.load(open(f'{results_dir}/BlockCovmat_{window_size_gp}_'
                                    f'{(depth_idx+1)*10:03d}{sub}.pkl', 'rb'))
        block_W = pkl.load(open(f'{results_dir}/BlockPremat_{window_size_gp}_'
                                    f'{(depth_idx+1)*10:03d}{sub}.pkl', 'rb'))

        tps_norew = ThinPlateSpline(lamb=lamb, knots=train_knots)
        tps_diag  = ThinPlateSpline(lamb=lamb, knots=train_knots)
        tps_block = ThinPlateSpline(lamb=lamb, knots=train_knots)
        tps_norew.fit(X, y)
        tps_diag.fit(X, y, W=W)
        tps_block.fit(X, y, W=block_W)
        ret0 = tps_norew.predict(test_X)
        ret1 = tps_diag.predict(test_X, sd=True, S=S, k=2)
        ret2 = tps_block.predict(test_X, sd=True, S=block_S, k=2)
        PREDS_NOREW[:, depth_idx] = ret0
        PREDS_DIAG[:, depth_idx] = ret1[0]
        PREDS_BLOCK[:, depth_idx] = ret2[0]
        STDEV_DIAG[:, depth_idx] = ret1[1]
        STDEV_BLOCK[:, depth_idx] = ret2[1]
        MASK_DIAG[:, depth_idx] = ret1[2]
        MASK_BLOCK[:, depth_idx] = ret2[2]
        THETA_BLOCK[:, depth_idx] = tps_block.theta
        BASIS_BLOCK[:, :] = tps_block._test_basis
        COV_THETA_BLOCK[:, :, depth_idx] = tps_block._cov_theta

    pkl.dump(PREDS_NOREW,
            open(f'{output_dir}/TPS_Preds_NoRew_{lamb}_{sub}.pkl', 'wb'))
    pkl.dump(PREDS_DIAG,
            open(f'{output_dir}/TPS_Preds_Diag_{lamb}_{sub}.pkl', 'wb'))
    pkl.dump(PREDS_BLOCK,
            open(f'{output_dir}/TPS_Preds_Block_{lamb}_{sub}.pkl', 'wb'))
    pkl.dump(STDEV_DIAG,
            open(f'{output_dir}/TPS_Stdev_Diag_{lamb}_{sub}.pkl', 'wb'))
    pkl.dump(STDEV_BLOCK,
            open(f'{output_dir}/TPS_Stdev_Block_{lamb}_{sub}.pkl', 'wb'))
    pkl.dump(MASK_DIAG,
            open(f'{output_dir}/TPS_Mask_Diag_{lamb}_{sub}.pkl', 'wb'))
    pkl.dump(MASK_BLOCK,
            open(f'{output_dir}/TPS_Mask_Block_{lamb}_{sub}.pkl', 'wb'))
    pkl.dump(THETA_BLOCK,
            open(f'{output_dir}/TPS_Theta_Block_{lamb}_{sub}.pkl', 'wb'))
    pkl.dump(BASIS_BLOCK,
            open(f'{output_dir}/TPS_Basis_Block_{lamb}_{sub}.pkl', 'wb'))
    pkl.dump(COV_THETA_BLOCK,
            open(f'{output_dir}/TPS_CovTheta_Block_{lamb}_{sub}.pkl', 'wb'))
