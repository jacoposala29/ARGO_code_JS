'''
author: addison@stat.cmu.edu

Plot time-depth
'''
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import os
import sys

sys.path.append('./implementations/')
from implementation_tools import grid
from Regressors import (
        KernelSmoother,
        ThinPlateSpline,
    )
from TC_and_TS_define_param import grid_lower, grid_upper, grid_stride, folder2use, var2use, depth_layers

DEPTH_IDX = depth_layers

if DEPTH_IDX > 1:
    NULL = object()
    
    def plot_2d(preds, shape, mask=NULL, clim=2, cbar=True,
            bounds_x1=(-8, +8),
    	bounds_x2=(-2, +20)):
        minima, maxima = -clim, +clim
        norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=False)
        cmap = cm.bwr
        cmap.set_bad(color='gray')
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        if mask is not NULL:
            preds = preds.copy()
            preds[~mask] = np.nan
        plt.imshow(preds.reshape(*(shape[::-1])),
                origin='lower',
                cmap=cmap, norm=norm,
                extent=(*bounds_x1, *bounds_x2),
                aspect='auto',
                )
        plt.gca().invert_yaxis()
        if cbar:
            plt.colorbar()
        # plt.axhline(0, color='k', linewidth=0.5)
        # plt.axvline(0, color='k', linewidth=0.5)
        return plt
    
    
    data_dir = str(folder2use) + '/Outputs/'
    est_dir =  str(folder2use) + '/Outputs/'
    fig_dir = '../Figures/B14/'
    df = pkl.load(open(f'{data_dir}/HurricaneAdjRawVariableCovDF.pkl', 'rb'))
    print(df.shape)
    width = 0.5
    df = df.loc[
            (df['standard_signed_angle'] > -width)
           &(df['standard_signed_angle'] < +width)
        ]
    print(df.shape)
    
    n_train = df.shape[0]
    
    grid_start = int(grid_lower)
    grid_end = int(grid_upper)
    grid_stride = int(grid_stride)
    DEPTH_IDX = int((grid_end - grid_start)/ grid_stride+1)
    
    #DEPTH_IDX = 20
    train_X = np.zeros((DEPTH_IDX * n_train, 2))
    train_y = np.zeros(DEPTH_IDX * n_train)
    S = np.zeros(DEPTH_IDX * n_train)
    for i in range(n_train):
        train_y[(DEPTH_IDX*i):(DEPTH_IDX*(i+1))] = (
                df['adj_after_variable'].iloc[i]
                - df['adj_before_variable'].iloc[i]
            )
        S[(DEPTH_IDX*i):(DEPTH_IDX*(i+1))] = (
                df['var'].iloc[0])
    
    train_X[:, 0] = np.repeat(df['hurricane_dtd'].values, DEPTH_IDX)
    train_X[:, 1] = np.tile(np.arange(grid_start, grid_end+1, grid_stride), n_train)
    
    h = 0.2
    bounds_x1 = (-2, +20)
    bounds_x2 = (10, 200)
    shape = (400, 100)
    test_X = grid(bounds_x1, bounds_x2, *shape)
    n_test = test_X.shape[0]
    
    export_fname = f'{est_dir}/ks_pm{width}_h{h}_smoothed_time_depth.pkl'
    try:
        ests = pkl.load(open(export_fname, 'rb'))
    except FileNotFoundError:
        # KS
        ks = KernelSmoother(h=h)
        ks.fit(train_X, train_y)
        ests = ks.predict(test_X)
    
    
    plt = plot_2d(ests, shape,
            bounds_x1=bounds_x1,
            bounds_x2=bounds_x2,
            clim=0.5)
    plt.ylim(50,10)
    plt.xticks([0, 5, 10, 15, 20])
    #plt.yticks([10, 50, 100, 150, 200])
    plt.xlabel("Time difference (days)")
    plt.ylabel("Pressure (decibars)")
    plt.savefig(f"{fig_dir}/DepthTime_KS_h{h}_{var2use}.pdf")
