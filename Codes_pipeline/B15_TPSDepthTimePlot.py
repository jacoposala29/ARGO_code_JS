import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import os
import sys

from TC_and_TS_define_param import grid_lower, grid_upper, grid_stride, var2use, folder2use, depth_layers

sys.path.append('./implementations/')
from implementation_tools import grid


DEPTH_IDX = depth_layers

if DEPTH_IDX > 1:
    NULL = object()
    
    def plot_2d(preds, mask=NULL, clim=2, cbar=True,
            bounds_x1=(-8, +8),
    	bounds_x2=(-2, +20)):
        minima, maxima = -clim, +clim
        norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=False)
        cmap = cm.bwr
        cmap.set_bad(color='gray')
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        if mask is not NULL:
            preds = preds.copy()
            preds[~mask.astype(bool)] = np.nan
        plt.imshow(preds,
                origin='lower',
                cmap=cmap, norm=norm,
                extent=(*bounds_x1, *bounds_x2),
                aspect='auto',
                )
        plt.gca().invert_yaxis()
        if cbar:
            plt.colorbar()
        # plt.axhline(0, color='k', linewidth=0.5)
        plt.axvline(0, color='k', linewidth=0.5)
        return plt
    
    filt_list = ['all_hurricanes',
            'all_tstd',
            'all_combined',
            'decreasing_hurricanes',
            'decreasing_tstd',
            'decreasing_combined',
            'increasing_hurricanes',
            'increasing_tstd',
            'increasing_combined',
            ]
    for filt in filt_list:
        data_dir =  str(folder2use) + '/Outputs/'
        fig_dir = '../Figures/B15/'
        
        tps_est_dir =str(folder2use) + '/Outputs/'
        
        BASIS = pkl.load(open(f'{tps_est_dir}/TPS_Basis_Block_0.5_{filt}.pkl',
            'rb'))
        THETA = pkl.load(open(f'{tps_est_dir}/TPS_Theta_Block_0.5_{filt}.pkl',
            'rb'))
        COV = pkl.load(open(f'{tps_est_dir}/TPS_CovTheta_Block_0.5_{filt}.pkl',
            'rb'))
        
        center = 0 #int(sys.argv[1]) da cambiare a mano in base al cross track angle che si vuole valutare
        ws = 0.5
        n_params = THETA.shape[0]
        
        bounds_x1 = (-8, +8)
        bounds_x2 = (-2, +20)
        shape = (100, 400)
        test_X = grid(bounds_x1, bounds_x2, *shape)
        
        xc_filt = ((test_X[:, 0] > center - ws)
                  *(test_X[:, 0] < center + ws))
        
        TAU = np.sort(np.unique(test_X[:, 1]))
        n_tau = len(TAU)
        grid_start = int(grid_lower)
        grid_end = int(grid_upper)
        grid_stride = int(grid_stride)
        DEPTH_IDX = int((grid_end - grid_start)/ grid_stride+1)
        
        predmat = np.zeros((DEPTH_IDX, n_tau))
        maskmat = np.zeros((DEPTH_IDX, n_tau))
        
        for tidx, tau in enumerate(TAU):
            idxs = xc_filt * (test_X[:, 1] == tau)
            x_pts = test_X[idxs, 0]  # 1 x 6
            y_pts = BASIS[idxs, :]   # 6 x 394
            # trap integration for each column
            int_basis = np.zeros(n_params)
            var = np.zeros(DEPTH_IDX)
            for pidx in range(n_params):
                int_basis[pidx] = np.trapz(y_pts[:, pidx], x_pts)
            for depth_idx in range(DEPTH_IDX):
                var[depth_idx] = np.linalg.multi_dot((int_basis,
                                    COV[:, :, depth_idx],
                                    int_basis))
            inprod = np.dot(int_basis, THETA)
            predmat[:, tidx] = inprod / (max(x_pts) - min(x_pts))
            sd = np.sqrt(var)
            maskmat[:, tidx] = (inprod > 2 * sd) | (inprod < -2 * sd)
        
        # TODO: implement "hypothesis testing"
        # TODO: cache int_basis
        
        bounds_x1 = (-2, +20)
        bounds_x2 = (grid_start-5, grid_end+5)
        
        plt = plot_2d(
                predmat,
                bounds_x1=bounds_x1,
                bounds_x2=bounds_x2,
                cbar=False, clim=0.4
                )
        
        plt.xticks([0, 5, 10, 15, 20])
        #plt.yticks([10, 50, 100, 150, 200])
        plt.xlabel("Time difference (days)")
        plt.ylabel("Pressure (decibars)")
        plt.ylim([grid_start-5, grid_end+5])
        plt.gca().invert_yaxis()
        
        plt.savefig(f"{fig_dir}/DepthTimeInt_TPS_{filt}_Center{center}_{var2use}.pdf")
        plt.close()
        
        plt = plot_2d(
                predmat,
                bounds_x1=bounds_x1,
                bounds_x2=bounds_x2, clim=0.4
                )
        
        plt.xticks([0, 5, 10, 15, 20])
        #plt.yticks([10, 50, 100, 150, 200])
        plt.xlabel("Time difference (days)")
        plt.ylabel("Pressure (decibars)")
        plt.ylim([grid_start-5, grid_end+5])
        plt.gca().invert_yaxis()
        
        plt.savefig(f"{fig_dir}/DepthTimeInt_TPS_{filt}_Center{center}_Cbar_{var2use}.pdf")
        plt.close()
        
        plt = plot_2d(
                predmat,
                mask=maskmat,
                bounds_x1=bounds_x1,
                bounds_x2=bounds_x2,
                cbar=False, clim=0.4
                )
        
        plt.xticks([0, 5, 10, 15, 20])
        #plt.yticks([10, 50, 100, 150, 200])
        plt.xlabel("Time difference (days)")
        plt.ylabel("Pressure (decibars)")
        plt.ylim([grid_start-5, grid_end+5])
        plt.gca().invert_yaxis()
        
        plt.savefig(f"{fig_dir}/DepthTimeIntMask_TPS_{filt}_Center{center}_{var2use}.pdf")
        plt.close()
        
        plt = plot_2d(
                predmat,
                mask=maskmat,
                bounds_x1=bounds_x1,
                bounds_x2=bounds_x2, clim=0.4
                )
        
        plt.xticks([0, 5, 10, 15, 20])
        #plt.yticks([10, 50, 100, 150, 200])
        plt.xlabel("Time difference (days)")
        plt.ylabel("Pressure (decibars)")
        plt.ylim([grid_start-5, grid_end+5])
        plt.gca().invert_yaxis()
        
        plt.savefig(f"{fig_dir}/DepthTimeIntMask_TPS_{filt}_Center{center}_Cbar_{var2use}.pdf")
        plt.close()
