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
from TC_and_TS_define_param import grid_lower, grid_upper, grid_stride, var2use, folder2use, depth_layers

DEPTH_IDX = depth_layers

if DEPTH_IDX > 1:
    NULL = object()
    # Set dimensions of the grid
    grid_start = int(grid_lower)
    grid_end = int(grid_upper)
    grid_stride = int(grid_stride)
    
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
        data_dir = str(folder2use) + '/Outputs'
        fig_dir = '../Figures/B16'
        tps_est_dir =  str(folder2use) + '/Outputs'
        
        BASIS = pkl.load(open(f'{tps_est_dir}/TPS_Basis_Block_0.5_{filt}.pkl',
            'rb'))
        THETA = pkl.load(open(f'{tps_est_dir}/TPS_Theta_Block_0.5_{filt}.pkl',
            'rb'))
        COV = pkl.load(open(f'{tps_est_dir}/TPS_CovTheta_Block_0.5_{filt}.pkl',
            'rb')) 
        
        phase_list= ['Forced', 'Recovery']
        for phase in phase_list:
            if phase == 'Forced':
                d1 = 0
                d2 = 3
            elif phase == 'Recovery':
                d1 = 3
                d2 = 20
            else:
                raise ValueError
        
            n_params = THETA.shape[0]
            
            bounds_x1 = (-8, +8)
            bounds_x2 = (-2, +20)
            shape = (100, 400)
            test_X = grid(bounds_x1, bounds_x2, *shape)
            
            xc_filt = ((test_X[:, 1] >= d1)
                      *(test_X[:, 1] < d2))
            
            D = np.sort(np.unique(test_X[:, 0]))
            n_d = len(D)
            DEPTH_IDX = int((grid_end - grid_start)/ grid_stride+1)
            
            predmat = np.zeros((DEPTH_IDX, n_d))
            maskmat = np.zeros((DEPTH_IDX, n_d))
            
            for tidx, d in enumerate(D):
                idxs = xc_filt * (test_X[:, 0] == d)
                x_pts = test_X[idxs, 1]  # 1 x 6
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
            
            predmat_small = predmat
            maskmat_small = maskmat
            
            predmat = np.zeros((DEPTH_IDX, n_d*4))
            maskmat = np.zeros((DEPTH_IDX, n_d*4))
            xnew = np.linspace(-8, +8, n_d*4)
            for depth_idx in range(DEPTH_IDX):
                predmat[depth_idx, :] = np.interp(xnew, D, predmat_small[depth_idx, :])
                maskmat[depth_idx, :] = np.repeat(maskmat_small[depth_idx, :], 4)
            
            
            bounds_x1 = (-8, +8)
            bounds_x2 = (grid_start-5, grid_end+5)
            
            plt = plot_2d(
                    predmat,
                    bounds_x1=bounds_x1,
                    bounds_x2=bounds_x2,
                    cbar=False,
                    clim = 0.4
                    )
            
            plt.xticks([-5, 0, +5])
            #plt.yticks([10, 20, 30, 40, 50])
            plt.ylim([grid_start-5, grid_end+5])
            plt.xlabel("Cross-track angle")
            plt.ylabel("Pressure (decibars)")
            plt.gca().invert_yaxis()
            
            plt.savefig(f"{fig_dir}/DepthCrossTrackInt_TPS_{filt}_Phase{phase}_{var2use}.pdf")
            plt.close()
            
            plt = plot_2d(
                    predmat,
                    bounds_x1=bounds_x1,
                    bounds_x2=bounds_x2,
                    clim = 0.4
                    )
            
            plt.xticks([-5, 0, +5])
            #plt.yticks([10, 20, 30, 40, 50])
            plt.ylim([grid_start-5, grid_end+5])
            plt.xlabel("Cross-track angle")
            plt.ylabel("Pressure (decibars)")
            plt.gca().invert_yaxis()
            
            plt.savefig(f"{fig_dir}/DepthCrossTrackInt_TPS_{filt}_Phase{phase}_Cbar_{var2use}.pdf")
            plt.close()
            
            plt = plot_2d(
                    predmat,
                    mask=maskmat,
                    bounds_x1=bounds_x1,
                    bounds_x2=bounds_x2,
                    cbar=False,
                    clim = 0.4
                    )
            
            plt.xticks([-5, 0, +5])
            #plt.yticks([10, 20, 30, 40, 50])
            plt.ylim([grid_start-5, grid_end+5])
            plt.xlabel("Cross-track angle")
            plt.ylabel("Pressure (decibars)")
            plt.gca().invert_yaxis()
            
            plt.savefig(f"{fig_dir}/DepthCrossTrackIntMask_TPS_{filt}_Phase{phase}_{var2use}.pdf")
            plt.close()
            
            plt = plot_2d(
                    predmat,
                    mask=maskmat,
                    bounds_x1=bounds_x1,
                    bounds_x2=bounds_x2,
                    clim = 0.4
                    )
            
            plt.xticks([-5, 0, +5])
            #plt.yticks([10, 20, 30, 40, 50])
            plt.ylim([grid_start-5, grid_end+5])
            plt.xlabel("Cross-track angle")
            plt.ylabel("Pressure (decibars)")
            plt.gca().invert_yaxis()
            
            plt.savefig(f"{fig_dir}/DepthCrossTrackIntMask_TPS_{filt}_Phase{phase}_Cbar_{var2use}.pdf")
            plt.close()
