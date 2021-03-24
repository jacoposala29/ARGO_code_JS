import argparse
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import pickle as pkl
import os
import sys
sys.path.append('./implementations/')

from implementation_tools import (
    grid,
    variable_diff,
)
from TC_and_TS_define_param import grid_lower, grid_upper, grid_stride, folder2use, depth_layers, var2use, clim, unit

#from plot_config import plot_config

parser = argparse.ArgumentParser(
        description='Plot kernel smoothed estimates (three panel)')
parser.add_argument('--integrated', dest='mode', action='store_const',
                    const='integrated', default='gridded',
                    help='process assuming integrated heat content (default: gridded)')
args = parser.parse_args()

input_dir = folder2use + '/Outputs'
plot_dir = '../Figures/B21/'
window_size_gp = 5
stage = 'adj'
DEPTH_IDX = depth_layers
STAGE = ('adj', 'raw', 'mf')
h = 0.2 #bandwidth

fs = 24
df = 2
plt.rcParams['font.family'] = 'Liberation Serif'
#plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.it'] = 'serif:italic'
plt.rcParams['mathtext.bf'] = 'serif:bold'
plt.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams.update({'font.size': fs})

# Set dimensions of the grid
grid_start = int(grid_lower)
grid_end = int(grid_upper)
grid_stride = int(grid_stride)
DEPTH_IDX = int((grid_end - grid_start)/ grid_stride+1)

ypos = -0.3

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
    #df_all = pkl.load(open(f'{input_dir}/KernelSmoothedMatx_0.2_{stage}_{sub}.pkl', 'rb'))
    prefix = '_LOOCV'
    
    # NEED TO CHANGE IN FUTURE
    lamb_choice = 'Custom'
    norew = pkl.load(open(f'{input_dir}/TPS{prefix}_Preds_NoRew_{lamb_choice}_{sub}.pkl', 'rb'))
    block = pkl.load(open(f'{input_dir}/TPS{prefix}_Preds_Block_{lamb_choice}_{sub}.pkl', 'rb'))
    bmask = pkl.load(open(f'{input_dir}/TPS{prefix}_Mask_Block_{lamb_choice}_{sub}.pkl', 'rb'))
      
    # mat1 = pkl.load(open(f'{input_dir}/KernelSmoothedMatx_0.2_{stage}_{subset}.pkl', 'rb'))
    # mat2 = pkl.load(open(f'{input_dir}/KernelSmoothedMatx_0.2_mf_combined.pkl', 'rb'))
    # mat3 = pkl.load(open(f'{input_dir}/KernelSmoothedMatx_0.2_adj_combined.pkl', 'rb'))
    n, k = norew.shape
    
    idx = 0
    
    for idx in range(k):
        preds_norew = norew[:, idx]
        preds_block = block[:, idx]
        preds_mask = preds_block.copy()
        preds_mask[~bmask[:, idx].astype(bool)] = np.nan
        

        bounds_x1 = (-8, +8)
        bounds_x2 = (-2, +20)
        shape = (100, 400)
        minima, maxima = -clim, +clim
        depth = int(grid_lower) + idx*int(grid_stride)
        
        norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=False)
        cmap = cm.bwr
        cmap.set_bad(color='gray')
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    
        fig = plt.figure(figsize=(20, 8))
        gs= gridspec.GridSpec(1, 3, figure=fig,
            width_ratios=[1.0, 1.0, 1.0666])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])
                
        # Left panel
        ax1.imshow(preds_norew.reshape(*(shape[::-1])),
                origin='lower',
                cmap=cmap, norm=norm,
                extent=(*bounds_x1, *bounds_x2),
                )
        ax1.invert_yaxis()
        ax1.axhline(0, color='k', linewidth=0.5)
        ax1.axvline(0, color='k', linewidth=0.5)
        ax1.set_xticks([-5, 0, +5])
        ax1.set_yticks([0, 5, 10, 15, 20])
        ax1.set_ylabel(r'Days since TC passage ($\tau$)',
                fontsize=fs)
        ax1.set_title(r'(a) No variance reweighting',
                y=ypos,
                fontsize=fs+df)
    
        # Central panel
        ax2.imshow(preds_block.reshape(*(shape[::-1])),
                origin='lower',
                cmap=cmap, norm=norm,
                extent=(*bounds_x1, *bounds_x2),
                )
        ax2.invert_yaxis()
        ax2.axhline(0, color='k', linewidth=0.5)
        ax2.axvline(0, color='k', linewidth=0.5)
        ax2.set_xticks([-5, 0, +5])
        ax2.set_yticks([])
        #ax2.xaxis.set_ticks_position('top') 
        ax2.set_xlabel(r'Cross-track angle, degrees ($d$)',
                fontsize=fs)
        ax2.set_title(r'(b) Block covariance',
                y=ypos,
                fontsize=fs+df)
    
        # Right panel
        ax3.imshow(preds_mask.reshape(*(shape[::-1])),
                origin='lower',
                cmap=cmap, norm=norm,
                extent=(*bounds_x1, *bounds_x2),
                )
        ax3.invert_yaxis()
        ax3.axhline(0, color='k', linewidth=0.5)
        ax3.axvline(0, color='k', linewidth=0.5)
        ax3.set_xticks([-5, 0, +5])
        #ax3.xaxis.set_ticks_position('top') 
        ax3.set_title(r'(c) Pointwise $\alpha=0.05$ test',
                y=ypos,
                fontsize=fs+df)
    
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(mapper, ax=ax3, cax=cax)
        cbar.set_label(f'{var2use} difference ({unit})',
                fontsize=fs)
        ax3.set_yticks([])
                
        plt.suptitle({sub, f'{depth:03d}'} , fontsize = 16)
        plt.subplots_adjust(hspace=0.1)

        plt.savefig(f'{plot_dir}/TPS_ThreePanel_{var2use}_{depth:03d}_{sub}_custom.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.savefig(f'{plot_dir}/TPS_ThreePanel_{var2use}_{depth:03d}_{sub}_custom.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
