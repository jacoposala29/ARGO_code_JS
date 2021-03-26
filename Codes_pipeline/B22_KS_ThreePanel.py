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
from TC_and_TS_define_param import grid_lower, grid_upper, grid_stride, folder2use, depth_layers, var2use, clim

#import plot_config

parser = argparse.ArgumentParser(
        description='Plot kernel smoothed estimates (three panel)')
parser.add_argument('--integrated', dest='mode', action='store_const',
                    const='integrated', default='gridded',
                    help='process assuming integrated heat content (default: gridded)')
args = parser.parse_args()

input_dir = folder2use + '/Outputs'
plot_dir = '../Figures/B22/'
window_size_gp = 5
DEPTH_IDX = depth_layers
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
    df_raw = pkl.load(open(f'{input_dir}/KernelSmoothedMatx_0.2_raw_{sub}.pkl', 'rb'))
    df_mf = pkl.load(open(f'{input_dir}/KernelSmoothedMatx_0.2_mf_{sub}.pkl', 'rb'))
    df_adj = pkl.load(open(f'{input_dir}/KernelSmoothedMatx_0.2_adj_{sub}.pkl', 'rb'))
    
    
    # mat1 = pkl.load(open(f'{input_dir}/KernelSmoothedMatx_0.2_{stage}_{subset}.pkl', 'rb'))
    # mat2 = pkl.load(open(f'{input_dir}/KernelSmoothedMatx_0.2_mf_combined.pkl', 'rb'))
    # mat3 = pkl.load(open(f'{input_dir}/KernelSmoothedMatx_0.2_adj_combined.pkl', 'rb'))
    
    
    n, k = df_raw.shape
    
    for idx in range(k):
        preds1 = df_raw[:, idx]
        preds2 = df_mf[:, idx]
        preds3 = df_adj[:, idx]
    
        bounds_x1 = (-8, +8)
        bounds_x2 = (-2, +20)
        shape = (100, 400)
        minima, maxima = -clim, +clim
        depth = int(grid_lower) + idx*int(grid_stride)

        norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=False)
        cmap = cm.bwr
        cmap.set_bad(color='gray')
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    
        fig = plt.figure(figsize=(20, 12))
        gs= gridspec.GridSpec(1, 3, figure=fig,
            width_ratios=[1.0, 1.0, 1.0666])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])
    
        ax1.imshow(preds1.reshape(*(shape[::-1])),
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
        ax1.set_title(r'(a) Raw',
                y=ypos,
                fontsize=fs+df)
    
    
        ax2.imshow(preds2.reshape(*(shape[::-1])),
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
        ax2.set_title(r'(b) Fitted mean field',
                y=ypos,
                fontsize=fs+df)
    
        ax3.imshow(preds3.reshape(*(shape[::-1])),
                origin='lower',
                cmap=cmap, norm=norm,
                extent=(*bounds_x1, *bounds_x2),
                )
        ax3.invert_yaxis()
        ax3.axhline(0, color='k', linewidth=0.5)
        ax3.axvline(0, color='k', linewidth=0.5)
        ax3.set_xticks([-5, 0, +5])
        #ax3.xaxis.set_ticks_position('top') 
        ax3.set_title(r'(c) Seasonally adjusted',
                y=ypos,
                fontsize=fs+df)
    
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(mapper, ax=ax3, cax=cax)
        cbar.set_label(r'Temperature difference ($^\circ$C)',
                fontsize=fs)
        ax3.set_yticks([])
    
        plt.savefig(f'{plot_dir}/KS_ThreePanel_{var2use}_{depth:03d}_{sub}.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.savefig(f'{plot_dir}/KS_ThreePanel_{var2use}_{depth:03d}_{sub}.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
