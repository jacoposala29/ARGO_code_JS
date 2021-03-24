import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import os
import sys

sys.path.append('./implementations/')
from implementation_tools import plot_2d


from TC_and_TS_define_param import grid_lower, grid_upper, grid_stride, var2use, folder2use

# Set dimensions of the grid
grid_start = int(grid_lower)
grid_end = int(grid_upper)
grid_stride = int(grid_stride)
DEPTH_IDX = int((grid_end - grid_start)/ grid_stride+1)


input_dir = folder2use + '/Outputs'
plot_dir = '../Figures/B12/'

f_preds = [f for f in os.listdir(input_dir) if 'TPS_Preds' in f]
for f_in in f_preds:
    print(f_in)
    estimates = pkl.load(open(f'{input_dir}/{f_in}', 'rb'))
    params = (f_in[:-4]).split('_')
    for idx in range(DEPTH_IDX):
        shape = (100, 400)
        plt = plot_2d(estimates[:, idx], shape, clim=0.4)
        depth = int(grid_lower) + idx*int(grid_stride)
        plt.title(f'Depth: {depth}, {params[2]}, lamb={params[3]}; {params[5]}, 'f'{params[4]}')
        f_out = f'TPS_{params[2]}_{params[3]}_{params[4]}_{params[5]}_{var2use}{depth:03d}'
        plt.tight_layout()
        #plt.savefig(f'{plot_dir}/{f_out}.pdf', bbox_inches = 'tight')
        plt.savefig(f'{plot_dir}/{f_out}.png',bbox_inches = 'tight', dpi=300)
        plt.close()

for f_in in f_preds:
    print(f_in)
    if 'NoRew' in f_in:
        continue
    estimates = pkl.load(open(f'{input_dir}/{f_in}', 'rb'))
    masks = pkl.load(open(f'{input_dir}/{f_in}'.replace('Preds', 'Mask'), 'rb')).astype(bool)
    params = (f_in[:-4]).split('_')
    for idx in range(DEPTH_IDX):
        shape = (100, 400)
        plt = plot_2d(estimates[:, idx], shape, mask=masks[:, idx], clim=0.4)
        depth = int(grid_lower) + idx*int(grid_stride)
        plt.title(f'Depth: {depth}, {params[2]}, lamb={params[3]}; {params[5]}, 'f'{params[4]}')
        f_out = f'TPS_{params[2]}_{params[3]}_{params[4]}_{params[5]}_Masked_{var2use}{depth:03d}'
        plt.tight_layout()
        #plt.savefig(f'{plot_dir}/{f_out}.pdf', bbox_inches = 'tight')
        plt.savefig(f'{plot_dir}/{f_out}.png', bbox_inches = 'tight', dpi=300)
        plt.close()
