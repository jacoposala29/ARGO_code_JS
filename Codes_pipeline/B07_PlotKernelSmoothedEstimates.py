import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import sys

from itertools import product

sys.path.append('./implementations/')
from implementation_tools import plot_2d
from TC_and_TS_define_param import folder2use, depth_layers

input_dir = str(folder2use) + '/Outputs/'
plot_dir = '../Figures/B07/'
h = 0.2
DEPTH_IDX = depth_layers
STAGE = (
    'adj',
    'raw',
    'mf',
)

if DEPTH_IDX > 1:
    SUB = (
        'all_combined',
        'all_hurricanes',
        'all_tstd',
        'increasing01_combined',
        'decreasing01_combined',
        'increasing01_hurricanes',
        'decreasing01_hurricanes',
        'increasing01_tstd',
        'decreasing01_tstd',
    )
else:
    SUB = (
        'all_combined',
        'all_hurricanes',
        'all_tstd',
    )

# Aggiungere caso con tutti (increasing + decreasing) 
from TC_and_TS_define_param import var2use,grid_lower,grid_stride
   
for stage, sub in product(STAGE, SUB):
    print(stage, sub)
    f_in = f'{input_dir}/KernelSmoothedMatx_{h}_{stage}_{sub}.pkl'
    estimates = pkl.load(open(f_in, 'rb'))
    n, k = estimates.shape
    for idx in range(k):
        shape = (100, 400)
        plt = plot_2d(estimates[:, idx], shape, clim=0.4)
        #depth = (idx+1)*10
        # vmin = -0.5
        # vmax = 0.5
        depth = int(grid_lower) + idx*int(grid_stride)
        plt.title(f'Depth: {depth}; {stage} {sub} {var2use}')
        plt.tight_layout()
        #plt.savefig(f'{plot_dir}/KS_{stage}_{sub}_{var2use}_{depth}.pdf', bbox_inches = 'tight')
        plt.savefig(f'{plot_dir}/KS_{stage}_{sub}_{var2use}' + "{0:0=3d}".format(depth) + '.png', bbox_inches = 'tight', dpi=300)
        plt.close()
