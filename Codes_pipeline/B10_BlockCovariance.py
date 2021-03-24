'''
author: addison@stat.cmu.edu

Construct block diagonal covariance matrices.
'''
import numpy as np
import pandas as pd
import pickle as pkl
import scipy.sparse as sps
import multiprocessing
import time 

from functools import partial
from itertools import product

from tools import covariance_matrix
from TC_and_TS_define_param import grid_lower, grid_upper, grid_stride, folder2use, depth_layers

CPU_COUNT = multiprocessing.cpu_count() # 4

# Set dimensions of the grid
grid_start = int(grid_lower)
grid_end = int(grid_upper)
grid_stride = int(grid_stride)
DEPTH_IDX = int((grid_end - grid_start)/ grid_stride+1)

# Import data from B09 output 
data_dir = folder2use + '/Outputs'
df = pkl.load(open(f'{data_dir}/HurricaneAdjRawVariableCovDF.pkl', 'rb'))
df.shape # (16674, 21)
df = df[np.log(df['var'].apply(lambda x: np.min(x))) >= -4.5].sort_values([
    'before_pid',
    'after_pid',
])

# Calculate differences 50 dbar - 10 dbar
DEPTH_IDX = depth_layers
if DEPTH_IDX > 1:
    diff_raw_after = np.zeros(df.shape[0])
    diff_raw_before = np.zeros(df.shape[0])
    for i in np.arange(0,df.shape[0]):
        diff_raw_after[i] = df.raw_after_variable.iloc[i][4] - df.raw_after_variable.iloc[i][0]
        diff_raw_before[i] = df.raw_before_variable.iloc[i][4] - df.raw_before_variable.iloc[i][0]

    # Separates increasing and decreasing profile pairs
    DFS = [
        (df[(diff_raw_before < 0)], 'decreasing_combined'),
        (df[(diff_raw_before > 0)], 'increasing_combined'),
        (df[(diff_raw_before < 0) & (df['wind'] >= 64)], 'decreasing_hurricanes'),
        (df[(diff_raw_before > 0) & (df['wind'] >= 64)], 'increasing_hurricanes'),
        (df[(diff_raw_before < 0) & (df['wind'] < 64)], 'decreasing_tstd'),
        (df[(diff_raw_before > 0) & (df['wind'] < 64)], 'increasing_tstd'),
        (df[df['wind'] < 64], 'all_tstd'),
        (df[df['wind'] >= 64], 'all_hurricanes'),
        (df, 'all_combined')
        ]
else:
    DFS = [
        (df[df['wind'] < 64], 'all_tstd'),
        (df[df['wind'] >= 64], 'all_hurricanes'),
        (df, 'all_combined')
        ]


# salvare DFS poi creo B10_2

time.sleep(5)

for (df, sub) in DFS:
    print(sub)
    # Saves file
    pkl.dump(df, open(f'{data_dir}/HurricaneVariableCovDF_Subset{sub}.pkl', 'wb'))
    
    # Import directions 
    results_dir = folder2use + '/Outputs'
    window_size_gp = 5
    DEPTH_IDX = int((grid_end - grid_start)/ grid_stride+1)
    
    depth_idx = 1
    before_pids = np.sort(df['before_pid'].unique())
    
    MLE = pkl.load(open(f'{results_dir}/MleCoefDF_{window_size_gp}.pkl', 'rb'))
    
    df_list = [df[df['before_pid']==bp] for bp in before_pids]
    for depth_idx in range(DEPTH_IDX):
        print(depth_idx)
        cov = partial(covariance_matrix, df_param=MLE, depth_idx=depth_idx)
    
        with multiprocessing.Pool(processes=CPU_COUNT) as pool:
            covmat_list = pool.map(cov, df_list)
    
        with multiprocessing.Pool(processes=CPU_COUNT) as pool:
            premat_list = pool.map(np.linalg.inv, covmat_list)
    
        C = sps.block_diag(covmat_list)
        P = sps.block_diag(premat_list)
        pkl.dump(C, open(f'{results_dir}/BlockCovmat_{window_size_gp}_'
                         f'{(depth_idx+1)*10:03d}{sub}.pkl', 'wb'))
        pkl.dump(P, open(f'{results_dir}/BlockPremat_{window_size_gp}_'
                         f'{(depth_idx+1)*10:03d}{sub}.pkl', 'wb'))
