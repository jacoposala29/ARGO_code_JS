'''
author: addison@stat.cmu.edu

Constructs a diagonal covariance matrix in the form of a DataFrame column.
'''
import numpy as np
import pandas as pd
import pickle as pkl

from itertools import product

from tools import covariance_matrix
from TC_and_TS_define_param import grid_lower, grid_upper, grid_stride, folder2use

# Import and read pkl files 
data_dir = folder2use + '/Outputs'
results_dir = folder2use + '/Outputs'
window_size_gp = 5

# Set dimensions of the grid
grid_start = int(grid_lower)
grid_end = int(grid_upper)
grid_stride = int(grid_stride)
DEPTH_IDX = int((grid_end - grid_start)/ grid_stride+1)

# Import B05 output
HU = pkl.load(open(f'{data_dir}/HurricaneAdjRawVariableDF.pkl', 'rb'))
# Import B08 output
MLE = pkl.load(open(f'{results_dir}/MleCoefDF_{window_size_gp}.pkl', 'rb'))

# Sort (ordinare) values 
HU = HU.sort_values(['before_pid', 'after_pid']).reset_index(drop=True)
df_sub = HU.iloc[[0]]
covariance_matrix(df_sub, MLE, 0)

# Creates covariance matrix as in eq. 12 in paper draft
depth_idx = 0
var_mat = np.zeros((HU.shape[0], DEPTH_IDX))
for idx, depth_idx in product(range(HU.shape[0]), range(DEPTH_IDX)):
    var_mat[idx, depth_idx] = covariance_matrix(HU.iloc[[idx]], MLE, depth_idx)[0][0]

HU['var'] = pd.Series((row for row in var_mat))
pkl.dump(HU, open(f'{data_dir}/HurricaneAdjRawVariableCovDF.pkl', 'wb'))
