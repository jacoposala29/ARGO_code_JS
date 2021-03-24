import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import sys
sys.path.append('./implementations/')
from implementation_tools import (
    variable_diff,
)

from TC_and_TS_define_param import grid_lower, grid_upper, grid_stride, folder2use

# Set global variables
output_dir = folder2use + '/Outputs'

# Set dimensions of the grid
grid_start = int(grid_lower)
grid_end = int(grid_upper)
grid_stride = int(grid_stride)
DEPTH_IDX = int((grid_end - grid_start)/ grid_stride+1)

data_dir = folder2use + '/Outputs'
results_dir = folder2use + '/Outputs'
window_size_gp = 5
stage = 'adj'

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
else:  # single depth
    sub_list = (
    'all_combined',
    'all_hurricanes',
    'all_tstd',
)
    
# Loop through all cases
for sub in sub_list:
    print(sub)
    # Load output from B10 (Argo profile data)
    df = pkl.load(open(f'{data_dir}/HurricaneVariableCovDF_Subset{sub}.pkl', 'rb'))

    y = variable_diff(df, stage, 0)
    ymat = np.zeros((1, DEPTH_IDX, len(y)))

    for depth_idx in range(DEPTH_IDX):
        ymat[0, depth_idx, :] = variable_diff(df, stage, depth_idx)
     
    # Load results from B32 (lambda values and LOOCV scores)
    #LAMB_SMALL, LOOCV_SMALL = pkl.load(open(f'{data_dir}/B32_LOOCV_Data_{sub}.pkl', 'rb'))
    # Load results from B35 (lambda values and LOOCV scores)
    LAMB_LARGE, LOOCV_LARGE = pkl.load(open(f'{data_dir}/B35_LOOCV_Data_{sub}.pkl', 'rb'))
    # Combine the two sets
    LAMB = LAMB_LARGE[1:]
    LOOCV = LOOCV_LARGE[1:, :, :, :] 
    n = LOOCV.shape[3]
    
    # 0 - ignored
    # 1 - predictions
    # 2 - estimated standard deviations
    # 3 - diag_H
    
    varmat = np.vstack(df['var']).T
    varmat = varmat.reshape(1, *varmat.shape)
    
    # LOOCV metric (eq 44 in paper)
    loocv_estimates = ((1/varmat) * (
        (LOOCV[:,:,1,:] - ymat) / (1-LOOCV[:,:,3,:]))**2)
    
    error_estimates = loocv_estimates.sum(axis=2)
    # Set NaN error to infinity
    error_estimates_nonan = error_estimates.copy()
    error_estimates_nonan[np.isnan(error_estimates)] = np.inf
    # Select lambas that minimize the cost function
    lhat = error_estimates_nonan.argmin(axis=0)
    LAMB[lhat]
    print(LAMB[lhat])
    
    # Rough version of plot 13b in paper
    plt.figure()
    plt.scatter(LAMB[lhat], np.linspace(grid_start,grid_end,DEPTH_IDX), marker='x')
    plt.plot(LAMB[lhat], np.linspace(grid_start,grid_end,DEPTH_IDX))
    plt.gca().invert_yaxis()
    plt.xscale('log')
    plt.xlim(min(LAMB), max(LAMB)*1.1)
    plt.xlabel('Regularization parameter, $\lambda$')
    plt.ylabel('Pressure, dbars ($z$)')
    plt.show()

#DA SISTEMARE!!!!!!!!
    # Plot 13a in paper
    plt.figure()
    ax1 = plt.subplot()
    for depth_idx in range(DEPTH_IDX):
        # Plot error function vs lambda
        if depth_idx < 10:
            _ = plt.plot(LAMB, error_estimates[:, depth_idx], label=f'{(depth_idx+1)*10}')
        else:
            _ = plt.plot(LAMB[1:-1], (error_estimates[1:-1, depth_idx] - error_estimates[0:-2, depth_idx])/(LAMB[1:-1]-(LAMB[0:-2])), label=f'{(depth_idx+1)*10}')#LAMB, error_estimates[:, depth_idx], label=f'{(depth_idx+1)*10}', linestyle='dotted')
        # Highlist selected lambda
        _ = plt.scatter(LAMB[lhat[depth_idx]], error_estimates[lhat[depth_idx], depth_idx],
                marker='x')


    # # Plot 13a in paper
    # plt.figure()
    # ax1 = plt.subplot()
    # for depth_idx in range(DEPTH_IDX):
    #     # Plot error function vs lambda
    #     plt.plot(LAMB[1:-1], (error_estimates[1:-1, depth_idx] - error_estimates[0:-2, depth_idx])/(LAMB[1:-1]-(LAMB[0:-2])), label=f'{(depth_idx+1)*10}')#LAMB, error_estimates[:, depth_idx], label=f'{(depth_idx+1)*10}', linestyle='dotted')
    #     ax1.set_ylim(-20,0)
    #     # Highlist selected lambda
    #     _ = plt.scatter(LAMB[lhat[depth_idx]], error_estimates[lhat[depth_idx], depth_idx],
    #             marker='x')
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Regularization parameter, $\lambda$')
    plt.ylabel('LOOCV Error')
    #plt.legend(title='Pressure')

    # # Plot 14 in paper
    # for depth_idx in range(DEPTH_IDX):
    #     _ = plt.plot(LAMB, error_estimates[:, depth_idx], label=f'{(depth_idx+1)*10}')
    #     _ = plt.scatter(LAMB[lhat[depth_idx]], error_estimates[lhat[depth_idx], depth_idx],
    #             marker='x')

    # plt.xscale('log')
    # #plt.legend()
    
    # depth_idx=2
    # _ = plt.plot(LAMB, error_estimates[:, depth_idx], label=f'{(depth_idx+1)*10}')
    # _ = plt.scatter(LAMB[lhat[depth_idx]], error_estimates[lhat[depth_idx], depth_idx],
    #         marker='x')
    
    # plt.xscale('log')
    # #plt.legend()

    # error_mean = LOOCV.mean(axis=2)
    # error_std = LOOCV.std(axis=2) / np.sqrt(n)
    # #plt.errorbar(LAMB, error_mean[:, depth_idx], yerr=error_std[:, depth_idx])
    # #plt.errorbar(LAMB[:10], error_mean[:10, depth_idx], yerr=error_std[:10, depth_idx])
