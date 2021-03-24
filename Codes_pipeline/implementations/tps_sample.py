import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import scipy.sparse as sps
import sys
import timeit

from Regressors import ThinPlateSpline
from tools import (
        grid,
        plot_2d,
        temp_diff,
        )

data_dir = '../pipeline-gridded/Data/'
results_dir = '../pipeline-gridded/Results/'
window_size_gp=5
CLIM = 2.0
stage = 'adj'

depth_idx = 1

# sub = 'All'
sub = sys.argv[1]

if sub == 'All':
    df = pkl.load(open(f'{data_dir}/HurricaneTempCovDF_Subset.pkl', 'rb'))
    block_S = pkl.load(open(f'{results_dir}/BlockCovmat_{window_size_gp}_'
                            f'{(depth_idx+1)*10:03d}.pkl', 'rb'))
    block_W = pkl.load(open(f'{results_dir}/BlockPremat_{window_size_gp}_'
                            f'{(depth_idx+1)*10:03d}.pkl', 'rb'))
else:
    df = pkl.load(open(f'{data_dir}/HurricaneTempCovDF_Subset{sub}.pkl', 'rb'))
    block_S = pkl.load(open(f'{results_dir}/BlockCovmat_{window_size_gp}_'
                            f'{(depth_idx+1)*10:03d}{sub}.pkl', 'rb'))
    block_W = pkl.load(open(f'{results_dir}/BlockPremat_{window_size_gp}_'
                            f'{(depth_idx+1)*10:03d}{sub}.pkl', 'rb'))


X = np.array(df[['standard_signed_angle', 'hurricane_dtd']]).copy()
y = temp_diff(df, stage, depth_idx)
var = df['var'].apply(lambda x: x[depth_idx]).values
W = 1 / var 


bounds_x1 = (-8, +8)
bounds_x2 = (-2, +20)
shape = (100, 400)
test_X = grid(bounds_x1, bounds_x2, *shape)

# No reweight
tps = ThinPlateSpline(lamb=0.5)
tps.fit(X, y)
preds_no_reweight = tps.predict(test_X)
plt = plot_2d(preds_no_reweight, shape, clim=CLIM)
plt.title(f'{sub}: Unweighted')
plt.savefig(f'img/Fit_0Unweighted{sub}.png', dpi=300)
plt.show()

# Diag reweight
t0 = timeit.default_timer()
tps = ThinPlateSpline(lamb=0.5)
tps.fit(X, y,
        W=W,
    )
preds, sd, mask = tps.predict(test_X, sd=True, S=var, k=2)
t_tps = timeit.default_timer() - t0
print(f'Time elapsed: {t_tps:0.2f} seconds')
plt = plot_2d(preds, shape, clim=CLIM)
plt.title(f'{sub}: Diagonal Weighted')
plt.savefig(f'img/Fit_1Diagonal{sub}.png', dpi=300)
plt.show()

plt = plot_2d(preds, shape, mask=mask, clim=CLIM)
plt.title(f'{sub}: Diagonal Weighted')
plt.savefig(f'img/SDFit_1Diagonal{sub}.png', dpi=300)
plt.show()



# Block reweight, 
t0 = timeit.default_timer()
tps = ThinPlateSpline(lamb=0.5)
tps.fit(X, y,
        W=block_W,
    )
preds, sd, mask = tps.predict(test_X, sd=True, S=block_S, k=2)
t_tps = timeit.default_timer() - t0
print(f'Time elapsed: {t_tps:0.2f} seconds')
plt = plot_2d(preds, shape, clim=CLIM)
plt.title(f'{sub}: Block Weighted')
plt.savefig(f'img/Fit_2Block{sub}.png', dpi=300)
plt.show()

plt = plot_2d(preds, shape, mask=mask, clim=CLIM)
plt.title(f'{sub}: Block Weighted')
plt.savefig(f'img/SDFit_2Block{sub}.png', dpi=300)
plt.show()


