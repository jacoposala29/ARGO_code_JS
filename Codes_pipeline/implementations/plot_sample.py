import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl

NULL = object()

def plot_2d(preds, shape, mask=NULL, clim=2, cbar=True):
    bounds_x1 = (-8, +8)
    bounds_x2 = (-2, +20)
    minima, maxima = -clim, +clim
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=False)
    cmap = cm.bwr
    cmap.set_bad(color='gray')
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    if mask is not NULL:
        preds = preds.copy()
        preds[~mask] = np.nan
    plt.imshow(preds.reshape(*(shape[::-1])),
            origin='lower',
            cmap=cmap, norm=norm,
            extent=(*bounds_x1, *bounds_x2),
            )
    plt.gca().invert_yaxis()
    if cbar:
        plt.colorbar()
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)
    plt.xticks([-5, 0, +5])
    plt.yticks([0, 5, 10, 15, 20])
    return plt



est_dir = '../pipeline-gridded/Estimates/'
predmat = pkl.load(open(f'{est_dir}/TPS_Preds_Block_0.5_Hur.pkl', 'rb'))
shape = (100, 400)

depth_idx = 3
plot_2d(predmat[:, depth_idx], shape, cbar=False)
plt.savefig('test.pdf', bbox_inches='tight')

