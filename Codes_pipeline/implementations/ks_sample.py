import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl

from Regressors import KernelSmoother
from tools import (
        grid,
        plot_2d,
        temp_diff,
        )

df = pkl.load(open('../pipeline-gridded/Data/HurricaneAdjRawTempDF.pkl', 'rb'))
depth_idx = 1
stage = 'adj'


X = np.array(df[['standard_signed_angle', 'hurricane_dtd']]).copy()
y = temp_diff(df, stage, depth_idx)

bounds_x1 = (-8, +8)
bounds_x2 = (-2, +20)
shape = (100, 400)
test_X = grid(bounds_x1, bounds_x2, *shape).copy()

ks = KernelSmoother(h=0.2)
ks.fit(X, y)
preds = ks.predict(test_X)


plt = plot_2d(preds, shape)
plt.show()

