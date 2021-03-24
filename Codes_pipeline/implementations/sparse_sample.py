import numpy as np
import scipy.sparse as sps

x1 = np.random.randn(2, 2)
x2 = np.random.randn(3, 3)
x = [x1, x2]
X = sps.block_diag(x)
y = np.arange(5)
X.dot(y)

np.dot(X.toarray(), y)
