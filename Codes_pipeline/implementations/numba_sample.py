import numpy as np
import timeit

from numba import njit, prange

N = 50000
m = 300
X = np.random.randn(N, m)
A = np.random.randn(m, m)

def _naive_quadform(X, A):
    k = X.shape[0]
    arr = np.zeros(k)
    for idx in prange(k):
        # Unfair, multi_dot actually slower
        # Use hand-dot to get faster than parallel due to
        # multithreading
        arr[idx] = np.linalg.multi_dot((
            X[idx, :], A, X[idx, :]
        ))
    return arr

@njit(parallel=True)
def _par_quadform(X, A):
    k = X.shape[0]
    arr = np.zeros(k)
    for idx in prange(k):
        x = X[idx, :]
        arr[idx] = np.dot(x, A.dot(x))
    return arr

t0 = timeit.default_timer()
print(_naive_quadform(X, A))
t_naive = timeit.default_timer() - t0
print(f'Time elapsed (naive): {t_naive:0.2f} seconds')

t0 = timeit.default_timer()
print(_par_quadform(X, A))
t_parallel = timeit.default_timer() - t0
print(f'Time elapsed (parallel): {t_parallel:0.2f} seconds')
