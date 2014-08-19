from __future__ import division

cimport cython
from libc.math cimport exp

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def compute_kernel_matrix(double alpha, double tau,
                          np.ndarray[DTYPE_t, ndim=2] x):
    cdef double d, v, itau = -0.5 / tau
    cdef unsigned int n, ndim, i, j, k
    n = x.shape[0]
    ndim = x.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=2] K = np.empty((n, n), dtype=DTYPE)

    for i in range(n):
        K[i, i] = alpha
        for j in range(i+1, n):
            v = 0.0
            for k in range(ndim):
                d = x[i, k] - x[j, k]
                v += d*d
            v = alpha * exp(itau * v)
            K[i, j] = v
            K[j, i] = v

    return K
