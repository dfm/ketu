from __future__ import division

cimport cython
from libc.math cimport fabs

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef class _box_model:
    cdef public double half_duration, t0

    @cython.boundscheck(False)
    def __call__(self, np.ndarray[DTYPE_t, ndim=1] t):
        cdef unsigned int n = t.shape[0]
        cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=DTYPE)
        for i in range(n):
            if fabs(t[i] - self.t0) < self.half_duration:
                result[i] = -1.0
            else:
                result[i] = 0.0
        return result


@cython.boundscheck(False)
def compute_hypotheses(lnlikefn, np.ndarray[DTYPE_t, ndim=1] t,
                       np.ndarray[DTYPE_t, ndim=1] durations,
                       np.ndarray[DTYPE_t, ndim=2] depths,
                       np.ndarray[DTYPE_t, ndim=2] d_ivars,
                       np.ndarray[DTYPE_t, ndim=2] results):
    cdef unsigned int i, j, k
    cdef unsigned int nt = t.shape[0]
    cdef unsigned int ndepths = depths.shape[0]
    cdef unsigned int ndurations = durations.shape[0]
    cdef double dll, dep, ide
    model = _box_model()
    for i in range(nt):
        model.t0 = t[i]
        for j in range(ndurations):
            model.half_duration = 0.5 * durations[k]
            dll, dep, ide = lnlikefn(model)
            results[i, j] = dll
            depths[i, j] = dep
            d_ivars[i, j] = ide
