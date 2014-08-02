from __future__ import division

cimport cython
from libc.math cimport fabs, fmod, floor, ceil

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef class _box_model:
    cdef public double depth, half_duration, t0

    @cython.boundscheck(False)
    def __call__(self, np.ndarray[DTYPE_t, ndim=1] t):
        cdef unsigned int n = t.shape[0]
        cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=DTYPE)
        for i in range(n):
            if fabs(t[i] - self.t0) < self.half_duration:
                result[i] = 1.0 - self.depth
            else:
                result[i] = 1.0
        return result


@cython.boundscheck(False)
def compute_hypotheses(lnlikefn, double ll0, np.ndarray[DTYPE_t, ndim=1] t,
                       np.ndarray[DTYPE_t, ndim=1] depths,
                       np.ndarray[DTYPE_t, ndim=1] durations,
                       np.ndarray[DTYPE_t, ndim=3] results):
    cdef unsigned int i, j, k
    cdef unsigned int nt = t.shape[0]
    cdef unsigned int ndepths = depths.shape[0]
    cdef unsigned int ndurations = durations.shape[0]
    model = _box_model()
    for i in range(nt):
        model.t0 = t[i]
        for j in range(ndepths):
            model.depth = depths[j]
            for k in range(ndurations):
                model.half_duration = 0.5 * durations[k]
                results[i, j, k] = lnlikefn(model) - ll0


@cython.boundscheck(False)
cdef int look_up_time(unsigned int strt, double t0,
                      np.ndarray[DTYPE_t, ndim=1] times,
                      double tol):
    cdef double d1, d2
    cdef unsigned int i
    cdef unsigned int n = times.shape[0]
    for i in range(strt+1, n):
        if times[i] >= t0:
            d1 = times[i] - t0
            d2 = t0 - times[i-1]
            if d1 > tol and d2 > tol:
                return -1
            if d1 < d2:
                return i
            return i-1
    return -1


@cython.boundscheck(False)
def grid_search(np.ndarray[DTYPE_t, ndim=1] times,
                np.ndarray[DTYPE_t, ndim=3] dll,
                np.ndarray[DTYPE_t, ndim=1] periods,
                double dt, double tmin, double tmax,
                double tol):
    cdef double t0, t, period
    cdef unsigned int i, j, k, l, strt, n
    cdef int ind

    cdef unsigned int nperiods = periods.shape[0]
    cdef unsigned int a = dll.shape[1]
    cdef unsigned int b = dll.shape[2]
    cdef unsigned int nt
    cdef unsigned int ntmx = int(ceil(periods.max() / dt))

    cdef np.ndarray[DTYPE_t, ndim=4] results = np.zeros((nperiods,
                                                         ntmx,
                                                         a, b), dtype=DTYPE)
    for i in range(nperiods):
        period = periods[i]
        t0 = 0.0
        for j in range(ntmx):
            # Loop over transit times for this given period and phase.
            ind, strt = 0, 0

            t = fmod(t0, period) + period * floor(tmin / period)
            nt = int(ceil((tmax - tmin) / period))
            for n in range(nt):
                ind = look_up_time(strt, t, times, tol)
                t += period
                if ind > 0:
                    strt = ind
                    for k in range(a):
                        for l in range(b):
                            results[i, j, k, l] += dll[ind, k, l]

            # Update the proposal time.
            t0 += dt
            if t0 >= period:
                break

    return results
