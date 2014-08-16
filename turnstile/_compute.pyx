from __future__ import division

cimport cython
from libc.math cimport fabs, fmod, floor, ceil, round, log, M_PI

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


# @cython.boundscheck(False)
def grid_search(double tmin, double tmax, double time_spacing,
                np.ndarray[DTYPE_t, ndim=2] depths,
                np.ndarray[DTYPE_t, ndim=2] d_ivars,
                np.ndarray[DTYPE_t, ndim=2] dll,
                np.ndarray[DTYPE_t, ndim=1] periods,
                double dt):
    # MAGIC
    cdef double lnn = log(60000) - log(2 * M_PI)

    cdef double t0, t, period
    cdef unsigned int i, j, k, l, strt, n
    cdef int ind

    cdef unsigned int nperiods = periods.shape[0]
    cdef unsigned int blah = dll.shape[0]
    cdef unsigned int a = dll.shape[1]
    cdef unsigned int nt
    cdef unsigned int ntmx = int(ceil(periods.max() / dt))
    cdef unsigned int nind, nimx = int(ceil((tmax - tmin) / periods.min()))

    cdef np.ndarray[DTYPE_t, ndim=3] bic1 = np.nan + np.zeros((nperiods,
                                                               ntmx, a),
                                                               dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] bic2 = np.nan + np.zeros((nperiods,
                                                               ntmx, a),
                                                               dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] dmax = np.empty(a, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] ivdmx = np.empty(a, dtype=DTYPE)
    cdef np.ndarray[np.int32_t, ndim=1] inds = np.empty(nimx, dtype=np.int32)
    cdef double norm, d

    for i in range(nperiods):
        period = periods[i]
        t0 = 0.0
        for j in range(ntmx):
            # Initialize the results array at zero.
            for k in range(a):
                bic2[i, j, k] = 0.0
                dmax[k] = 0.0
                ivdmx[k] = 0.0

            # Loop over transit times for this given period and phase.
            nind, ind, strt = 0, 0, 0
            t = fmod(t0, period) + period * floor(tmin / period)
            while t < tmax:
                ind = int(round((t - tmin) / time_spacing))
                if ind > 0:
                    for k in range(a):
                        bic2[i, j, k] += dll[ind, k]
                        dmax[k] += depths[ind, k] * d_ivars[ind, k]
                        ivdmx[k] += d_ivars[ind, k]
                        inds[nind] = ind
                        nind += 1
                t += period

            # Compute the marginalized likelihood.
            for k in range(a):
                bic2[i, j, k] = -2 * bic2[i, j, k] + (tmax - tmin) / period * lnn

                if ivdmx[k] > 0:
                    dmax[k] /= ivdmx[k]

                    # Compute the maximum likelihood value for the single
                    # depth model.
                    norm = 0.0
                    for l in range(nind):
                        ind = inds[l]
                        d = depths[ind, k] - dmax[k]
                        norm += dll[ind, k] - 0.5 * d * d * d_ivars[ind, k]

                    bic1[i, j, k] = -2 * norm + lnn
                    # results[i, j, k] = 0.5 * log(2 * M_PI / ivdmx[k]) + norm
                else:
                    bic1[i, j, k] = np.inf

            # Update the proposal time.
            t0 += dt
            if t0 >= period:
                break

    return bic1, bic2
