from __future__ import division

cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport ceil, round, log, INFINITY, sqrt, M_PI

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef double ln2pi = log(2.0 * M_PI)

@cython.boundscheck(False)
cdef int evaluate_single(int min_transits, double alpha,
                         double period, double t0,
                         double tmin, double tmax, double time_spacing,
                         int nduration, double* dll_1d,
                         const double* depth_1d, const double* depth_ivar_1d,
                         double* phic_variable, double* phic_same,
                         double* depth_2d, double* depth_ivar_2d,
                         int* inds):
    cdef double t
    cdef int k, ind, nind
    nind = 0

    # Initialize the results array at zero.
    for k in range(nduration):
        phic_variable[k] = 0.0
        phic_same[k] = 0.0
        depth_2d[k] = 0.0
        depth_ivar_2d[k] = 0.0

    # For simplicity, we'll make the outer loop be the duration.
    for k in range(nduration):
        # Start from the earliest possible transit time.
        t = tmin + t0
        nind, ind = 0, 0

        # Loop through time checking every possible transit time.
        while t <= tmax - time_spacing:
            # Compute the nearest 1-d index for this transit time.
            ind = int(round((t - tmin) / time_spacing)) * nduration + k

            # Update the next time to the following transit.
            t += period

            # Skip this transit if there is no depth measured.
            if depth_ivar_1d[ind] <= 0.0:
                continue

            # Remember the fact that we put a transit at this
            # index. We'll need to use this to compute the PHIC of
            # the single depth model.
            inds[nind] = ind
            nind += 1

            # Accumulate the likelihood for the variable depth
            # model. Note: the single depth model is computed
            # below using this result.

            # First incorporate the delta log-likelihood for this
            # transit time.
            phic_variable[k] += dll_1d[ind]
            phic_same[k] += dll_1d[ind]

            # And then the uncertainty in the depth measurement.
            # phic_variable[k] += 0.5 * (ln2pi - log(depth_ivar_1d[ind]))
            phic_same[k] -= 0.5 * (ln2pi - log(depth_ivar_1d[ind]))
            phic_same[k] -= 0.5 * depth_1d[ind] * depth_1d[ind] * depth_ivar_1d[ind]

            # Here, we'll accumulate the weighted depth
            # measurement for the single depth model.
            depth_2d[k] += depth_1d[ind] * depth_ivar_1d[ind]
            depth_ivar_2d[k] += depth_ivar_1d[ind]

        # Now, use the results of the previous computation to evaluate the
        # single depth PHIC.
        phic_same[k] -= 0.5 * alpha
        phic_variable[k] -= 0.5 * nind * alpha

        # If there was any measurement for the depth, update the
        # single depth model.
        if depth_ivar_2d[k] > 0 and nind >= min_transits:
            depth_2d[k] /= depth_ivar_2d[k]

            phic_same[k] += 0.5 * depth_2d[k] * depth_2d[k] * depth_ivar_2d[k]
            # phic_same[k] += 0.5 * (ln2pi - log(depth_ivar_2d[k]))

        else:
            depth_2d[k] = 0.0
            depth_ivar_2d[k] = 0.0
            phic_same[k] = -INFINITY

    return nind


@cython.boundscheck(False)
def grid_search(int min_transits, double alpha,
                double tmin, double tmax, double time_spacing,
                np.ndarray[DTYPE_t, ndim=2] depth_1d,
                np.ndarray[DTYPE_t, ndim=2] depth_ivar_1d,
                np.ndarray[DTYPE_t, ndim=2] dll_1d,
                np.ndarray[DTYPE_t, ndim=1] periods,
                double dt):

    cdef double t0, period
    cdef int i, k

    # Array dimensions.
    cdef int nperiod = periods.shape[0]
    cdef int nduration = dll_1d.shape[1]

    # Allocate the output arrays.
    cdef np.ndarray[DTYPE_t, ndim=2] phic_same = \
            -np.inf + np.zeros((nperiod, nduration), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] phic_same_2 = \
            -np.inf + np.zeros((nperiod, nduration), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] phic_variable = \
            -np.inf + np.zeros((nperiod, nduration), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] depth_2d = \
            np.zeros((nperiod, nduration), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] depth_ivar_2d = \
            np.zeros((nperiod, nduration), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] phases_2d = \
            np.zeros((nperiod, nduration), dtype=DTYPE)

    # Temporary arrays.
    cdef double* phic_same_tmp = <double*>malloc(nduration * sizeof(double))
    cdef double* phic_variable_tmp = <double*>malloc(nduration * sizeof(double))
    cdef double* depth_2d_tmp = <double*>malloc(nduration * sizeof(double))
    cdef double* depth_ivar_2d_tmp = <double*>malloc(nduration * sizeof(double))

    # Pointers to the input arrays.
    cdef double* dll_1d_data = <double*>dll_1d.data
    cdef double* depth_1d_data = <double*>depth_1d.data
    cdef double* depth_ivar_1d_data = <double*>depth_ivar_1d.data

    # Workspace for the transit indices.
    cdef int nimx = int(ceil((tmax - tmin) / periods.min()))
    cdef int* inds = <int*>malloc(nimx * sizeof(int))

    # Loop over hypothesized periods.
    for i in range(nperiod):
        period = periods[i]
        t0 = 0.0

        # Loop over every possible phase for the given period.
        while 1:
            evaluate_single(min_transits, alpha, period,
                            t0, tmin, tmax, time_spacing,
                            nduration, dll_1d_data, depth_1d_data,
                            depth_ivar_1d_data,
                            phic_variable_tmp, phic_same_tmp, depth_2d_tmp,
                            depth_ivar_2d_tmp, inds)

            # Loop over durations and decide if this should be accepted.
            for k in range(nduration):
                if (depth_2d_tmp[k] > 0.0 and phic_same_tmp[k] > phic_variable_tmp[k]):
                    if phic_same_tmp[k] > phic_same[i, k]:
                        phic_same_2[i, k] = phic_same[i, k]
                        phic_same[i, k] = phic_same_tmp[k]
                        phic_variable[i, k] = phic_variable_tmp[k]
                        depth_2d[i, k] = depth_2d_tmp[k]
                        depth_ivar_2d[i, k] = depth_ivar_2d_tmp[k]
                        phases_2d[i, k] = t0
                    elif phic_same_tmp[k] > phic_same_2[i, k]:
                        phic_same_2[i, k] = phic_same_tmp[k]

            # Update the proposal time.
            t0 += dt
            if t0 >= period:
                break

    free(phic_same_tmp)
    free(phic_variable_tmp)
    free(depth_2d_tmp)
    free(depth_ivar_2d_tmp)
    free(inds)

    return (phases_2d, phic_same, phic_same_2, phic_variable,
            depth_2d, depth_ivar_2d)
