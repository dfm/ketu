from __future__ import division

cimport cython
from libc.math cimport fabs, fmod, floor, ceil, round, log, M_PI

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False)
def grid_search(double alpha,
                double tmin, double tmax, double time_spacing,
                np.ndarray[DTYPE_t, ndim=2] depth_1d,
                np.ndarray[DTYPE_t, ndim=2] depth_ivar_1d,
                np.ndarray[DTYPE_t, ndim=2] dll_1d,
                np.ndarray[DTYPE_t, ndim=1] periods,
                double dt):
    # MAGIC!
    cdef double CONST = -0.5*log(2*M_PI)

    cdef double t0, t, period, d
    cdef unsigned int i, j, k, l, ind

    # Array dimensions.
    cdef unsigned int nperiod = periods.shape[0]
    cdef unsigned int nduration = dll_1d.shape[1]

    # The maximum number of phase points.
    cdef unsigned int ntmx = int(ceil(periods.max() / dt))

    # The maximum number of transits.
    cdef unsigned int nind, nimx = int(ceil((tmax - tmin) / periods.min()))

    # Allocate the output arrays.
    cdef np.ndarray[DTYPE_t, ndim=3] phic_same = \
        np.nan + np.zeros((nperiod, ntmx, nduration), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] phic_variable = \
        np.nan + np.zeros((nperiod, ntmx, nduration), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] depth_2d = \
        np.nan + np.zeros((nperiod, ntmx, nduration), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] depth_ivar_2d = \
        np.nan + np.zeros((nperiod, ntmx, nduration), dtype=DTYPE)

    # This array will hold the transit indices for each hypothesis.
    cdef np.ndarray[np.int32_t, ndim=1] inds = np.empty(nimx, dtype=np.int32)

    # Loop over hypothesized periods.
    for i in range(nperiod):
        period = periods[i]
        t0 = 0.0

        # Loop over every possible phase for the given period.
        for j in range(ntmx):
            # Initialize the results array at zero.
            for k in range(nduration):
                phic_variable[i, j, k] = 0.0
                depth_2d[i, j, k] = 0.0
                depth_ivar_2d[i, j, k] = 0.0

            # Search through transit times for this given period and phase.
            # We'll start from the earliest possible transit time (this will
            # often be before the beginning of the time series.
            t = tmin + t0 # fmod(t0, period) + period * floor(tmin / period)
            nind, ind = 0, 0
            while t <= tmax - time_spacing:
                # Compute the nearest 1-d index for this transit time.
                ind = int(round((t - tmin) / time_spacing))

                # Remember the fact that we put a transit at this
                # index. We'll need to use this to compute the PHIC of
                # the single depth model.
                inds[nind] = ind
                nind += 1

                for k in range(nduration):
                    # Accumulate the likelihood for the variable depth
                    # model. Note: the single depth model is computed
                    # below using this result.

                    if depth_ivar_1d[ind, k] <= 0.0:
                        continue

                    # First incorporate the delta log-likelihood for this
                    # transit time.
                    phic_variable[i, j, k] += dll_1d[ind, k]

                    # And then the uncertainty in the depth measurement.
                    phic_variable[i, j, k] += 0.5*log(depth_ivar_1d[ind, k]) + CONST

                    # Here, we'll accumulate the weighted depth
                    # measurement for the single depth model.
                    depth_2d[i, j, k] += depth_1d[ind, k] * depth_ivar_1d[ind, k]
                    depth_ivar_2d[i, j, k] += depth_ivar_1d[ind, k]

                # Go to the next transit.
                t += period

            # Now, use the results of the previous computation to evaluate the
            # single depth PHIC.
            for k in range(nduration):
                # Penalize the PHICs for the number of free parameters.
                phic_same[i, j, k] = phic_variable[i, j, k] - 0.5 * alpha
                phic_variable[i, j, k] -= 0.5 * nind * alpha

                # If there was any measurement for the depth, update the
                # single depth model.
                if depth_ivar_2d[i, j, k] > 0:
                    depth_2d[i, j, k] /= depth_ivar_2d[i, j, k]

                    # Loop over the saved list of transit times and evaluate
                    # the depth measurement at the maximum likelihood location.
                    for l in range(nind):
                        ind = inds[l]
                        if depth_ivar_1d[ind, k] <= 0.0:
                            continue
                        d = depth_1d[ind, k] - depth_2d[i, j, k]
                        phic_same[i, j, k] -= 0.5*d*d*depth_ivar_1d[ind, k]
                else:
                    depth_2d[i, j, k] = 0.0
                    depth_ivar_2d[i, j, k] = 0.0
                    phic_same[i, j, k] = -np.inf

            # Update the proposal time.
            t0 += dt
            if t0 >= period:
                break

    return phic_same, phic_variable, depth_2d, depth_ivar_2d
