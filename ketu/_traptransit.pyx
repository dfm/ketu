from __future__ import division

cimport cython

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
def traptransit(np.ndarray[DTYPE_t] ts, np.ndarray[DTYPE_t] pars,
                np.ndarray[DTYPE_t] fs):
    """
    pars = [T,delta,T_over_tau,tc]

    full duration, depth, full duration / ingress time, center time

    """
    cdef DTYPE_t t1 = pars[3] - pars[0]/2.
    cdef DTYPE_t t2 = pars[3] - pars[0]/2. + pars[0]/pars[2]
    cdef DTYPE_t t3 = pars[3] + pars[0]/2. - pars[0]/pars[2]
    cdef DTYPE_t t4 = pars[3] + pars[0]/2.
    cdef long npts = len(ts)
    cdef unsigned int i
    for i in range(npts):
        if (ts[i]  > t1) and (ts[i] < t2):
            fs[i] *= 1-pars[1]*pars[2]/pars[0]*(ts[i] - t1)
        elif (ts[i] > t2) and (ts[i] < t3):
            fs[i] *= 1-pars[1]
        elif (ts[i] > t3) and (ts[i] < t4):
            fs[i] *= 1-pars[1] + pars[1]*pars[2]/pars[0]*(ts[i]-t3)


@cython.boundscheck(False)
def periodic_traptransit(np.ndarray[DTYPE_t] ts, np.ndarray[DTYPE_t] pars):
    """
    pars = [T,delta,T_over_tau,period,t0]

    full duration, depth, full duration / ingress time, center time

    """
    cdef double period = pars[3]
    cdef double t0 = pars[4]
    cdef long npts = len(ts)
    cdef np.ndarray[DTYPE_t] fs = np.ones(npts, dtype=float)
    cdef double tmin = ts.min()
    cdef double tmax = ts.max()
    cdef np.ndarray[DTYPE_t] tt = np.array(ts) - tmin
    cdef double tc = (t0 - tmin) % period
    cdef np.ndarray[DTYPE_t] p = np.array(pars[:-1])
    while tc < tmax:
        p[3] = tc
        traptransit(tt, p, fs)
        tc += period
    return fs
