# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["estimate_tau", "kernel"]

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage.filters import gaussian_filter


def acor_fn(x):
    """Compute the autocorrelation function of a time series."""
    n = len(x)
    f = np.fft.fft(x-np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    return acf / acf[0]


def estimate_tau(t, y):
    """Estimate the correlation length of a time series."""
    dt = np.min(np.diff(t))
    tt = np.arange(t.min(), t.max(), dt)
    yy = np.interp(tt, t, y, 1)
    f = acor_fn(yy)
    fs = gaussian_filter(f, 50)
    w = dt * np.arange(len(f))
    m = np.arange(1, len(fs)-1)[(fs[1:-1] > fs[2:]) & (fs[1:-1] > fs[:-2])]
    if len(m):
        return w[m[np.argmax(fs[m])]]
    return w[-1]


def kernel(tau, t):
    """Matern-3/2 kernel function"""
    r = np.sqrt(3 * ((t[:, None] - t[None, :]) / tau) ** 2)
    return (1 + r) * np.exp(-r)


def optimize_gp_params(tau0, K_b, t, y, yerr):
    def nll(p):
        K_t = np.exp(p[0]) * kernel(np.exp(p[1]), t)
        i = np.diag_indices_from(K_t)
        K_t[i] += yerr ** 2
        factor = cho_factor(K_t + K_b)
        halflndet = np.sum(np.log(np.diag(factor[0])))
        r = 0.5*np.dot(y, cho_solve(factor, y)) + halflndet
        return r

    p0 = np.log([np.var(y), tau0])
    r = minimize(nll, p0, method="L-BFGS-B", bounds=[
        (np.var(y) * 0.1, 10.0 * np.var(y)),
        (np.log(0.1), np.log(50.0)),
    ])
    print("Optimized tau = {0}".format(np.exp(r.x[1])))
    return np.exp(r.x)
