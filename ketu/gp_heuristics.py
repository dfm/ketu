# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["estimate_tau", "kernel"]

import numpy as np
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
