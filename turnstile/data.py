#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["LightCurve"]

import copy
import numpy as np

import george
from george.kernels import Matern32Kernel


class LightCurve(object):
    """
    A wrapper object around a Kepler light curve. When initializing, any
    missing data are masked and then fluxes (and uncertainties) are
    normalized by the median.

    :param time:
        The timestamps in days (KBJD).

    :param flux:
        The fluxes corresponding to the timestamps in ``time``.

    :param ferr:
        The uncertainties on ``flux``.

    :param quality: (optional)
        If provided, this should be a boolean array where ``True`` indicates
        "good" measurements and ``False`` indicates measurements that should
        be removed. Any NaNs are removed automatically so you don't need to
        include the mask for that in here.

    """

    def __init__(self, time, flux, ferr, quality=None, normalize=True,
                 meta=None):
        self.meta = meta

        # Mask bad data.
        m = np.isfinite(time) * np.isfinite(flux) * np.isfinite(ferr)
        if quality is not None:
            m *= quality
        self.time = np.atleast_1d(time)[m]
        self.flux = np.atleast_1d(flux)[m]
        self.ferr = np.atleast_1d(ferr)[m]

        # Normalize by the median.
        if normalize:
            mu = np.median(self.flux)
            self.flux /= mu
            self.ferr /= mu

    def split(self, ts, normalize=True):
        """
        Split a light curve into a list of light curves at specified times.
        All other properties of the light curve are "deepcopy"-ed.

        :param ts:
            The list of times where the light curve should be split.

        :param normalize: (optional)
            By default, the resulting light curves have independently
            normalized fluxes and uncertainties. Set ``normalize=False`` to
            retain the original values.

        """
        ts = np.concatenate([[-np.inf], np.sort(np.atleast_1d(ts)), [np.inf]])
        datasets = []
        for i, t0 in enumerate(ts[:-1]):
            m = (np.isfinite(self.time) * (self.time >= t0)
                 * (self.time < ts[i + 1]))
            if np.any(m):
                ds = copy.deepcopy(self)
                ds.time = ds.time[m]
                ds.flux = ds.flux[m]
                ds.ferr = ds.ferr[m]
                if normalize:
                    mu = np.median(ds.flux)
                    ds.flux /= mu
                    ds.ferr /= mu
                datasets.append(ds)

        return datasets

    def autosplit(self, ttol, max_length=None):
        """
        Automatically split a light curve at any time gaps longer than a fixed
        tolerance.

        :param ttol:
            The maximum allowed gap length in days (the same units as the
            ``time`` object).

        :param max_length: (optional)
            If provided this will spilt the resulting datasets in half until
            they are all shorter than ``max_length``. This is probably only
            useful for testing.

        """
        dt = self.time[1:] - self.time[:-1]
        m = dt > ttol
        ts = 0.5 * (self.time[1:][m] + self.time[:-1][m])
        datasets = self.split(ts)
        if max_length is not None:
            while any([len(d.time) > max_length for d in datasets]):
                datasets = [[d] if len(d.time) <= max_length
                            else d.split([d.time[int(0.5*len(d.time))]])
                            for d in datasets]
                datasets = [d for ds in datasets for d in ds]

        return datasets

    def median_detrend(self, dt=4.):
        """
        "De-trend" the light curve using a running windowed median.

        :param dt: (optional)
            The width of the median window in days. (default: 4)

        """
        x, y = np.atleast_1d(self.time), np.atleast_1d(self.flux)
        assert len(x) == len(y)
        r = np.empty(len(y))
        for i, t in enumerate(x):
            inds = np.abs(x-t) < 0.5 * dt
            r[i] = np.median(y[inds])
        self.flux /= r
        self.ferr /= r
        return r
