#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import os
import pyfits
import numpy as np


def load_dataset():
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data",
                      "data.fits")

    with pyfits.open(fn) as hdus:
        time = np.array(hdus[1].data["time"])
        flux = np.array(hdus[1].data["flux"])
        ivar = np.array(hdus[1].data["ferr"])

    inds = ~(np.isnan(flux) * np.isnan(time))
    return time[inds], flux[inds], ivar[inds]


def test_fold_and_bin():
    import matplotlib.pyplot as pl
    time, flux, ferr = load_dataset()
    ivar = np.zeros_like(time)
    inds = ferr > 0
    ivar = 1.0 / ferr[inds] ** 2

    dt = 0.1
    bins = np.arange(time.min(), time.max(), dt)
    vals = np.zeros_like(bins)
    ivar = np.zeros_like(bins)
    for i, t in enumerate(bins):
        mask = (time >= t) * (time < t + dt)
        if np.sum(mask):
            vals[i] = np.mean(flux[mask])
            ivar[i] = 1.0 / np.mean(ferr[mask] ** 2)

    pl.errorbar(time, flux, yerr=ferr, fmt=".k")

    inds = ivar > 0
    print(np.sum(inds), len(time))
    pl.errorbar(bins[inds], vals[inds], yerr=1.0 / np.sqrt(ivar[inds]),
                fmt=".r")

    pl.savefig("blah.png")
