#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import os
import pyfits
import numpy as np
from . import _turnstile


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
    inds = (ferr > 0) * (~np.isnan(flux))
    ivar = 1.0 / ferr[inds] ** 2
    time = time[inds]
    flux = flux[inds]
    ferr = ferr[inds]

    period = 15
    folded = _turnstile.find_periods(time, flux, ivar, period, 2.18, 0.1, 0.1)

    pl.errorbar(time % period, flux, yerr=ferr, fmt=".k", alpha=0.3)
    pl.errorbar(folded[0], folded[1], yerr=1.0 / np.sqrt(folded[2]), fmt=".r")

    pl.savefig("blah.png")
