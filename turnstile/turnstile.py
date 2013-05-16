#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["process"]

import kplr
import numpy as np

from . import _turnstile


def process(kepid, lc_only=True):
    # Find the list of data files.
    client = kplr.API()

    # Load and untrend the data.
    time, flux, ivar = [], [], []
    for d in client.data(kepid):
        if lc_only and "slc" in d.filename:
            continue
        d.fetch()
        dataset = kplr.Dataset(d.filename, untrend=True)
        time.append(dataset.time[dataset.mask])
        flux.append(dataset.flux[dataset.mask])
        ivar.append(dataset.ivar[dataset.mask])
    time = np.concatenate(time)
    flux = np.concatenate(flux)
    ivar = np.concatenate(ivar)

    # Define the grid.
    min_period, max_period, dperiod = 213, 214, 0.2
    # min_period, max_period, dperiod = 180, 750, 0.2
    tau = 0.4

    # Fit for periods.
    periods = _turnstile.find_periods(time, flux, ivar,
                                      min_period, max_period, dperiod,
                                      tau)

    return periods
