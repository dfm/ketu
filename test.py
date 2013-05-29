#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as pl

import kplr
from turnstile import _turnstile

if __name__ == "__main__":
    client = kplr.API()
    planet = client.planet("62e").koi
    period = planet.koi_period
    t0 = planet.koi_time0bk % period

    data = []
    time = []
    flux = []
    ivar = []
    for d in planet.data:
        if "slc" in d.filename:
            continue
        ds = kplr.Dataset(d.fetch().filename, untrend=True)
        data.append(ds)
        time.append(ds.time[ds.mask])
        flux.append(ds.flux[ds.mask])
        ivar.append(ds.ivar[ds.mask])

    time = np.concatenate(time)
    flux = np.concatenate(flux)
    ivar = np.concatenate(ivar)

    periods, depths = _turnstile.find_periods(time, flux, ivar, 90, 200,
                                              0.1, 0.3)

    pl.plot(periods, depths, "k")
    pl.gca().axvline(period, alpha=0.3)
    # pl.plot(time % (periods[np.argmax(depths)]), flux, ".k", alpha=0.3)
    # pl.xlim(40, 45)
    # pl.xlim(t0 - 1, t0 + 1)
    pl.savefig("blah.png")
