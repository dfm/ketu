#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import time as timer

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
        time.append(ds.time[ds.mask * ds.qualitymask])
        flux.append(ds.flux[ds.mask * ds.qualitymask])
        ivar.append(ds.ivar[ds.mask * ds.qualitymask])

    time = np.concatenate(time)
    flux = np.concatenate(flux)
    ivar = np.concatenate(ivar)

    pl.plot(time, flux, ".k")
    pl.savefig("sup.png")

    strt = timer.time()
    # periods, depths, epochs = _turnstile.find_periods(time, flux, ivar,
    #                                                   100, 1.5 * 365,
    #                                                   0.1, 0.3)
    periods, depths, epochs = _turnstile.find_periods(time, flux, ivar,
                                                      100, 300,
                                                      0.1, 0.3)
    # periods, depths, epochs = _turnstile.find_periods(time, flux, ivar,
    #                                                   period - 10,
    #                                                   period + 10,
    #                                                   0.1, 0.3)
    print("Took {0} minutes.".format((timer.time() - strt) / 60.))

    pl.clf()
    pl.plot(periods, depths, "k")
    # pl.gca().axvline(122.3874, alpha=0.3)
    # pl.gca().axvline(267.291, alpha=0.3)
    pl.xlabel("Period [days]")
    pl.ylabel("Transit Depth")
    pl.savefig("blah.png")

    for i, ind in enumerate(np.argsort(depths)[::-1][:300]):
        demo_period = periods[ind]
        demo_epoch = epochs[ind]

        pl.clf()
        pl.plot(time % demo_period - demo_epoch, flux, ".k")
        pl.title("{0} days".format(demo_period))
        pl.xlim(-2, 2)
        pl.savefig("figs/sup-{0:03d}.png".format(i))
