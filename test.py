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
    #                                                   0.1)
    # periods, depths, epochs, chi2 = _turnstile.find_periods(time, flux, ivar,
    #                                                         100, 300,
    #                                                         0.05)
    periods, depths, epochs, chi2 = _turnstile.find_periods(time, flux, ivar,
                                                            190, 210,
                                                            0.05)
    # periods, depths, epochs, chi2 = _turnstile.find_periods(time, flux, ivar,
    #                                                         period - 10,
    #                                                         period + 10,
    #                                                         0.1)
    print("Took {0} minutes.".format((timer.time() - strt) / 60.))

    pl.clf()
    pl.plot(periods, chi2, "k")
    pl.gca().axvline(122.3874, alpha=0.3)
    pl.gca().axvline(267.291, alpha=0.3)
    pl.xlabel("Period [days]")
    pl.ylabel(r"$\chi^2$")
    pl.gca().set_ylim(chi2.min(), 0)
    pl.savefig("blah.png")

    dt = time.max() - time.min()
    pl.figure(figsize=(16, 16))
    for i, ind in enumerate(np.argsort(chi2)[:1]):
        pl.clf()
        for j, f in enumerate([1, 2, 3, 5, 7, 11, 13, 17, 19]):
            demo_period = periods[ind] / f
            demo_epoch = epochs[ind]

            print(f,
                  _turnstile.compute_chi2(time, flux, ivar,
                                          demo_period, demo_epoch))

            pl.subplot(3, 3, j + 1)
            pl.plot((time - demo_epoch + 0.5 * demo_period) % demo_period
                    - 0.5 * demo_period,
                    flux + 0.001 * np.array((time - demo_epoch)
                                            / demo_period, dtype=int),
                    ".k", ms=1)
            pl.title("{0} days".format(demo_period))
            pl.xlim(-2, 2)
            pl.savefig("figs/sup-{0:03d}.png".format(i))
