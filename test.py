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
    # planet = client.koi("2311.01")
    # planet = client.planet("62e").koi
    # period = planet.koi_period
    # t0 = planet.koi_time0bk % period

    star = client.planet("62e").koi
    # star = client.star("10124866")
    # star = client.star("5108214")

    data = []
    time = []
    flux = []
    ivar = []
    for d in star.data:
        if "slc" in d.filename:
            continue
        ds = kplr.Dataset(d.fetch().filename, untrend=True,
                          fill_times=1, dt=1)
        data.append(ds)
        time.append(ds.time[ds.mask * ds.qualitymask])
        flux.append(ds.flux[ds.mask * ds.qualitymask])
        ivar.append(ds.ivar[ds.mask * ds.qualitymask])

    time = np.concatenate(time)
    flux = np.concatenate(flux)
    ivar = np.concatenate(ivar)

    pl.plot(time, flux, ".k")
    pl.savefig("sup.png")

    pl.clf()

    # pl.plot((time - t0 + 0.5 * period) % period - 0.5 * period, flux, ".k")
    # pl.xlim(-1, 1)
    # pl.savefig("sup-folded.png")

    strt = timer.time()
    # periods, depths, epochs = _turnstile.find_periods(time, flux, ivar,
    #                                                   100, 1.5 * 365,
    #                                                   0.1)
    periods, depths, epochs, chi2 = _turnstile.find_periods(time, flux, ivar,
                                                            100, 300,
                                                            0.05)
    # periods, depths, epochs, chi2 = _turnstile.find_periods(time, flux, ivar,
    #                                                         190, 210,
    #                                                         0.05)
    # periods, depths, epochs, chi2 = _turnstile.find_periods(time, flux, ivar,
    #                                                         period - 10,
    #                                                         period + 10,
    #                                                         0.1)
    # periods, depths, epochs, chi2 = _turnstile.find_periods(time, flux, ivar,
    #                                                         59,
    #                                                         63,
    #                                                         0.1)
    print("Took {0} minutes.".format((timer.time() - strt) / 60.))

    pl.clf()
    pl.plot(periods, chi2, "k")
    # pl.gca().axvline(122.3874, alpha=0.3)
    # pl.gca().axvline(267.291, alpha=0.3)
    # pl.gca().axvline(191.8857, alpha=0.3)
    pl.xlabel("Period [days]")
    pl.ylabel(r"$\chi^2$")
    pl.gca().set_ylim(chi2.min(), 0)
    pl.savefig("blah.png")

    dt = time.max() - time.min()
    pl.figure(figsize=(20, 16))
    for i, ind in enumerate(np.argsort(chi2)[:100]):
        print("\nPeriod: {0}".format(periods[ind]))
        pl.clf()
        chi2_min = None
        duration0 = 0.5 * np.exp(0.44 * np.log(periods[ind]) - 2.97)
        for j, f in enumerate([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]):
            demo_period = periods[ind] / f
            demo_epoch = epochs[ind]

            duration = 0.5 * np.exp(0.44 * np.log(demo_period) - 2.97)

            c2 = np.min([_turnstile.compute_chi2(time, flux, ivar,
                                                 demo_period,
                                                 demo_epoch + off)
                         for off in np.arange(0, duration0, 0.5 * duration)])
            print(j, demo_period, c2)
            if chi2_min is None:
                chi2_min = c2
            elif c2 < chi2_min:
                c2 = None
                break

            pl.subplot(3, 4, j + 1)
            pl.plot((time - demo_epoch + 0.5 * demo_period) % demo_period
                    - 0.5 * demo_period + 0.5 * duration0 - 0.5 * duration,
                    flux + 0.0 * np.array((time - demo_epoch)
                                          / demo_period, dtype=int),
                    ".k", ms=1)
            pl.title("{0} days".format(demo_period))
            pl.xlim(-2, 2)

        if c2 is not None:
            print("made it.")
            pl.savefig("figs/sup-{0:03d}.png".format(i))
        else:
            print("didn't.")
