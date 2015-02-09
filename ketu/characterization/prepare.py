# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["prepare_characterization"]

import kplr
import transit
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as pl

import george
from george import kernels

from ..prepare import Prepare
from ..download import Download
from ..discontinuity import Discontinuity


def prepare_characterization(kicid, periods, time0s, rors, impacts,
                             es=None,
                             data_window_hw=3.0, min_data_window_hw=0.5):
    # Download and process the light curves.
    pipe = Download()
    pipe = Prepare(pipe)
    pipe = Discontinuity(pipe)
    r = pipe.query(kicid=kicid)

    # Find the data chunks that hit a transit.
    lcs = []
    for lc in r.light_curves:
        # Build the mask of times that hit transits.
        m = np.zeros_like(lc.time, dtype=bool)
        mmin = np.zeros_like(lc.time, dtype=bool)
        for p, t0 in zip(periods, time0s):
            hp = 0.5 * p
            t0 = t0 % p
            dt = np.abs((lc.time - t0 + hp) % p - hp)
            m += dt < data_window_hw
            mmin += dt < min_data_window_hw

        # Trim the dataset and set up the Gaussian Process model.
        if np.any(mmin) and np.sum(m) > 10:
            # Re-normalize the trimmed light curve.
            mu = np.median(lc.flux[m])
            lc.time = np.ascontiguousarray(lc.time[m])
            lc.flux = np.ascontiguousarray(lc.flux[m] / mu)
            lc.ferr = np.ascontiguousarray(lc.ferr[m] / mu)

            # Make sure that the light curve knows its integration time.
            lc.texp = kplr.EXPOSURE_TIMES[1] / 86400.0

            # Heuristically guess the Gaussian Process parameters.
            lc.factor = 1000.0
            amp = np.median((lc.factor * (lc.flux-1.0))**2)
            kernel = amp*kernels.Matern32Kernel(4.0)
            lc.gp = george.GP(kernel)

            # Run an initial computation of the GP.
            lc.gp.compute(lc.time, lc.ferr * lc.factor)

            # Save this light curve.
            lcs.append(lc)

    # Set up the initial system model.
    spars = r.star.huber
    star = transit.Central(mass=spars.M, radius=spars.R)
    s = transit.System(star)
    for i in range(len(periods)):
        planet = transit.Body(r=rors[i] * star.radius,
                              period=periods[i],
                              t0=time0s[i] % periods[i],
                              b=impacts[i],
                              e=0.0 if es is None else es[i])
        s.add_body(planet)

    # Approximate the stellar mass and radius measurements as log-normal.
    q = np.array(spars[["R", "E_R", "e_R"]], dtype=float)
    lnsr = (np.log(q[0]),
            1.0 / np.mean([np.log(q[0] + q[1]) - np.log(q[0]),
                           np.log(q[0]) - np.log(q[0] - q[2])]) ** 2)
    q = np.array(spars[["M", "E_M", "e_M"]], dtype=float)
    lnsm = (np.log(q[0]),
            1.0 / np.mean([np.log(q[0] + q[1]) - np.log(q[0]),
                           np.log(q[0]) - np.log(q[0] - q[2])]) ** 2)

    return ProbabilisticModel(lcs, s, lnsr, lnsm)


class ProbabilisticModel(object):

    def __init__(self, lcs, system, lnsr, lnsm):
        self.lcs = lcs
        self.system = system
        self.lnsr = lnsr
        self.lnsm = lnsm
        self.fit_star = False

    def pack(self):
        star = self.system.central
        planets = self.system.bodies

        vec = list(self.lcs[0].gp.kernel.vector)
        if self.fit_star:
            vec += [np.log(star.radius), np.log(star.mass)]
        vec += [
            star.q1,
            star.q2,
        ]
        vec += [v for p in planets for v in (
            np.log(p.r), np.log(p.period), p.t0, p.b,
            np.sqrt(p.e) * np.sin(p.pomega),
            np.sqrt(p.e) * np.cos(p.pomega)
        )]
        return np.array(vec)

    def unpack(self, pars):
        # Update the kernel.
        i = len(self.lcs[0].gp.kernel)
        for lc in self.lcs:
            lc.gp.kernel[:] = pars[:i]

        # Update the star.
        star = self.system.central
        if self.fit_star:
            star.radius, star.mass = np.exp(pars[i:i+2])
            i += 2
        star.q1, star.q2 = pars[i:i+2]
        i += 2

        # Update the planets.
        for p in self.system.bodies:
            p.r, p.period = np.exp(pars[i:i+2])
            i += 2
            p.t0, p.b = pars[i:i+2]
            i += 2
            sqesn, sqecs = pars[i:i+2]
            p.e = sqesn**2 + sqecs**2
            p.pomega = np.arctan2(sqesn, sqecs)
            i += 2

    def lnprior(self):
        lnp = 0.0

        # Apply the stellar parameter constraints.
        star = self.system.central
        if not (0 < star.q1 < 1 and 0 < star.q2 < 1):
            return -np.inf
        lnsr = np.log(star.radius)
        lnp -= 0.5 * self.lnsr[1] * (self.lnsr[0] - lnsr) ** 2
        lnsm = np.log(star.mass)
        lnp -= 0.5 * self.lnsm[1] * (self.lnsm[0] - lnsm) ** 2

        # And the planet parameters.
        for p in self.system.bodies:
            if p.b < 0.0 or not (-2 * np.pi < p.pomega < 2 * np.pi):
                return -np.inf

            # Kipping (2013)
            lnp += beta(1.12, 3.09).logpdf(p.e)

        return lnp

    def lnlike(self):
        ll = 0.0
        for lc in self.lcs:
            try:
                mu = self.system.light_curve(lc.time, texp=lc.texp)
            except RuntimeError:
                return -np.inf
            r = (lc.flux - mu) * lc.factor
            ll += lc.gp.lnlikelihood(r, quiet=True)
            if not np.isfinite(ll):
                return -np.inf
        return ll

    def lnprob(self, p):
        try:
            self.unpack(p)
        except ValueError:
            return -np.inf
        lp = self.lnprior()
        if not np.isfinite(lp):
            return -np.inf
        ll = self.lnlike()
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    def plot(self, dy=1e-2):
        fig = pl.figure()
        ax = fig.add_subplot(111)

        period = self.system.bodies[0].period
        t0 = self.system.bodies[0].t0
        for i, lc in enumerate(self.lcs):
            t = (lc.time - t0 + 0.5 * period) % period - 0.5 * period
            ax.plot(t, lc.flux + i*dy, ".k", alpha=0.5)

            mu = self.system.light_curve(lc.time, texp=lc.texp)
            r = lc.factor * (lc.flux - mu)
            pred = lc.gp.predict(r, lc.time, mean_only=True) / lc.factor
            ax.plot(t, pred + 1.0 + i*dy, "r", alpha=0.5)
            ax.plot(t, pred + mu + i*dy, "b", alpha=0.5)

        ax.axvline(0.0, color="k", alpha=0.3, lw=3)

        return fig
