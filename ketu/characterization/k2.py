# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["prepare_characterization"]

import os
import transit
import numpy as np
import matplotlib.pyplot as pl

from ..k2_data import K2Data
from ..k2_likelihood import K2Likelihood


def prepare_characterization(light_curve_file, basis_file,
                             periods, time0s, rors, impacts,
                             es=None):
    # Download and process the light curves.
    pipe = K2Data(cache=False)
    pipe = K2Likelihood(pipe, cache=False)
    query = dict(
        basis_file=os.path.abspath(basis_file),
        light_curve_file=os.path.abspath(light_curve_file)
    )
    r = pipe.query(**query)

    lc = r.model_light_curves[0]

    # Set up the initial system model.
    star = transit.Central()
    s = transit.System(star)
    for i in range(len(periods)):
        planet = transit.Body(r=rors[i],
                              period=periods[i],
                              t0=time0s[i] % periods[i],
                              b=impacts[i],
                              e=0.0 if es is None else es[i])
        s.add_body(planet)

    return ProbabilisticModel(lc, s)


class ProbabilisticModel(object):

    def __init__(self, lc, system):
        self.lc = lc
        self.system = system

    def pack(self):
        star = self.system.central
        planets = self.system.bodies
        vec = [np.log(star.mass), star.q1, star.q2, ]
        vec += [v for p in planets for v in (
            np.log(p.r), np.log(p.period), p.t0, p.b,
            np.sqrt(p.e) * np.sin(p.pomega),
            np.sqrt(p.e) * np.cos(p.pomega)
        )]
        return np.array(vec)

    def unpack(self, pars):
        # Update the star.
        star = self.system.central
        star.mass = np.exp(pars[0])
        star.q1, star.q2 = pars[1:3]

        # Update the planets.
        i = 3
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

        # And the planet parameters.
        for p in self.system.bodies:
            if p.b < 0.0 or not (-2 * np.pi < p.pomega < 2 * np.pi):
                return -np.inf
            if not 0.0 <= p.e < 1.0:
                return -np.inf

#             # Kipping (2013)
#             lnp += beta(0.867, 3.03).logpdf(p.e)

        return lnp

    def lnlike(self):
        # Compute the predicted light curve.
        try:
            mu = self.system.light_curve(self.lc.time, texp=self.lc.texp) - 1.0
        except RuntimeError:
            return -np.inf

        r = self.lc.flux - mu * 1e3
        bkg = self.lc.predict(r)
        return -0.5 * np.sum((r - bkg) ** 2) * self.lc.ivar

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

    def plot(self, fold_on=None):
        fig = pl.figure()
        ax = fig.add_subplot(111)

        mu = self.system.light_curve(self.lc.time, texp=self.lc.texp) - 1
        r = self.lc.flux - mu * 1e3
        bkg = self.lc.predict(r)

        if fold_on is None:
            t = self.lc.time
        else:
            p, t0 = fold_on
            t = (self.lc.time - t0 + 0.5*p) % p - 0.5*p

        ax.plot(t, self.lc.flux - bkg, ".k", alpha=0.5)
        i = np.argsort(t)
        ax.plot(t[i], mu[i] * 1e3, "g")

        if fold_on is not None:
            pl.xlim(-2.5, 2.5)

        return fig
