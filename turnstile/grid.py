#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Grid"]

import transit
import numpy as np
from scipy.spatial import cKDTree

import kplr
from kplr.ld import get_quad_coeffs

from .data import LightCurve

client = kplr.API()

# Newton's constant in $R_\odot^3 M_\odot^{-1} {days}^{-2}$.
_G = 2945.4625385377644


class Grid(object):

    def __init__(self, kicid):
        self.kicid = kicid
        self.kic = client.star(kicid)
        self._data = None
        self.injections = []

    def get_data(self, ttol=0.5, force=False):
        """
        Get the list of light curve datasets associated with this grid.

        :param ttol: (optional)
            The maximum allowed time gap in days. Passed directly to
            :func:`LightCurve.autosplit`. (default: 0.5)

        :param force: (optional)
            If ``True``, force the data to be re-processed even if a cached
            version exists.

        """
        # Returned the cached value if we have one.
        if not force and self._data is not None:
            return self._data

        # Loop over the long cadence light curves from mask and pre-process
        # them all.
        datasets = []
        lcs = self.kic.get_light_curves(short_cadence=False)
        for lc in lcs:
            data = lc.read()
            datasets += LightCurve(data["TIME"], data["SAP_FLUX"],
                                   data["SAP_FLUX_ERR"],
                                   data["SAP_QUALITY"] == 0).autosplit(ttol)

        # Cache the result.
        self._data = datasets
        return self._data

    def inject_transit(self, period, rp, t0=None, b=None, teff=None, logg=None,
                       feh=None, mstar=None, rstar=None,
                       texp=kplr.EXPOSURE_TIMES[1]/86400, tol=0.1, maxdepth=3):
        # Get the KIC stellar parameters if they weren't given.
        teff = teff if teff is not None else self.kic.kic_teff
        logg = logg if logg is not None else self.kic.kic_logg
        feh = feh if feh is not None else self.kic.kic_feh
        mstar = mstar if mstar is not None else 1.0
        rstar = rstar if rstar is not None else self.kic.kic_radius
        if rstar is None:
            rstar = 1.0

        # Get the quadratic limb darkening coefficients for the KIC stellar
        # parameters.
        u1, u2 = get_quad_coeffs(teff, logg=logg, feh=feh)

        # If epoch and or impact parameter were not given, randomly sample
        # them.
        if t0 is None:
            t0 = period*np.random.rand()
        if b is None:
            b = (1+rp/rstar)*np.random.rand()

        # Compute the semi-major axis.
        a = (_G*period*period*mstar/(4*np.pi*np.pi)) ** (1./3)

        # Compute the inclination angle given an impact parameter.
        ix = np.degrees(np.arctan2(b, a / rstar))

        # Inject the transit into the datasets.
        for lc in self.get_data():
            lc.flux *= transit.ldlc_kepler(lc.time, u1, u2, mstar, rstar, [0],
                                           [0], [rp], [a], [t0], [0], [0],
                                           [ix], [0], texp, tol, maxdepth)

        # Save the injected transit specs.
        self.injections.append(dict(
            period=period,
            a=a,
            rp=rp,
            t0=t0,
            b=b,
            ix=ix,
            teff=teff,
            logg=logg,
            feh=feh,
            mstar=mstar,
            rstar=rstar,
            u1=u1,
            u2=u2,
        ))
        return self.injections[-1]

    def optimize_hyperparams(self, p0=None, N=3):
        pars = []
        for lc in self.get_data():
            pars.append(lc.optimize_hyperparams(p0=p0, N=N))
            print(pars[-1])
        return pars

    def compute_hypotheses(self, depths, durations):
        # Make sure that the durations and depths are iterable.
        self.durations = np.atleast_1d(durations)
        self.depths = np.atleast_1d(depths)

        # Pre-allocate a stub for the hypotheses to be appended to.
        self.times = np.empty(0)
        self.delta_lls = np.empty((0, len(depths), len(durations)))

        # Loop over datasets and compute the grid of hypotheses for each of
        # those.
        for lc in self.get_data():
            t, dll = lc.compute_hypotheses(depths, durations)
            self.times = np.append(self.times, t)
            self.delta_lls = np.concatenate((self.delta_lls, dll), axis=0)
            print(len(self.times))

        # Build a KDTree index.
        self.index = cKDTree(np.atleast_2d(self.times).T)
# print(tree.query(np.array([[15.0], [40.0], ]), distance_upper_bound=0.1))
