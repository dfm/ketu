# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["GPLikelihood"]

import numpy as np

try:
    from emcee.autocorr import integrated_time
except ImportError:
    integrated_time = None

try:
    import george
except ImportError:
    george = None
else:
    from george.kernels import ExpSquaredKernel

from ..pipeline import Pipeline


class GPLikelihood(Pipeline):

    query_parameters = dict(
        matern=(False, False),
    )

    def __init__(self, *args, **kwargs):
        if integrated_time is None:
            raise ImportError("emcee is required for the GP model")
        if george is None:
            raise ImportError("george is required for the GP model")

        kwargs["cache"] = kwargs.pop("cache", False)
        super(GPLikelihood, self).__init__(*args, **kwargs)

    def get_result(self, query, parent_response):
        lcs = [LCWrapper(lc, matern=query["matern"])
               for lc in parent_response.light_curves]
        return dict(model_light_curves=lcs)


class LCWrapper(object):

    def __init__(self, lc, dist_factor=10.0, time_factor=0.1, matern=False):
        self.time = lc.time
        self.flux = lc.flux - 1.0
        self.ferr = lc.ferr

        # Convert to parts per thousand.
        self.flux *= 1e3
        self.ferr *= 1e3

        # Hackishly build a kernel.
        tau = np.median(np.diff(self.time)) * integrated_time(self.flux)
        tau = max(0.1, tau)  # Tau should be floored.
        amp = np.median((self.flux - np.median(self.flux))**2)
        self.kernel = amp * ExpSquaredKernel(tau ** 2)
        self.gp = george.GP(self.kernel, solver=george.HODLRSolver)
        self.gp.compute(self.time, self.ferr, seed=1234)

        # Compute the likelihood of the null model.
        self.ll0 = self.lnlike()

    def lnlike(self, model=None):
        y = self.flux
        Cf = self.gp.solver.apply_inverse(y)

        # No model is given. Just evaluate the lnlikelihood.
        if model is None:
            return -0.5 * np.dot(y, Cf)

        # A model is given, use it to do a linear fit.
        m = model(self.time)
        if m[0] != 0.0 or m[-1] != 0.0 or np.all(m == 0.0):
            return 0.0, 0.0, 0.0

        # Compute the inverse variance.
        Cm = self.gp.solver.apply_inverse(m)
        S = np.dot(m, Cm)
        if S <= 0.0:
            return 0.0, 0.0, 0.0

        # Compute the depth.
        d = np.dot(m, Cf) / S
        if not np.isfinite(d):
            return 0.0, 0.0, 0.0

        # Compute the lnlikelihood.
        dll = -0.5*np.dot(self.flux-d*m, Cf-d*Cm) - self.ll0
        if not np.isfinite(dll):
            return 0.0, 0.0, 0.0

        return dll, d, S

    def predict(self, y=None):
        if y is None:
            y = self.flux
        return self.gp.predict(y, self.time, mean_only=True)
