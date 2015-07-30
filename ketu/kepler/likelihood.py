# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["GPLikelihood"]

import numpy as np

try:
    import george
except ImportError:
    george = None
else:
    from george.kernels import Matern32Kernel

from ..pipeline import Pipeline
from ..gp_heuristics import estimate_tau


class GPLikelihood(Pipeline):

    query_parameters = dict(
        tau_frac=(0.25, False),
    )

    def __init__(self, *args, **kwargs):
        if george is None:
            raise ImportError("george is required for the GP model")
        kwargs["cache"] = kwargs.pop("cache", False)
        super(GPLikelihood, self).__init__(*args, **kwargs)

    def get_result(self, query, parent_response):
        lcs = [LCWrapper(lc, tau_frac=query["tau_frac"])
               for lc in parent_response.light_curves]
        return dict(model_light_curves=lcs)


class LCWrapper(object):

    def __init__(self, lc, dist_factor=10.0, time_factor=0.1, tau_frac=0.25):
        self.time = lc.time
        mu = np.median(lc.flux)
        self.flux = lc.flux / mu - 1.0
        self.ferr = lc.ferr / mu

        # Convert to parts per thousand.
        self.flux *= 1e3
        self.ferr *= 1e3

        # Estimate the kernel parameters.
        tau = tau_frac * estimate_tau(self.time, self.flux)
        self.kernel = np.var(self.flux) * Matern32Kernel(tau ** 2)
        self.gp = george.GP(self.kernel, solver=george.HODLRSolver)
        self.K_0 = self.gp.get_matrix(self.time)
        self.gp.compute(self.time, self.ferr, seed=1234)
        self.alpha = self.gp.solver.apply_inverse(self.flux)

        # Compute the likelihood of the null model.
        self.ll0 = self.lnlike()

    # def lnlike(self, model=None):
    #     # No model is given. Just evaluate the lnlikelihood.
    #     if model is None:
    #         return -0.5 * np.dot(self.flux, self.alpha)

    #     # A model is given, use it to do a linear fit.
    #     m = model(self.time)
    #     if m[0] != 0.0 or m[-1] != 0.0 or np.all(m == 0.0):
    #         return 0.0, 0.0, 0.0

    #     # Compute the inverse variance.
    #     Cm = self.gp.solver.apply_inverse(m)
    #     S = np.dot(m, Cm)
    #     if S <= 0.0:
    #         return 0.0, 0.0, 0.0

    #     # Compute the depth.
    #     d = np.dot(m, Cf) / S
    #     if not np.isfinite(d):
    #         return 0.0, 0.0, 0.0

    #     # Compute the lnlikelihood.
    #     dll = -0.5*np.dot(self.flux-d*m, Cf-d*Cm) - self.ll0
    #     if not np.isfinite(dll):
    #         return 0.0, 0.0, 0.0

    #     return dll, d, S

    # def predict(self, y=None):
    #     if y is None:
    #         y = self.flux
    #     return self.gp.predict(y, self.time, mean_only=True)

    def lnlike_eval(self, y):
        return -0.5 * np.dot(y, self.gp.solver.apply_inverse(y))

    def lnlike(self, model=None):
        if model is None:
            return -0.5 * np.dot(self.flux, self.alpha)

        # Evaluate the transit model.
        m = model(self.time)
        if m[0] != 0.0 or m[-1] != 0.0 or np.all(m == 0.0):
            return 0.0, 0.0, 0.0

        Km = self.gp.solver.apply_inverse(m)
        Ky = self.alpha
        ivar = np.dot(m, Km)
        depth = np.dot(m, Ky) / ivar
        r = self.flux - m*depth
        ll = -0.5 * np.dot(r, Ky - depth * Km)
        return ll - self.ll0, depth, ivar

    def predict(self, y=None):
        if y is None:
            y = self.flux
        return np.dot(self.K_0, self.gp.solver.apply_inverse(y))
