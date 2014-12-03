# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Discontinuity"]

import numpy as np

from .pipeline import Pipeline
from .prepare import LightCurve


class Discontinuity(Pipeline):

    query_parameters = dict(
        discont_window=(51, False),
        discont_duration=(0.4, False),
        discont_min_sig=(75., False),
        discont_min_fact=(0.5, False),
        discont_min_dt=(1.0, False),
        discont_min_size=(20, False),
    )

    def get_result(self, query, parent_response):
        lcs = parent_response.light_curves

        # Parameters.
        N = query["discont_window"]
        duration = query["discont_duration"]
        min_dis_sig = query["discont_min_sig"]
        min_dis_fact = query["discont_min_fact"]
        min_dis_dt = query["discont_min_dt"]
        min_dis_size = query["discont_min_size"]

        # Pre-allocate some shit.
        t0 = N // 2
        x = np.arange(N)
        A = np.vander(x, 2)

        lc_out = []
        for k, lc in enumerate(lcs):
            # Compute the typical time spacing in the LC.
            dt = int(0.5 * duration / np.median(np.diff(lc.time)))

            # The step function hypothesis.
            model1 = np.ones(N)
            model1[t0:] = -1.0

            # The transit hypothesis.
            model2 = np.zeros(N)
            model2[t0-dt:t0+dt] = -1.0

            # Initialize the work arrays.
            chi2 = np.empty((len(lc.time) - N, 3))

            # Loop over each time and compare the hypotheses.
            for i in range(len(lc.time) - N):
                y = np.array(lc.flux[i:i+N])
                ivar = 1. / np.array(lc.ferr[i:i+N]) ** 2

                # Loop over the different models, do the fit, and compute the
                # chi^2.
                for j, model in enumerate((None, model1, model2)):
                    if model is not None:
                        A1 = np.hstack((A, np.atleast_2d(model).T))
                    else:
                        A1 = np.array(A)
                    ATA = np.dot(A1.T, A1 * ivar[:, None])
                    w = np.linalg.solve(ATA, np.dot(A1.T, y * ivar))
                    pred = np.dot(A1, w)
                    chi2[i, j] = np.sum((pred - y) ** 2 * ivar)

            # Detect the peaks.
            z = chi2[:, 2] - chi2[:, 1]
            p1 = (z[1:-1] > z[:-2]) * (z[1:-1] > z[2:])
            p0 = p1 * (z[1:-1] > min_dis_sig)

            # Remove peaks with other nearby higher peaks.
            m = z[p0][:, None] - z[p0][None, :] > min_dis_fact * z[p0][:, None]

            # Remove the nearby peaks.
            t = lc.time[t0:t0+len(z)]
            m += np.abs(t[p0][:, None] - t[p0][None, :]) > min_dis_dt
            m[np.diag_indices_from(m)] = True
            m = np.all(m, axis=1)
            peak_inds = np.arange(1, len(z)-1)[p0][m]

            # Split on the peaks.
            peak_inds = np.concatenate(([0], peak_inds + t0, [len(lc.time)]))
            for i in range(len(peak_inds) - 1):
                m = np.arange(peak_inds[i], peak_inds[i+1])
                if len(m) < min_dis_size:
                    continue
                lc_out.append(
                    LightCurve(lc.time[m], lc.flux[m], lc.ferr[m],
                               np.zeros_like(lc.time[m], dtype=int),
                               (p[m] for p in lc.predictors))
                )

        return dict(light_curves=lc_out)
