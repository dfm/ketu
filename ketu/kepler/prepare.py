# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Prepare"]

import numpy as np
try:
    from itertools import izip
except ImportError:
    izip = zip

from ..pipeline import Pipeline


class Prepare(Pipeline):

    query_parameters = {
        "split_tol": (20, False),
        "min_chunk_size": (0, False),
    }

    def get_result(self, query, parent_response):
        split_tol = query["split_tol"]
        min_chunk_size = query["min_chunk_size"]

        chunks = []
        for lc, plcs in izip(parent_response.target_datasets,
                             parent_response.predictor_datasets):
            chunks += prepare_light_curve(lc, plcs, tol=split_tol,
                                          min_length=min_chunk_size)

        if not len(chunks):
            raise ValueError("No light curves were retained after Prepare")

        return dict(light_curves=chunks)


def prepare_light_curve(lc, plcs, tol=20, min_length=100):
    data = lc.read(columns=["TIME", "SAP_FLUX", "SAP_FLUX_ERR", "SAP_QUALITY"])
    time = data["TIME"]
    flux = data["SAP_FLUX"]
    ferr = data["SAP_FLUX_ERR"]
    qual = data["SAP_QUALITY"]

    # Loop over the time array and break it into "chunks" when there is "a
    # sufficiently long gap" with no data.
    count, current, chunks = 0, [], []
    for i, t in enumerate(time):
        if np.isnan(t):
            count += 1
        else:
            if count > tol or (qual[i] & (1)) != 0:
                chunks.append(list(current))
                current = []
                count = 0
            current.append(i)
    if len(current):
        chunks.append(current)

    # Loop over the predictors and read the data.
    predictors = [l.read(columns=["SAP_FLUX"])["SAP_FLUX"] for l in plcs]

    # Loop over the chunks and construct the output.
    light_curves = []
    for chunk in chunks:
        if len(chunk) < min_length:
            continue
        lc = LightCurve(time[chunk], flux[chunk], ferr[chunk], qual[chunk],
                        (p[chunk] for p in predictors))
        if len(lc.time) < min_length:
            continue
        light_curves.append(lc)

    return light_curves


class LightCurve(object):

    def __init__(self, time, flux, ferr, quality, predictors):
        # Mask missing data in the target light curve.
        m = np.isfinite(time)*np.isfinite(flux)*np.isfinite(ferr)
        m *= quality == 0
        self.time = np.array(time, dtype=np.float64)[m]
        self.flux = np.array(flux, dtype=np.float64)[m]
        self.ferr = np.array(ferr, dtype=np.float64)[m]

        # Normalize by the median.
        mu = np.median(self.flux)
        self.flux /= mu
        self.ferr /= mu

        # Loop over predictor light curves and interpolate over the missing
        # data points in each one.
        x = self.time
        self.predictors = []
        for pred in predictors:
            y = np.array(pred, dtype=np.float64)[m]
            bad = ~np.isfinite(y)
            if np.any(bad):
                y[bad] = np.interp(x[bad], x[~bad], y[~bad])
            y /= np.median(y)
            self.predictors.append(y)
        if len(self.predictors):
            self.predictors = np.vstack(self.predictors).T

    def __len__(self):
        return len(self.time)

    def median_detrend(self, dt=4.):
        x, y = np.atleast_1d(self.time), np.atleast_1d(self.flux)
        assert len(x) == len(y)
        r = np.empty(len(y))
        for i, t in enumerate(x):
            inds = np.abs(x-t) < 0.5 * dt
            r[i] = np.median(y[inds])
        self.flux /= r
        self.ferr /= r
        return r
