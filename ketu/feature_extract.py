# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["FeatureExtract"]

import os
import h5py
import numpy as np

from .pipeline import Pipeline


def check_orbits(p1, t1, p2, t2, tmn, tmx, tol):
    n1 = t1 + p1 * np.arange(np.floor((tmn-t1)/p1), np.ceil((tmx-t1)/p1))
    n1 = n1[(tmn <= n1) * (n1 <= tmx)]
    n2 = t2 + p2 * np.arange(np.floor((tmn-t2)/p2), np.ceil((tmx-t2)/p2))
    n2 = n2[(tmn <= n2) * (n2 <= tmx)]
    delta = np.fabs(n1[:, None] - n2[None, :])
    return max(len(n1), len(n2)) == np.sum(delta < tol)


class FeatureExtract(Pipeline):

    cache_ext = ".h5"
    query_parameters = dict()

    def get_result(self, query, parent_response):
        # Build a data type with the peak data.
        peaks = parent_response.peaks
        dtype = [(k, float) for k in sorted(peaks[0].keys())]

        # Add in the meta data columns.
        dtype += [
            ("meta_starid", int), ("meta_hasinj", bool), ("meta_isrec", bool),
            ("meta_inverted", bool),
            ("inj_period", float), ("inj_t0", float), ("inj_radius", float),
            ("inj_b", float), ("inj_e", float), ("inj_pomega", float),
        ]
        injections = query.get("injections", [])
        inj_cols = ["period", "t0", "radius", "b", "e", "pomega"]

        # Find the minimum and maximum observed times.
        lcs = parent_response.model_light_curves
        tmn = np.min([np.min(lc.time) for lc in lcs])
        tmx = np.max([np.max(lc.time) for lc in lcs])

        # Loop over the peaks and insert the data.
        features = np.empty(len(peaks), dtype=dtype)
        for i, peak in enumerate(peaks):
            features[i]["meta_starid"] = parent_response.starid
            features[i]["meta_inverted"] = query.get("invert", False)

            # Peak data.
            for k, v in peak.items():
                features[i][k] = v

            # Injections.
            for k in inj_cols:
                features[i]["inj_" + k] = np.nan
            if len(injections):
                features[i]["meta_hasinj"] = True
                isinj = False
                for inj in injections:
                    if check_orbits(
                            peak["period"], peak["t0"],
                            inj["period"], inj["t0"],
                            tmn, tmx, 0.5 * peak["duration"]):
                        isinj = True
                        for k in inj_cols:
                            features[i]["inj_" + k] = inj[k]
                        break
                features[i]["meta_isrec"] = isinj
            else:
                features[i]["meta_hasinj"] = False
                features[i]["meta_isrec"] = False

        return dict(features=features)

    def save_to_cache(self, fn, response):
        try:
            os.makedirs(os.path.dirname(fn))
        except os.error:
            pass

        with h5py.File(fn, "w") as f:
            f.create_dataset("features", data=response["features"])

    def load_from_cache(self, fn):
        if os.path.exists(fn):
            with h5py.File(fn, "r") as f:
                return dict(features=f["features"][...])
        return None
