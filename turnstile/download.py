# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Download"]

import os
import sys
import kplr
import tarfile
from functools import partial

from .pipeline import Pipeline


def prepare_download(query, parent_response):
    pass


class Download(Pipeline):

    query_parameters = {
        "kicid": (None, True),
        "tarball_root": (None, True),
        "data_root": (None, True),
        "short_cadence": (False, False),
        "npredictor": (50, False),
    }

    def get_result(self, query, parent_response):
        # Connect to the API.
        client = kplr.API(data_root=query["data_root"])
        kicid = query["kicid"]
        kic = client.star(kicid)
        kic.kois

        # Download the light curves.
        short_cadence = query["short_cadence"]
        data = kic.get_light_curves(short_cadence=short_cadence)
        if not len(data):
            raise ValueError("No light curves for KIC {0}"
                             .format(query["kicid"]))

        # Find predictor stars sorted by distance. TODO: try other sets.
        npredictor = query["npredictor"]
        print("Downloading predictor light curves")
        q = dict(
            ktc_kepler_id="!={0:d}".format(kicid),
            ra=kic.kic_degree_ra, dec=kic.kic_dec, radius=1000,
            ktc_target_type="LC", max_records=npredictor,
        )
        predictor_lcs = []
        for lc in data:
            sys.stdout.write(".")
            sys.stdout.flush()
            q["sci_data_quarter"] = lc.sci_data_quarter
            predictor_lcs += [client.light_curves(**q)]

        # Work out all of the KIC IDs that we'll touch.
        kicids = (set("{0:09d}".format(int(kicid)))
                  | set(lc.kepid for quarter in predictor_lcs
                        for lc in quarter))

        # Extract the relevant tarfiles.
        map(partial(self._extract_light_curves, query["tarball_root"],
                    os.path.join(query["data_root"], "data")),
            kicids)

        #
        print([lc.cache_exists for lc in data])
        print(len([lc.cache_exists for quarter in predictor_lcs for lc in quarter]))
        assert 0

        return dict(star=kic, target_datasets=data,
                    predictor_datasets=predictor_lcs)

    def _extract_light_curves(self, tarball_root, data_root, kicid):
        tarball = os.path.join(tarball_root, kicid + ".tar.gz")
        try:
            with tarfile.open(tarball, "r") as f:
                f.extractall(data_root)
        except:
            print("fail: ", tarball)
        else:
            print("success: ", tarball)
