# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Download", "PreparedDownload"]

import os
import sys
import kplr
import shutil
import cPickle as pickle

from .pipeline import Pipeline


def prepare_download(query, fetch=True):
    # Connect to the API.
    client = kplr.API(data_root=query["data_root"])
    kicid = query["kicid"]
    kic = client.star(kicid)
    kic.kois

    # Get the entry in the Huber et al. catalog.
    hcat = kplr.huber.get_catalog()
    kic.huber = hcat[hcat.KIC == kicid].iloc[0]

    # Download the light curves.
    short_cadence = query["short_cadence"]
    data = kic.get_light_curves(short_cadence=short_cadence, fetch=fetch)
    if not len(data):
        raise ValueError("No light curves for KIC {0}".format(kicid))

    # Find predictor stars sorted by distance. TODO: try other sets.
    npredictor = query["npredictor"]
    if npredictor > 0:
        print("Downloading predictor light curves")
        q = dict(
            ktc_kepler_id="!={0:d}".format(kicid),
            ra=kic.kic_degree_ra, dec=kic.kic_dec, radius=1000,
            ktc_target_type="LC", max_records=npredictor, fetch=fetch,
        )
        predictor_lcs = []
        for lc in data:
            sys.stdout.write(".")
            sys.stdout.flush()
            q["sci_data_quarter"] = lc.sci_data_quarter
            predictor_lcs += [client.light_curves(**q)]
        print()

        # Work out all of the KIC IDs that we'll touch.
        kicids = (set(["{0:09d}".format(int(kicid))])
                  | set(lc.kepid for quarter in predictor_lcs
                        for lc in quarter))

    else:
        predictor_lcs = [[] for i in range(len(data))]
        kicids = set(["{0:09d}".format(int(kicid))])

    return kicids, kic, data, predictor_lcs


class Download(Pipeline):

    query_parameters = {
        "kicid": (None, True),
        "short_cadence": (False, False),
        "npredictor": (0, False),
        "data_root": (None, False),
    }

    def get_result(self, query, parent_response):
        _, kic, data, predictor_lcs = prepare_download(query)
        return dict(star=kic, target_datasets=data,
                    predictor_datasets=predictor_lcs)


class PreparedDownload(Pipeline):

    query_parameters = dict(
        kicid=(None, True),
        prepared_file=(None, True),
    )

    def get_result(self, query, parent_response):
        with open(query["prepared_file"], "rb") as f:
            kic, data, predictor_lcs = pickle.load(f)
        assert int(query["kicid"]) == int(kic.kepid), \
            "Invalid prepared download."
        return dict(star=kic, target_datasets=data,
                    predictor_datasets=predictor_lcs)

    @classmethod
    def prepare(cls, outfn, archive_root, data_root, kicid,
                short_cadence=False, npredictor=50):
        # Parse the query.
        query = dict(kicid=kicid, short_cadence=short_cadence,
                     npredictor=npredictor, data_root=data_root)
        kicids, kic, data, predictor_lcs = prepare_download(query, fetch=False)

        # Extract all the relevant light curves.
        print("Copying light curves")
        for id_ in kicids:
            dest = os.path.join(data_root, "data", "lightcurves", id_)
            if os.path.exists(dest):
                continue
            try:
                os.makedirs(os.path.dirname(dest))
            except os.error:
                pass
            src = os.path.join(archive_root, id_)
            shutil.copytree(src, dest)

        # Save the prepared listing.
        with open(outfn, "wb") as f:
            pickle.dump((kic, data, predictor_lcs), f, -1)
