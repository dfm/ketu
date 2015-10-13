# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Catalog"]

import os
import requests
import pandas as pd
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


class Catalog(object):

    url = ("http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/"
           "nph-nstedAPI?table=k2targets&select=*")

    def __init__(self, filename):
        self._df = None
        self.name = "epic"
        self.filename = filename

    @property
    def df(self):
        if self._df is None:
            self.download()
            self._df = pd.read_hdf(self.filename, self.name)
        return self._df

    def download(self, clobber=False):
        if os.path.exists(self.filename) and not clobber:
            return

        # Request the table.
        r = requests.get(self.url)
        if r.status_code != requests.codes.ok:
            r.raise_for_stataus()

        # Load the contents using pandas.
        self._df = pd.read_csv(StringIO(r.content))

        # Save it to an HDF5 file.
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.filename)))
        except os.error:
            pass
        self._df.to_hdf(self.filename, self.name, format="t")
