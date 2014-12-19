#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import os
import sys
import kplr
import sqlite3
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("base_dir", help="the directory for output")
    parser.add_argument("--min_mag", type=float, default=0.0)
    parser.add_argument("--max_mag", type=float, default=30.0)
    parser.add_argument("--min_teff", type=float, default=0.0)
    parser.add_argument("--max_teff", type=float, default=50000.0)
    parser.add_argument("--min_logg", type=float, default=0.0)
    parser.add_argument("--max_logg", type=float, default=500.0)

    args = parser.parse_args()
    print("Running with the following arguments:")
    print("sys.argv:")
    print(sys.argv)
    print("args:")
    print(args)

    # Make the output directory.
    try:
        os.makedirs(args.base_dir)
    except os.error:
        pass

    # Load the stellar catalog.
    kic = kplr.huber.get_catalog()

    # Make the magnitude cut.
    m = (args.min_mag < kic.kic_kepmag) & (kic.kic_kepmag < args.max_mag)

    # Make the T_eff cut.
    m &= (args.min_mag < kic.Teff) & (kic.Teff < args.max_teff)

    # Make the log_g cut.
    m &= (args.min_logg < kic["log(g)"]) & (kic["log(g)"] < args.max_logg)
    print("Applying cuts for a total of {0} stars".format(m.sum()))

    # Open the database and add the stellar parameters.
    with sqlite3.connect(os.path.join(args.base_dir, "stars.db")) as conn:
        c = conn.cursor()

        # Save the selection cuts.
        c.execute("drop table if exists selection")
        c.execute("create table selection (par name, mn real, mx real)")
        c.execute("insert into selection(par,mn,mx) values (?,?,?)",
                  ("mag", args.min_mag, args.max_mag))
        c.execute("insert into selection(par,mn,mx) values (?,?,?)",
                  ("teff", args.min_teff, args.max_teff))
        c.execute("insert into selection(par,mn,mx) values (?,?,?)",
                  ("logg", args.min_logg, args.max_logg))

        # Save the stars.
        c.execute("drop table if exists stars")
        c.execute("""create table stars (
            kic integer,
            radius real,
            mass real,
            kepmag real,
            teff real,
            logg real,
            ninj integer,
            started bool,
            finished bool
        )""")

        for i, row in kic[m].iterrows():
            c.execute("insert into stars values (?,?,?,?,?,?,?,?,?)",
                      (row.KIC, row.R, row.M, row.kic_kepmag, row.Teff,
                       row["log(g)"], 0, False, False))
