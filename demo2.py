#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys
import cPickle as pickle
import matplotlib.pyplot as pl
from turnstile.grid import Grid


if "--pre" in sys.argv:
    # grid = Grid(12506954)
    grid = Grid(2860283)
    print(grid.kic.kic_kepmag)
    inj = grid.inject_transit(50.0, 0.025)

    lc = grid.get_data()[3]

    pl.clf()
    pl.plot(lc.time, lc.flux)
    pl.savefig("dude.png")

    # assert 0
    print("{0} datasets.".format(len(grid.get_data())))
    # grid.optimize_hyperparams()
    [l.set_gp_pars([1.174e-6, 1.038]) for l in grid.get_data()]
    pickle.dump(grid, open("demo.pkl", "wb"), -1)
else:
    grid = pickle.load(open("demo.pkl"))
    inj = grid.injections[0]

ror = inj["rp"]/inj["rstar"]
print(ror)

grid.compute_hypotheses([ror**2], [0.4])

pl.clf()
pl.plot(grid.times, grid.delta_lls[:, 0, 0], "-k")
pl.savefig("huge.png")
