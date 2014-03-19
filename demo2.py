#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys
import cPickle as pickle
import matplotlib.pyplot as pl
from turnstile.grid import Grid


if "--pre" in sys.argv:
    grid = Grid(12506954)
    injection = grid.inject_transit(275.0, 0.08)
    grid.optimize_hyperparams()
    pickle.dump(grid, open("demo.pkl", "wb"), -1)
else:
    grid = pickle.load(open("demo.pkl"))

inj = grid.injections[0]
ror = inj["rp"]/inj["rstar"]
print(ror)

times, delta_lls = grid.compute_hypotheses([ror**2], [0.4])
