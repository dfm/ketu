#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as pl


data = np.loadtxt("out.txt", skiprows=3)
inds = data[:, 1] > 0
pl.plot(data[inds, 0], data[inds, 1], ".k")
# pl.xlim(0.2, 1.0)
pl.savefig("trying.png")
