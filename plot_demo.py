#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as pl


data = np.loadtxt("lc.txt")
pl.plot(data[:, 0] % 4.0, data[:, 1], ".k")
pl.savefig("demo.png")
