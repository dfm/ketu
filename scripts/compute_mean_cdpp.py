#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()

# Required arguments.
parser.add_argument("pattern", help="the directory pattern")
parser.add_argument("results", help="the results location")

args = parser.parse_args()
print("Running with the following arguments:")
print("sys.argv:")
print(sys.argv)
print("args:")
print(args)

try:
    os.makedirs(args.results)
except os.error:
    pass

results = dict(ID=[], cdpp3=[], cdpp6=[], cdpp12=[])
for i, fn in enumerate(glob.iglob(args.pattern)):
    data = pd.read_csv(fn, delim_whitespace=True, skiprows=1,
                       compression="gzip")
    data = data.loc[1:]
    for k in results:
        results[k] += list(data[k])

df = pd.DataFrame(results)
df = df.groupby("ID").median()
df["kicid"] = np.array(df.index, dtype=int)
df.to_hdf(os.path.join(args.results, "cdpp.h5"), "cdpp", mode="w")
