#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Box least-squares algorithm based on: http://arxiv.org/abs/astro-ph/0206099
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["bls"]

import numpy as np

import _bls


def bls(t, x, nf, fmin, df, nb, qmi, qma):
    u, v = np.zeros_like(t), np.zeros_like(t)
    return _bls.eebls(t, x, u, v, nf, fmin, df, nb, qmi, qma)


if __name__ == "__main__":
    import matplotlib.pyplot as pl
    import pyfits
    with pyfits.open("data.fits") as f:
        time = np.array(f[1].data["TIME"], dtype=float)
        flux = np.array(f[1].data["FLUX"], dtype=float) - 1.0

    inds = ~np.isnan(flux)

    fmin, fmax = 1.0 / 100, 1.0 / 40
    nf = 10000
    df = (fmax - fmin) / nf
    p, bper, bpow, depth, qtran, in1, in2 = bls(time[inds], flux[inds], nf,
                                                fmin, df, 500, 0.05, 0.5)

    print(bper, depth, qtran)

    f = np.linspace(fmin, fmax, nf)
    print(f.shape, p.shape)
    pl.plot(1. / f, p)
    pl.savefig("bls.png")

    pl.clf()
    pl.plot(time % bper, flux, ".k", alpha=0.3)
    pl.xlim(0, 6)
    pl.savefig("lc.png")
