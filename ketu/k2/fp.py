# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["FP"]

import numpy as np
from scipy.linalg import cho_factor, cho_solve

try:
    from astropy.io import fits
    from astropy.wcs import WCS
except ImportError:
    fits = None

from ..pipeline import Pipeline
from .._compute import _box_model


# These are some useful things to pre-compute and use later.
_x, _y = np.meshgrid(range(-1, 2), range(-1, 2), indexing="ij")
_x, _y = _x.flatten(), _y.flatten()
_AT = np.vstack((_x*_x, _y*_y, _x*_y, _x, _y, np.ones_like(_x)))
_ATA = np.dot(_AT, _AT.T)
factor = cho_factor(_ATA, overwrite_a=True)


# This function finds the centroid and second derivatives in a 3x3 patch.
def fit_3x3(img):
    a, b, c, d, e, f = cho_solve(factor, np.dot(_AT, img.flatten()))
    m = 1. / (4 * a * b - c*c)
    x = (c * e - 2 * b * d) * m
    y = (c * d - 2 * a * e) * m
    return x, y


# This function finds the centroid in an image.
# You can provide an estimate of the centroid using WCS.
def find_centroid(img, init=None):
    if init is None:
        xi, yi = np.unravel_index(np.argmax(img), img.shape)
    else:
        xi, yi = map(int, map(np.round, init))
        ox, oy = np.unravel_index(np.argmax(img[xi-1:xi+2, yi-1:yi+2]), (3, 3))
        xi += ox - 1
        yi += oy - 1
    assert (xi >= 1 and xi < img.shape[0] - 1), "effed, x"
    assert (yi >= 1 and yi < img.shape[1] - 1), "effed, y"
    ox, oy = fit_3x3(img[xi-1:xi+2, yi-1:yi+2])
    return ox + xi, oy + yi


class FP(Pipeline):

    query_parameters = {
        "target_pixel_file": (None, True),
    }

    def __init__(self, *args, **kwargs):
        if fits is None:
            raise ImportError("astropy is required")
        super(FP, self).__init__(*args, **kwargs)

    def get_result(self, query, parent_response):
        # Get the light curve object.
        lc = parent_response.model_light_curves[0]

        # Compute the WCS coordinates.
        with fits.open(query["target_pixel_file"]) as hdus:
            hdr = hdus[2].header
            wcs = WCS(hdr)
            # The order is the opposite of what I would normally use...
            init = wcs.wcs_world2pix(hdr["RA_OBJ"], hdr["DEC_OBJ"], 0.0)[::-1]

            # Loop over the images and measure the centroid.
            coords = np.empty((len(lc.time), 2))
            j = 0
            for i, img in enumerate(hdus[1].data["FLUX"]):
                if lc.m[i]:
                    coords[j] = find_centroid(img, init=init)
                    j += 1

        return dict(
            fp_model=FPModel(lc, coords)
        )


class FPModel(object):

    def __init__(self, lc, coords):
        self.lc = lc
        self.coords = coords

    def compute_odd_even(self, period, t0, duration):
        lc = self.lc
        model = np.zeros((2, lc.basis.shape[1]))
        for i in (0, 1):
            m = np.abs((lc.time - t0 + i * period) % (2 * period) - period
                       < 0.5 * duration)
            model[i, m] = -1

        # Compute the offset.
        A = np.concatenate((lc.basis, model))
        ATA = np.dot(A, A.T)
        ATA[np.diag_indices_from(ATA)] += 1e-10

        # Compute the depths and uncertainties.
        w = np.linalg.solve(ATA, np.dot(A, lc.flux))
        S = np.linalg.inv(ATA)

        return w[-2:], np.sqrt(np.diag(S[-2:, -2:] / lc.ivar))

    def compute_offsets(self, period, t0, duration):
        lc = self.lc

        # Compute the offset.
        A = np.concatenate((lc.basis, np.zeros((1, lc.basis.shape[1]))))
        ATA = np.dot(A, A.T)
        ATA[np.diag_indices_from(ATA)] += 1e-10

        # Loop over transits.
        model = _box_model()
        model.half_duration = 0.5 * duration
        model.t0 = t0
        num = 0.0
        denom = 0.0
        while model.t0 < lc.time.max():
            # Compute the box model.
            m = model(lc.time)
            if np.any(m != 0.0):
                # Update the matrix.
                v = np.dot(lc.basis, m)
                ATA[:-1, -1] = v
                ATA[-1, :-1] = v
                ATA[-1, -1] = np.dot(m, m)
                A[-1, :] = m

                # Solve for the offset and weight.
                w = np.linalg.solve(ATA, np.dot(A, self.coords))
                S = np.linalg.inv(ATA)
                ivar = 1.0 / np.sqrt(S[-1, -1])

                # Update the weighted sum.
                num += np.sqrt(np.sum(w[-1]**2)) * ivar
                denom += ivar
            model.t0 += period

        return num / denom
