# -*- coding: utf-8 -*-

from __future__ import print_function, division

__all__ = ["run"]

import os
import numpy as np

try:
    from astropy.io import fits
    from astropy.wcs import WCS
except ImportError:
    fits = None

from ..cdpp import compute_cdpp

apertures = np.arange(0.5, 5.5, 0.5)
dt = np.dtype([("cadenceno", np.int32), ("time", np.float32),
               ("timecorr", np.float32), ("pos_corr1", np.float32),
               ("pos_corr2", np.float32), ("quality", np.int32),
               ("flux", (np.float32, len(apertures))),
               ("bkg", (np.float32, len(apertures)))])
dt2 = np.dtype([("radius", np.float32), ("cdpp6", np.float32)])


def run(fn, clobber=False):
    if fits is None:
        raise ImportError("astropy is required for photometry.")

    # Skip short cadence targets.
    if "spd" in fn:
        return

    # Skip custom targets.
    epicid = os.path.split(fn)[-1].split("-")[0][4:]
    if int(epicid) < 201000000:
        return

    # Construct the output filename.
    pre, post = os.path.split(fn)
    a, b, _ = post.split("-")
    outfn = os.path.join(pre.replace("target_pixel_files/", "lightcurves/"),
                         a+"-"+b+"-lc.fits")

    # Don't overwrite.
    if os.path.exists(outfn) and not clobber:
        return

    # Read the data.
    with fits.open(fn) as hdus:
        data = hdus[1].data
        table = np.empty(len(data["TIME"]), dtype=dt)
        t = data["TIME"]
        m = np.isfinite(t)
        print("Mean time: {0}".format(0.5*(t[m].min()+t[m].max())))

        # Initialize the new columns to NaN.
        for k in ["flux", "bkg"]:
            table[k] = np.nan

        # Copy across the old columns.
        for k in ["cadenceno", "time", "timecorr", "pos_corr1", "pos_corr2",
                  "quality"]:
            table[k] = data[k.upper()]

        # Use the WCS to find the center of the star.
        hdr = hdus[2].header
        wcs = WCS(hdr)
        cy, cx = wcs.wcs_world2pix(hdr["RA_OBJ"], hdr["DEC_OBJ"], 0.0)

        # Choose the set of apertures.
        aps = []
        shape = data["FLUX"][0].shape
        xi, yi = np.meshgrid(range(shape[0]), range(shape[1]), indexing="ij")
        for r in apertures:
            r2 = r*r
            aps.append((xi - cx) ** 2 + (yi - cy) ** 2 < r2)

        # Loop over the frames and do the aperture photometry.
        for i, img in enumerate(data["FLUX"]):
            fm = np.isfinite(img)
            fm[fm] = img[fm] > 0.0
            if not np.any(fm):
                continue
            for j, mask in enumerate(aps):
                # Choose the pixels in and out of the aperture.
                m = mask * fm
                bgm = (~mask) * fm

                # Skip if there are no good pixels in the aperture.
                if not np.any(m):
                    continue

                # Estimate the background and flux.
                if np.any(bgm):
                    bkg = np.median(img[bgm])
                else:
                    bkg = np.median(img[mask])
                table["flux"][i, j] = np.sum(img[m] - bkg)
                table["bkg"][i, j] = bkg

        # Compute the number of good times.
        nt = int(np.sum(np.any(np.isfinite(table["flux"]), axis=1)))
        print("{0} -> {1} ; {2}".format(fn, outfn, nt))

        # Skip it if there aren't *any* good times.
        if nt == 0:
            return

        # Save the aperture information and precision.
        ap_info = np.empty(len(apertures), dtype=dt2)
        for i, r in enumerate(apertures):
            ap_info[i]["radius"] = r

            # Compute the precision.
            t, f = table["time"], table["flux"][:, i]
            ap_info[i]["cdpp6"] = compute_cdpp(t, f, 6.)

        try:
            os.makedirs(os.path.split(outfn)[0])
        except os.error:
            pass
        hdr = hdus[1].header
        hdr["CEN_X"] = float(cx)
        hdr["CEN_Y"] = float(cy)
        hdus_out = fits.HDUList([
            fits.PrimaryHDU(header=hdr),
            fits.BinTableHDU.from_columns(table, header=hdr),
            fits.BinTableHDU.from_columns(ap_info),
        ])

        hdus_out.writeto(outfn, clobber=True)
        hdus_out.close()
        del hdus_out
