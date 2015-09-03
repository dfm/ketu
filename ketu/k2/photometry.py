# -*- coding: utf-8 -*-

from __future__ import print_function, division

__all__ = ["run"]

import os
import numpy as np
from scipy.ndimage.measurements import label

try:
    from astropy.io import fits
    from astropy.wcs import WCS
except ImportError:
    fits = None

from ..cdpp import compute_cdpp


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

    print("{0} -> {1} ; {2}".format(fn, outfn))
    with fits.open(fn) as hdus:
        data = hdus[1].data
        t = data["TIME"]
        # m = np.isfinite(t)
        # print("Midtime: {0}".format(0.5*(t[m].min()+t[m].max())))
        q = data["QUALITY"] == 0

        # Use the WCS to find the center of the star.
        hdr = hdus[2].header
        wcs = WCS(hdr)
        cy, cx = wcs.wcs_world2pix(hdr["RA_OBJ"], hdr["DEC_OBJ"], 0.0)

        # Compute the summed image.
        sum_img = np.zeros(data["FLUX"][0].shape)
        for img in data["FLUX"][q]:
            m = np.isfinite(img)
            if not np.any(m):
                continue
            sum_img[m] += img[m]
        sum_img[~np.isfinite(sum_img)] = 0.0

        # Compute some stats on the image.
        x = sum_img[np.isfinite(sum_img) & (sum_img > 0.0)].flatten()
        mu = np.median(x)
        std = np.median(np.abs(x - mu))

        # Find the shape that overlaps the WCS coordinates.
        shape = data["FLUX"][0].shape
        x_grid, y_grid = np.meshgrid(range(shape[0]), range(shape[1]),
                                     indexing="ij")
        r2 = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        xi, yi = np.unravel_index(np.argmin(r2), shape)

        # Loop over thresholds and compute the apertures.
        params = []
        flx_aps = []
        bkg_aps = []
        for thresh in np.arange(5, 50, 5):
            sum_img[sum_img < mu + thresh*std] = 0.0
            labels, _ = label(sum_img)
            bkg = labels == 0
            flx = labels == labels[xi, yi]
            if labels[xi, yi] != 0 and np.any(bkg) and np.any(flx):
                bkg_aps.append(bkg)
                flx_aps.append(flx)
                params.append(-thresh)

        # Loop over radii and compute circular apertures.
        for rad in np.arange(1, 15, 0.5):
            bkg = r2 >= rad * rad
            flx = r2 < rad * rad
            if np.any(bkg) and np.any(flx):
                bkg_aps.append(bkg)
                flx_aps.append(flx)
                params.append(rad)

        # Loop over the images and do the photometry.
        background = np.nan + np.zeros((len(data), len(params)))
        flux = np.nan + np.zeros((len(data), len(params)))
        for i, img in enumerate(data["FLUX"]):
            fm = np.isfinite(img)
            fm[fm] = img[fm] > 0.0
            for j, (bkg_ap, flx_ap) in enumerate(zip(bkg_aps, flx_aps)):
                if (not np.any(fm)) or (not np.any(fm & flx_ap)):
                    continue
                background[i, j] = np.median(img[fm])
                flux[i, j] = np.mean(img[fm & flx_ap] - background[i, j])
        m = np.any(np.isfinite(flux) & np.isfinite(background), axis=0)
        background = background[:, m]
        flux = flux[:, m]

        # Build the dataset and save the light curve file.
        dt = np.dtype([
            ("cadenceno", np.int32), ("time", np.float32),
            ("timecorr", np.float32), ("quality", np.int32),
            ("flux", (np.float32, flux.shape[1])),
            ("bkg", (np.float32, background.shape[1]))
        ])
        table = np.empty(len(data), dtype=dt)
        for k in ["cadenceno", "time", "timecorr", "quality"]:
            table[k] = data[k.upper()]
        table["flux"] = flux
        table["bkg"] = background

    # Save the information and precision for each aperture.
    ap_info = np.empty(len(params), dtype=[("parameter", np.float32),
                                           ("cdpp6", np.float32)])
    for i, p in enumerate(params):
        ap_info[i]["parameter"] = p
        ap_info[i]["cdpp6"] = compute_cdpp(t[q], flux[q, i], 6., robust=True)

    # Save the file.
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
