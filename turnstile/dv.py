# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Validate"]

from matplotlib import rcParams
rcParams["font.size"] = 10
rcParams["text.usetex"] = False

import os
import json
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pl

from .pipeline import Pipeline


PEAK_TEMPLATE = "<img src=\"peak-{0:04d}.png\"><br>"
MAIN_TEMPLATE = """
<html>
<head>
<title>KIC {resp.kicid}</title>
</head>
<body>
<center>
<h1>KIC {resp.kicid}</h1>
<p>kepmag = {resp.kic_kepmag}</p>
<p>T_eff = {resp.kic_teff}</p>
<p>logg = {resp.kic_logg}</p>
<p>{rec_inj} / {tot_inj} injection(s) recovered</p>
<p>{rec_koi} / {tot_koi} KOI(s) recovered</p>
<img src="phic-periodogram.png"><br>
<img src="transit-times.png"><br>
<img src="period-phase.png"><br>
{peaks}

</center>

</body>
</html>

"""


class Validate(Pipeline):

    query_parameters = dict(
        validation_path=(None, True),
    )

    def __init__(self, *args, **kwargs):
        kwargs["cache"] = kwargs.pop("cache", False)
        super(Validate, self).__init__(*args, **kwargs)

    def get_result(self, query, parent_response):
        bp = os.path.join(query["validation_path"],
                          "{0}".format(parent_response.kicid))
        try:
            os.makedirs(bp)
        except os.error:
            pass

        peaks = parent_response.features

        # Plot the PHIC periodogram.
        fig = pl.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        x = parent_response.periods
        y = parent_response.scaled_phic_same
        ax.plot(x, y, "k")
        for i, peak in enumerate(peaks):
            x0, y0 = peak["period"], peak["scaled_phic_same"]
            ax.plot(x0, y0, ".r")
            ax.annotate("{0}".format(i), xy=(x0, y0), ha="center",
                        xytext=(0, 5), textcoords="offset points")
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(min(y), 1.1 * max(y))
        ax.set_xlabel("period")
        ax.set_ylabel("PHIC")
        ax.set_title("periodogram")
        fig.savefig(os.path.join(bp, "phic-periodogram.png"))

        # Plot the locations of the transits.
        fig.clf()
        ax = fig.add_subplot(111)
        tmn, tmx = parent_response.min_time_1d, parent_response.max_time_1d
        for i, peak in enumerate(peaks):
            p, t0 = peak["period"], peak["t0"]
            tt = t0 + p * np.arange(np.floor((tmn-t0)/p), np.ceil((tmx-t0)/p))
            tt = tt[(tmn <= tt) * (tt <= tmx)]
            ax.plot(tt, i*np.ones_like(tt), ".k")
            [ax.axvline(t, color="k", lw=0.5, alpha=0.6) for t in tt]
        ax.set_xlim(tmn, tmx)
        ax.set_ylim(len(parent_response.features) - 0.5, -0.5)
        ax.set_xlabel("time [KBJD]")
        ax.set_ylabel("peak number")
        ax.set_title("transit times")
        fig.savefig(os.path.join(bp, "transit-times.png"))

        # Plot the locations of the peaks in period and phase.
        fig.clf()
        ax = fig.add_subplot(111)
        pmn, pmx = parent_response.min_period, parent_response.max_period
        for i, peak in enumerate(peaks):
            x0, y0 = peak["period"], peak["t0"]
            ax.plot(x0, y0, ".k")
            ax.annotate("{0}".format(i), xy=(x0, y0), ha="center",
                        xytext=(0, 5), textcoords="offset points")
        ax.plot([pmn, pmx], [pmn, pmx], ":k")
        ax.set_xlim(pmn, pmx)
        ax.set_ylim(0, pmx)
        ax.set_xlabel("period [days]")
        ax.set_ylabel("epoch [days]")
        ax.set_title("period vs. offset")
        fig.savefig(os.path.join(bp, "period-phase.png"))

        # Loop over the peaks and plot the light curve for each one.
        fig = pl.figure(figsize=(10, 5))
        cmap = pl.cm.get_cmap("jet")
        dt = parent_response.lc_window_width
        for i, peak in enumerate(peaks):
            fig.clf()
            ax = fig.add_subplot(111)

            lc = peak["corr_lc"]
            ax.scatter(lc["time"], lc["flux"], cmap=cmap,
                       c=lc["transit_number"].astype(float),
                       lw=0)

            # Annotate with the features.
            features = sorted(filter(lambda k: k not in ["corr_lc", "bin_lc",
                                                         "is_koi", "koi_id",
                                                         "is_injection"],
                                     peak.keys()))
            features = "\n".join("{0}: {1:6.3f}".format(k, peak[k])
                                 for k in features).replace("_", " ")
            ax.annotate(features, xy=(1, 0.5), va="center", ha="left",
                        xytext=(5, 0), textcoords="offset points",
                        xycoords="axes fraction", fontsize=11)
            fig.subplots_adjust(left=0.1, bottom=0.15, top=0.9, right=0.7)

            ax.set_xlim(-0.5*dt, 0.5*dt)
            ax.set_ylim(max(np.abs(ax.get_ylim())) * np.array([-1, 1]))
            if peak.get("is_injection", False):
                ax.set_title("peak {0} (injection)".format(i))
            elif peak.get("is_koi", False):
                ax.set_title("peak {0} (KOI {1:.2f})".format(i,
                                                             peak["koi_id"]))
            else:
                ax.set_title("peak {0}".format(i))
            ax.set_xlabel("time since transit")
            ax.set_ylabel("relative flux [ppt]")

            fig.savefig(os.path.join(bp, "peak-{0:04d}.png".format(i)))

        # Format the HTML report.
        inj_rec = parent_response["inj_rec"]
        koi_rec = parent_response["koi_rec"]
        with open(os.path.join(bp, "index.html"), "w") as f:
            f.write(MAIN_TEMPLATE.format(
                resp=parent_response,
                peaks="\n".join(map(PEAK_TEMPLATE.format, range(len(peaks)))),
                rec_inj=sum(inj_rec["rec"]), tot_inj=len(inj_rec),
                rec_koi=sum(koi_rec["rec"]), tot_koi=len(koi_rec),
            ))

        # Save the parent response.
        fn = os.path.join(bp, "features" + self.parent.cache_ext)
        self.parent.save_to_cache(fn, parent_response)

        # Save the query.
        with open(os.path.join(bp, "query.json"), "w") as f:
            json.dump(query, f, sort_keys=True, indent=2)

        # Save the pipeline.
        with open(os.path.join(bp, "pipeline.pkl"), "wb") as f:
            pickle.dump(self, f, -1)
