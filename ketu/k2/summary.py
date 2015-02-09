# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Summary"]

import fitsio
import numpy as np

try:
    import matplotlib.pyplot as pl
except ImportError:
    pl = None
else:
    from matplotlib.ticker import MaxNLocator
    from matplotlib.backends.backend_pdf import PdfPages

from ..pipeline import Pipeline


class Summary(Pipeline):

    query_parameters = dict(
        summary_file=(None, True),
        tpf_file=(None, True),
        signals=([], False),
    )

    def __init__(self, *args, **kwargs):
        if pl is None:
            raise ImportError("matplotlib is required")
        super(Summary, self).__init__(*args, **kwargs)

    def get_result(self, query, parent_response):
        from matplotlib import rcParams
        rcParams["text.usetex"] = True

        epic = parent_response.epic

        tpf = fitsio.read(query["tpf_file"])
        lc_data = fitsio.read(query["light_curve_file"])
        x = np.all(lc_data["flux"], axis=1)
        i = np.arange(len(x))[np.isfinite(x)][-1]
        # xi, yi = x[i], y[i]

        # Get the light curve object.
        lc = parent_response.model_light_curves[0]

        # Loop over signals and compute the mean model.
        model = np.zeros_like(lc.time)
        for row in query["signals"]:
            p, t0, d, tau = (row[k] for k in ["period", "t0", "depth",
                                              "duration"])
            hp = 0.5 * p
            m = np.abs((lc.time - t0 + hp) % p - hp) < 0.5*tau
            model[m] -= d

        # Make the prediction.
        t = lc.time
        f = lc.flux - lc.predict(lc.flux - model)

        # Dimensions.
        full_h = 0.97 / 6
        vspace_b, vspace_t = 0.2 * full_h, 0.1 * full_h
        inner_h = full_h - vspace_t - vspace_b
        full_w = 0.5
        hspace_l, hspace_r = 0.15 * full_w, 0.05 * full_w
        inner_w = full_w - hspace_l - hspace_r

        label_dict = dict(xy=(0, 0), xycoords="axes fraction",
                          xytext=(6, 6), textcoords="offset points",
                          ha="left", va="bottom")

        with PdfPages(query["summary_file"]) as pdf:
            # Initialize the figure.
            fig = pl.figure(figsize=(9, 12))

            # Make a figure for every signal.
            colors = "grb"
            for ntransit, (row, color) in enumerate(zip(query["signals"],
                                                        colors)):
                fig.clf()

                # Plot the full light curve.
                ax = pl.axes([hspace_l, 5. * full_h + vspace_b,
                              2*inner_w + hspace_l + hspace_r, inner_h])
                ax.plot(t, f, ".k", rasterized=True)
                ax.set_title((r"EPIC {0} \#{1} --- period: {2:.3f} d, "
                              "depth: {3:.3f} ppt").format(
                                  epic.id,
                                  ntransit + 1,
                                  row["period"],
                                  row["depth"],
                ))

                # Plot the transit locations.
                for mod, c in zip(query["signals"], colors):
                    p, t0 = (mod[k] for k in ["period", "t0"])
                    while True:
                        ax.axvline(t0, color=c, lw=1.5, alpha=0.5)
                        t0 += p
                        if t0 > t.max():
                            break

                # Format the full light curve plot.
                ax.set_xlim(t.min(), t.max())
                ax.set_ylabel("rel. flux [ppt]")
                ax.set_xlabel("time [days]")
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ylim = ax.get_ylim()

                # Compute the folded times.
                p, t0, depth = (row[k] for k in ["period", "t0", "depth"])
                hp = 0.5 * p
                t_fold = (t - t0 + hp) % p - hp

                # Plot the zoomed folded light curve.
                ax = pl.axes([hspace_l, 4. * full_h + vspace_b, inner_w,
                              inner_h])
                ax.plot(t_fold, f, ".k", rasterized=True)
                ax.axvline(0, color=color, lw=1.5, alpha=0.5)
                ax.annotate("phased (zoom)", **label_dict)
                ax.set_xlim(-0.7, 0.7)
                ax.set_ylim(ylim)
                ax.set_ylabel("rel. flux [ppt]")
                ax.set_xlabel("time since transit [days]")
                ax.yaxis.set_major_locator(MaxNLocator(5))

                # Plot the folded light curve.
                ax = pl.axes([hspace_l, 3. * full_h + vspace_b, inner_w,
                              inner_h])
                ax.plot(t_fold, f, ".k", rasterized=True)
                ax.axvline(0, color=color, lw=1.5, alpha=0.5)
                ax.annotate("phased", **label_dict)
                ax.set_xlim(-1.7, 1.7)
                ax.set_ylim(ylim)
                ax.set_ylabel("rel. flux [ppt]")
                ax.set_xlabel("time since transit [days]")
                ax.yaxis.set_major_locator(MaxNLocator(5))

                # Plot the secondary.
                ax = pl.axes([hspace_l, 2. * full_h + vspace_b, inner_w,
                              inner_h])
                t_fold_2 = (t - t0) % p - hp
                ax.plot(t_fold_2, f, ".k", rasterized=True)
                ax.axvline(0, color=color, lw=1.5, alpha=0.5)
                ax.annotate("phased secondary", **label_dict)
                ax.set_xlim(-1.7, 1.7)
                ax.set_ylim(ylim)
                ax.set_ylabel("rel. flux [ppt]")
                ax.set_xlabel("time since secondary [days]")
                ax.yaxis.set_major_locator(MaxNLocator(5))

                # Plot the stacked transits.
                ax = pl.axes([full_w + hspace_l, 2. * full_h + vspace_b,
                              inner_w,
                              3 * inner_h + 2 * vspace_b + 2 * vspace_t])
                num = ((t - t0 - hp) // p).astype(int)
                num -= min(num)
                offset = 2 * depth
                for n in set(num):
                    m = (num == n) & (np.abs(t_fold) < 1.7)
                    c = "rb"[n % 2]
                    ax.plot(t_fold[m], f[m] + offset * n, c)
                    ax.axhline(offset * n, color="k", linestyle="dashed")
                ax.axvline(0, color="k", lw=1.5, alpha=0.5)
                ax.set_yticklabels([])
                ax.set_ylim(-offset, (max(num) + 0.5) * offset)
                ax.set_xlim(-1.7, 1.7)
                ax.set_xlabel("time since transit [days]")

                # Plot the tpf.
                ax = pl.axes([hspace_l, vspace_b, inner_w,
                              2 * inner_h + vspace_b + vspace_t])
                img = tpf["FLUX"][i].T
                ax.imshow(-img, cmap="gray", interpolation="nearest")
                ax.annotate("frame", **label_dict)
                ax.set_xlim(-0.5, img.shape[1] - 0.5)
                ax.set_ylim(-0.5, img.shape[0] - 0.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

                # Plot the log-tpf.
                ax = pl.axes([full_w + hspace_l, vspace_b, inner_w,
                              2 * inner_h + vspace_b + vspace_t])
                limg = np.nan + np.zeros_like(img)
                m = np.isfinite(img) & (img > 0)
                limg[m] = np.log(img[m])
                ax.imshow(-limg, cmap="gray", interpolation="nearest")
                ax.annotate("log(frame)", **label_dict)
                ax.set_xlim(-0.5, img.shape[1] - 0.5)
                ax.set_ylim(-0.5, img.shape[0] - 0.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

                pdf.savefig(fig)

        pl.close()
        return parent_response
