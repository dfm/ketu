#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "turnstile.h"

lightcurve *lightcurve_alloc (int length)
{
    int i;
    lightcurve *lc = malloc(sizeof(lightcurve));
    lc->length = length;
    lc->time = malloc(length * sizeof(double));
    lc->flux = malloc(length * sizeof(double));
    lc->ivar = malloc(length * sizeof(double));

    for (i = 0; i < length; ++i) {
        lc->flux[i] = 0.0;
        lc->ivar[i] = 0.0;
    }

    lc->min_time = -1;
    lc->max_time = -1;
    return lc;
}

void lightcurve_free (lightcurve *lc)
{
    free(lc->time);
    free(lc->flux);
    free(lc->ivar);
    free(lc);
}

void lightcurve_compute_extent (lightcurve *lc)
{
    int i, n;
    double *t = lc->time;
    double mx = t[0], mn = t[0];
    for (i = 1, n = lc->length; i < n; ++i) {
        if (t[i] < mn) mn = t[i];
        if (t[i] > mx) mx = t[i];
    }
    lc->min_time = mn;
    lc->max_time = mx;
}

lightcurve *lightcurve_fold_and_bin (lightcurve *lc, double period, double dt)
{
    int i, n, bin;

    int nbins = (int)((period + 0.5 * dt) / dt);
    if (nbins == 0) nbins = 1;

    lightcurve *folded = lightcurve_alloc(nbins);
    for (i = 0; i < nbins; ++i)
        folded->time[i] = i * dt + 0.5 * dt;

    for (i = 0, n = lc->length; i < n; ++i) {
        bin = (int)(fmod(lc->time[i], period) / dt);
        if (bin >= nbins) {
            fprintf(stderr, "Index failure.");
            bin = nbins - 1;
        }
        folded->flux[bin] += lc->flux[i] * lc->ivar[i];
        folded->ivar[bin] += lc->ivar[i];
    }

    for (i = 0; i < nbins; ++i) {
        folded->flux[i] /= folded->ivar[i];
        printf("%f %f %f\n", folded->time[i], folded->flux[i], folded->ivar[i]);
    }

    return folded;
}
