#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "turnstile.h"

double quick_select(double arr[], int n);

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

lightcurve *lightcurve_fold_and_bin (lightcurve *lc, double period, double dt,
                                     int method)
{
    int i, n = lc->length, bin;

    int nbins = (int)(period / dt) + 1;
    if (nbins == 0) nbins = 1;

    lightcurve *folded = lightcurve_alloc(nbins);
    for (i = 0; i < nbins; ++i)
        folded->time[i] = i * dt + 0.5 * dt;

    if (method == 0) { // Weighted mean.
        for (i = 0; i < n; ++i) {
            bin = (int)(fmod(lc->time[i], period) / dt);
            if (bin >= nbins) {
                fprintf(stderr, "Index failure (t=%f)\n", lc->time[i]);
                bin = nbins - 1;
            }
            folded->flux[bin] += lc->flux[i] * lc->ivar[i];
            folded->ivar[bin] += lc->ivar[i];
        }

        for (i = 0; i < nbins; ++i)
            folded->flux[i] /= folded->ivar[i];

    } else if (method == 1) { // Median.
        int *counts = malloc(nbins * sizeof(int));
        double **samples = malloc(nbins * sizeof(double*));

        for (i = 0; i < nbins; ++i) {
            counts[i] = 0;
            samples[i] = malloc(n * sizeof(double));
        }

        for (i = 0; i < n; ++i) {
            bin = (int)(fmod(lc->time[i], period) / dt);
            if (bin >= nbins) {
                fprintf(stderr, "Index failure (t=%f)\n", lc->time[i]);
                bin = nbins - 1;
            }
            samples[bin][counts[bin]++] = lc->flux[i];
            folded->ivar[bin] += lc->ivar[i];
        }

        for (bin = 0; bin < nbins; ++bin) {
            folded->flux[bin] = quick_select(samples[bin], counts[bin]);
            free(samples[bin]);
        }

        free(samples);
        free(counts);
    }

    return folded;
}

void test_epoch(lightcurve *lc, int nbins, double *depth, int *epoch)
{
    int i, j, n, m;
    double depth0, w;
    *depth = -1.0;
    *epoch = -1.0;
    for (i = 0, n = lc->length; i < n; ++i) {
        depth0 = 0.0;
        w = 0.0;
        for (j = 0; j < nbins; ++j) {
            m = (i + j) % n;
            depth0 += lc->flux[m] * lc->ivar[m];
            w += lc->ivar[m];
        }
        depth0 = (1 - depth0 / w);
        if (depth0 > *depth) {
            *depth = depth0;
            *epoch = i;
        }
    }
}

// MEDIANS.
#define ELEM_SWAP(a,b) { register double t=(a);(a)=(b);(b)=t; }
double quick_select(double arr[], int n)
{
    int low, high;
    int median;
    int middle, ll, hh;

    low = 0;
    high = n-1;
    median = (low + high) / 2;
    for (;;) {
        if (high <= low) /* One element only */
            return arr[median];

        if (high == low + 1) {  /* Two elements only */
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]);
            return arr[median];
        }

        /* Find median of low, middle and high items; swap into position low */
        middle = (low + high) / 2;
        if (arr[middle] > arr[high])    ELEM_SWAP(arr[middle], arr[high]) ;
        if (arr[low] > arr[high])       ELEM_SWAP(arr[low], arr[high]) ;
        if (arr[middle] > arr[low])     ELEM_SWAP(arr[middle], arr[low]) ;

        /* Swap low item (now in position middle) into position (low+1) */
        ELEM_SWAP(arr[middle], arr[low+1]) ;

        /* Nibble from each end towards middle, swapping items when stuck */
        ll = low + 1;
        hh = high;
        for (;;) {
            do ll++; while (arr[low] > arr[ll]) ;
            do hh--; while (arr[hh]  > arr[low]) ;

            if (hh < ll)
            break;

            ELEM_SWAP(arr[ll], arr[hh]) ;
        }

        /* Swap middle item (in position low) back into correct position */
        ELEM_SWAP(arr[low], arr[hh]) ;

        /* Re-set active partition */
        if (hh <= median)
            low = ll;
            if (hh >= median)
            high = hh - 1;
    }
}
#undef ELEM_SWAP
