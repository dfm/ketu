#ifndef _TURNSTILE_H_
#define _TURNSTILE_H_

typedef struct lightcurve {

    int length;
    double *time, *flux, *ivar;
    double min_time, max_time;

} lightcurve;

lightcurve *lightcurve_alloc (int length);
void lightcurve_free (lightcurve *lc);
void lightcurve_compute_extent (lightcurve *lc);

lightcurve *lightcurve_fold_and_bin (lightcurve *lc, double period, double dt);

#endif
