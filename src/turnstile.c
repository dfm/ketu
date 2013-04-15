#include <stdio.h>
#include <math.h>
#include "kepler.h"

void stupid_lc (float *t, int nt,
                float period, float epoch, float duration, float depth,
                float *lc)
{
    float t0;
    for (int i = 0; i < nt; ++i) {
        t0 = fmod(t[i], period) - epoch;
        if (t0 <= duration) lc[i] = 1.0 - depth;
        else lc[i] = 1.0;
    }
}

int fit_dataset (dataset *data)
{
    int i, N = data->length;
    float *time = data->time, *flux = data->flux, *ivar = data->ivar;
    float duration, period, epoch, depth;
    float dperiod, fmin, fmax, tnorm, maxdepth;
    int nmin, nmax;

    // Compute min and max times.
    float tmin = -1.0, tmax = -1.0, delta;
    for (i = 0; i < N; ++i) {
        if (tmin < 0.0 || tmin > time[i]) tmin = time[i];
        if (tmax < 0.0 || tmax < time[i]) tmax = time[i];
    }
    fprintf(stdout, "t_min = %f, t_max = %f\n", tmin, tmax);
    delta = 1 / (tmax - tmin);

    duration = 0.1;
    /* for (duration = KEPLER_LONG; duration < 1.0; duration += 10 * KEPLER_LONG) { */
        dperiod = 1 + duration * delta;
        for (period = 0.8; period < 0.9; period *= dperiod) {
            maxdepth = 0.0;
            for (epoch = 0.0; epoch < period; epoch += KEPLER_LONG) {
                fmin = 0.0;
                fmax = 0.0;
                nmin = 0;
                nmax = 0;
                for (i = 0; i < N; ++i) {
                    tnorm = fmod(time[i] - epoch, period);
                    if (tnorm < duration) {
                        fmin += flux[i];
                        nmin++;
                    } else {
                        fmax += flux[i];
                        nmax++;
                    }
                }
                depth = fmax / (float)nmax - fmin / (float)nmin;
                if (depth > maxdepth) maxdepth = depth;
            }
            fprintf(stdout, "%f %f %f\n", duration, period, maxdepth);
        }
    /* } */

    return 0;
}

int main()
{
    const char *filename = "data.fits";

    dataset *data = read_kepler_lc(filename);
    printf("%d ", data->length);
    mask_dataset(data);
    printf("%d\n", data->length);
    fit_dataset(data);
    free_dataset(data);

    return 0;
}
