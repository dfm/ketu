#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "george.h"
#include "turnstile.h"

turnstile *turnstile_allocate (
    double duration,
    double depth,
    int ndatasets,
    int *ndata,
    double **time,
    double **flux,
    double **ferr,
    double *hyperpars
)
{
    int i, j, k;

    // Allocate the turnstile object.
    turnstile *self = malloc(sizeof(turnstile));
    self->ndatasets = ndatasets;
    self->ndata = ndata;
    self->duration = duration;
    self->time = malloc(ndatasets * sizeof(double*));
    self->delta_lnlike = malloc(ndatasets * sizeof(double*));

    george_gp *gp;
    double *model, null, t0;
    for (i = 0; i < ndatasets; ++i) {
        // Set up the Gaussian process.
        gp = george_allocate_gp (3, hyperpars, NULL, *george_kernel);
        george_compute (ndata[i], time[i], ferr[i], gp);

        // Compute the null likelihood.
        model = malloc(ndata[i] * sizeof(double));
        for (j = 0; j < ndata[i]; ++j) model[j] = flux[i][j] - 1.0;
        null = george_log_likelihood(model, gp);

        // Loop over the data points and compute the delta log-likelihood at
        // each phase.
        self->delta_lnlike[i] = malloc(ndata[i] * sizeof(double));
        self->time[i] = malloc(ndata[i] * sizeof(double));
        for (j = 0; j < ndata[i]; ++j) {
            self->time[i][j] = t0 = time[i][j];
            for (k = 0; k < ndata[i]; ++k) {
                if (fabs(time[i][k] - t0) < duration)
                    model[k] = flux[i][k] - 1.0 + depth;
                else
                    model[k] = flux[i][k] - 1.0;
            }
            self->delta_lnlike[i][j] = george_log_likelihood(model, gp)
                                       - null;
        }

        free(model);
        george_free_gp(gp);
    }
    return self;
}

void turnstile_free (turnstile *self)
{
    int i;
    for (i = 0; i < self->ndatasets; ++i) {
        free(self->time[i]);
        free(self->delta_lnlike[i]);
    }
    free(self->time);
    free(self->delta_lnlike);
    free(self);
}

void turnstile_evaluate (double period, int nphases, double *phases,
                         double *lnlikes, turnstile *self)
{
    int i, j, k;
    double *folded, phase, best, prev, dist, max_ll, max_phase;

    for (i = 0; i < nphases; ++i) lnlikes[i] = 0.0;

    for (i = 0; i < self->ndatasets; ++i) {
        // Compute the period folded phases of each data point.
        folded = malloc(self->ndata[i] * sizeof(double));
        for (j = 0; j < self->ndata[i]; ++j)
            folded[j] = fmod(self->time[i][j], period);

        // Loop over test phases.
        for (j = 0; j < nphases; ++j) {
            phase = phases[j];

            // Find the closest ln-like measurement (assuming that the times
            // are properly sorted).
            prev = -1.0;
            best = 0.0;
            for (k = 0; k < self->ndata[i]; ++k) {
                dist = fabs(folded[k] - phase);
                if (prev > 0.0 && (dist > prev || k == self->ndata[i]-1)
                        && dist < self->duration) {
                    best = self->delta_lnlike[i][k - 1];
                    break;
                }
                prev = dist;
            }

            // Update the log likelihood.
            lnlikes[j] += best;
        }

        free(folded);
    }

    // max_ll = -INFINITY;
    // max_phase = 0.0;
    // for (i = 0; i < nphases; ++i) {
    //     if (lnlikes[i] > max_ll) {
    //         max_ll = lnlikes[i];
    //         max_phase = phases[i];
    //     }
    // }

    // printf("P = %e max(delta lnlike) = %e at t0 = %e\n",
    //        period, max_ll, max_phase);
}
