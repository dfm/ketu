#ifndef _TURNSTILE_H_
#define _TURNSTILE_H_

typedef struct turnstile_struct {

    int ndatasets;
    int *ndata;
    double duration;
    double **time;
    double **delta_lnlike;

} turnstile;

turnstile *turnstile_allocate (
    double duration,
    double depth,
    int ndatasets,
    int *ndata,
    double **time,
    double **flux,
    double **ferr,
    double *hyperpars
);
void turnstile_free (turnstile *self);

void turnstile_evaluate (
    double period,
    int nphases,
    double *phases,
    double *lnlikes,
    turnstile *self
);

#endif
// /_TURNSTILE_H_
