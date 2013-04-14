#include <stdio.h>
#include <math.h>
#include "kepler.h"

#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

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

int main()
{
    const char *filename = "data/kplr011904151-2012179063303_llc-untrend.fits";

    dataset *data = read_kepler_lc(filename);

    int i;
    for (i = 0; i < data->length; ++i)
        printf("%f %f %f\n", data->time[i], data->flux[i], data->ivar[i]);

    free_dataset(data);

    return 0;
}
