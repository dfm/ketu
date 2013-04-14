#include <stdio.h>
#include "kepler.h"

#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

int main()
{
    const char *filename = "data/kplr011904151-2012179063303_llc.fits";

    dataset *data = read_kepler_lc(filename);

    int i;
    for (i = 0; i < data->length; ++i)
        printf("%f %f %f\n", data->time[i], data->flux[i], data->ivar[i]);

    free_dataset(data);

    return 0;
}
