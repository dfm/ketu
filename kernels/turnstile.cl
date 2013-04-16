__kernel void turnstile_kernel (
                                __global float *time,
                                __global float *flux,
                                const unsigned int ndata,
                                __global float *periods,
                                const unsigned int nperiods,
                                const unsigned int nepochs,
                                const float duration,
                                __global float *depths
                               )
{
    int i = get_global_id(0),
        j = get_local_id(0);
    if (i < nperiods * nepochs) {
        float maxdepth = 0.0, epoch, period = periods[(i - j) / nepochs],
              fmin = 0.0, fmax = 0.0, t, tnorm;
        int k, nmin = 0, nmax = 0;

        epoch = (float)j * period / (float)nepochs;

        for (k = 0; k < ndata; ++k) {
            t = time[k] - epoch;
            tnorm = t / period - (int)(t / period);
            if (tnorm < duration) {
                fmin += flux[k];
                nmin++;
            } else {
                fmax += flux[k];
                nmax++;
            }
        }

        depths[i] = fmax / (float)nmax - fmin / (float)nmin;
    }
}
