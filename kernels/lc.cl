__kernel void lc (float epoch, float period, float duration, float depth,
                  __global float *times, __global float *lc)
{
    int i = get_global_id(0);
    if (fmod(times[i], period) - epoch <= duration)
        lc[i] = 1.0f - depth;
    else
        lc[i] = 1.0f;
}
