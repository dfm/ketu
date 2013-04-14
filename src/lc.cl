__kernel void chi2_kernel (float t0, float P, float v, float r, float bmin,
                           __global float4 *data, __global float *chi)
{
    int i = get_global_id(0);
    float b, area, tm, c;

    tm = v * (fmod(data[i].x, P) - t0);
    b = sqrt(tm * tm + bmin * bmin);

    if (b >= 1 + r)
        area = 0.0;
    else if (b <= 1 - r)
        area = r * r;
    else if (b <= r - 1)
        area = 1.0;
    else {
        float p2 = r * r, b2 = b * b,
              k1 = acos(0.5 * (b2 + p2 - 1) / b / r),
              k2 = acos(0.5 * (b2 + 1 - p2) / b),
              k3 = sqrt((r+1-b) * (b+r-1) * (b-r+1) * (b+1+r));
        area = (p2 * k1 + k2 - 0.5 * k3) / M_PI;
    }

    c = (1.0 - area - data[i].y) / data[i].z;
    chi[i] = chi * chi;
}
