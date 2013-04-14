#ifndef GROUP_SIZE
#define GROUP_SIZE (64)
#endif

#ifndef OPERATIONS
#define OPERATIONS (1)
#endif

#define LOAD_GLOBAL_F1(s, i) \
    ((__global const float*)(s))[(size_t)(i)]

#define STORE_GLOBAL_F1(s, i, v) \
    ((__global float*)(s))[(size_t)(i)] = (v)

#define LOAD_LOCAL_F1(s, i) \
    ((__local float*)(s))[(size_t)(i)]

#define STORE_LOCAL_F1(s, i, v) \
    ((__local float*)(s))[(size_t)(i)] = (v)

#define ACCUM_LOCAL_F1(s, i, j) \
{ \
    float x = ((__local float*)(s))[(size_t)(i)]; \
    float y = ((__local float*)(s))[(size_t)(j)]; \
    ((__local float*)(s))[(size_t)(i)] = (x + y); \
}

__kernel void reduce(__global float *output,
                     __global const float *input,
                     __local float *shared,
                     const unsigned int n)
{
    const float zero = 0.0f;
    const unsigned int group_id = get_global_id(0) / get_local_size(0);
    const unsigned int group_size = GROUP_SIZE;
    const unsigned int group_stride = 2 * group_size;
    const size_t local_stride = group_stride * group_size;

    unsigned int op = 0;
    unsigned int last = OPERATIONS - 1;
    for(op = 0; op < OPERATIONS; op++) {
        const unsigned int offset = (last - op);
        const size_t local_id = get_local_id(0) + offset;

        STORE_LOCAL_F1(shared, local_id, zero);

        size_t i = group_id * group_stride + local_id;
        while (i < n) {
            float a = LOAD_GLOBAL_F1(input, i);
            float b = LOAD_GLOBAL_F1(input, i + group_size);
            float s = LOAD_LOCAL_F1(shared, local_id);
            STORE_LOCAL_F1(shared, local_id, (a + b + s));
            i += local_stride;
        }

    barrier(CLK_LOCAL_MEM_FENCE);
    #if (GROUP_SIZE >= 512)
        if (local_id < 256) { ACCUM_LOCAL_F1(shared, local_id, local_id + 256); }
    #endif

    barrier(CLK_LOCAL_MEM_FENCE);
    #if (GROUP_SIZE >= 256)
        if (local_id < 128) { ACCUM_LOCAL_F1(shared, local_id, local_id + 128); }
    #endif

    barrier(CLK_LOCAL_MEM_FENCE);
    #if (GROUP_SIZE >= 128)
        if (local_id <  64) { ACCUM_LOCAL_F1(shared, local_id, local_id +  64); }
    #endif

    barrier(CLK_LOCAL_MEM_FENCE);
    #if (GROUP_SIZE >= 64)
        if (local_id <  32) { ACCUM_LOCAL_F1(shared, local_id, local_id +  32); }
    #endif

    barrier(CLK_LOCAL_MEM_FENCE);
    #if (GROUP_SIZE >= 32)
        if (local_id <  16) { ACCUM_LOCAL_F1(shared, local_id, local_id +  16); }
    #endif

    barrier(CLK_LOCAL_MEM_FENCE);
    #if (GROUP_SIZE >= 16)
        if (local_id <   8) { ACCUM_LOCAL_F1(shared, local_id, local_id +   8); }
    #endif

    barrier(CLK_LOCAL_MEM_FENCE);
    #if (GROUP_SIZE >= 8)
        if (local_id <   4) { ACCUM_LOCAL_F1(shared, local_id, local_id +   4); }
    #endif

    barrier(CLK_LOCAL_MEM_FENCE);
    #if (GROUP_SIZE >= 4)
        if (local_id <   2) { ACCUM_LOCAL_F1(shared, local_id, local_id +   2); }
    #endif

    barrier(CLK_LOCAL_MEM_FENCE);
    #if (GROUP_SIZE >= 2)
        if (local_id <   1) { ACCUM_LOCAL_F1(shared, local_id, local_id +   1); }
    #endif

    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0) {
        float v = LOAD_LOCAL_F1(shared, 0);
        STORE_GLOBAL_F1(output, group_id, v);
    }
}
