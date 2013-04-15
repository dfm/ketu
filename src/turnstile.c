#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
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

static char* load_kernel_source(const char *filename)
{
    struct stat statbuf;
    FILE *fh;
    char *source;

    fh = fopen(filename, "r");
    if (fh == 0)
        return 0;

    stat(filename, &statbuf);
    source = (char*)malloc(statbuf.st_size + 1);
    fread(source, statbuf.st_size, 1, fh);
    source[statbuf.st_size] = '\0';

    return source;
}

int fit_gpu (dataset *data)
{
    int i, N = data->length;
    float *time = data->time, *flux = data->flux, *ivar = data->ivar;

    // Compute min and max times.
    float tmin = -1.0, tmax = -1.0, delta;
    for (i = 0; i < N; ++i) {
        if (tmin < 0.0 || tmin > time[i]) tmin = time[i];
        if (tmax < 0.0 || tmax < time[i]) tmax = time[i];
    }
    fprintf(stdout, "t_min = %f, t_max = %f\n", tmin, tmax);
    delta = 1 / (tmax - tmin);

    // Fix the duration for speed.
    float duration = 0.1;

    // Make the grid of periods.
    float min_period = 0.8, max_period = 0.9, dperiod = 1 + duration * delta;
    int nperiods = (int)((max_period - min_period) / dperiod);
    float *periods = malloc(nperiods * sizeof(float));

    // Set up the GPU.
    int err;
    cl_device_id device_id;
    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to connect to GPU\n");
        return EXIT_FAILURE;
    }

    // Get the maximum work group size.
    size_t wg_size;
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                          sizeof(size_t), &wg_size, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to get device info\n");
        return EXIT_FAILURE;
    }
    fprintf(stdout, "Maximum work group size: %d\n", (int)wg_size);

    // Load the kernel source.
    const char *filename = "kernels/reduce_test.cl",
               *source = load_kernel_source(filename);
    if (!source) {
        fprintf(stderr, "Error: Failed to load source from file\n");
        return EXIT_FAILURE;
    }

    // Create a compute context...
    cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context || err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to create a compute context\n");
        return EXIT_FAILURE;
    }

    // ...and a command queue.
    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (!queue || err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to create a command queue\n");
        return EXIT_FAILURE;
    }

    // Allocate some memory on the device.
    cl_mem time_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        N * sizeof(float), NULL, NULL),
           flux_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        N * sizeof(float), NULL, NULL),
           periods_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        nperiods * sizeof(float), NULL, NULL),
           results_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        nperiods * sizeof(float), NULL, NULL);
    if (!time_buffer || !flux_buffer || !periods_buffer || !results_buffer) {
        fprintf(stderr, "Error: Failed to allocate buffers on device\n");
        return EXIT_FAILURE;
    }

    // Copy the data and period grid over.
    err = CL_SUCCESS;
    err |= clEnqueueWriteBuffer(queue, time_buffer, CL_TRUE, 0,
                                N * sizeof(float), time, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, flux_buffer, CL_TRUE, 0,
                                N * sizeof(float), flux, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to copy data\n");
        return EXIT_FAILURE;
    }

    // Create the program.
    cl_program program = clCreateProgramWithSource(context, 1,
                                                   (const char**)&source,
                                                   NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to create a program\n");
        return EXIT_FAILURE;
    }

    // Build the program.
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t length;
        char build_log[2048];
        fprintf(stderr, "%s\n", source);
        fprintf(stderr, "Error: Failed to build program\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                sizeof(build_log), build_log, &length);
        fprintf(stderr, "%s\n", build_log);
        return EXIT_FAILURE;
    }

    // Compile the kernel.
    cl_kernel kernel = clCreateKernel(program, "fit_depths", &err);
    if (!kernel || err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to compile kernel\n");
        return EXIT_FAILURE;
    }



    // Clean up.
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(time_buffer);
    clReleaseMemObject(flux_buffer);
    clReleaseMemObject(periods_buffer);
    clReleaseMemObject(results_buffer);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(periods);

    return 0;
}

int main()
{
    const char *filename = "data.fits";

    dataset *data = read_kepler_lc(filename);
    printf("%d ", data->length);
    mask_dataset(data);
    printf("%d\n", data->length);
    int val = fit_dataset(data);
    free_dataset(data);

    return val;
}
