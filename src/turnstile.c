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
    int i, j, N = data->length;
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

    // Set up the GPU.
    int err;
    cl_device_id device_id;
    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    printf("sup\n");
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

    // Set up the grid.
    int nperiods = 64, nepochs = 64;
    float min_period = 0.4, max_period = 100.0, dperiod,
          *periods = malloc(nperiods * sizeof(float));

    dperiod = (log(max_period) - log(min_period)) / nperiods;
    for (i = 0; i < nperiods; ++i) {
        periods[i] = exp(log(min_period) + i * dperiod);
    }


    // Load the kernel source.
    const char *filename = "kernels/turnstile.cl",
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
    cl_mem time_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                        N * sizeof(float), NULL, NULL),
           flux_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                        N * sizeof(float), NULL, NULL),
           periods_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                        nperiods * sizeof(float), NULL, NULL),
           results_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                            nepochs * nperiods * sizeof(float), NULL, NULL);
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
    err |= clEnqueueWriteBuffer(queue, periods_buffer, CL_TRUE, 0,
                            nperiods * sizeof(float), periods, 0, NULL, NULL);
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
    cl_kernel kernel = clCreateKernel(program, "turnstile_kernel", &err);
    if (!kernel || err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to compile kernel\n");
        return EXIT_FAILURE;
    }

    // Set the arguments.
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &time_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &flux_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &N);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &periods_buffer);
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &nperiods);
    err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &nepochs);
    err |= clSetKernelArg(kernel, 6, sizeof(float), &duration);
    err |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &results_buffer);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to set arguments\n");
        return EXIT_FAILURE;
    }

    // Run the kernel.
    size_t global = nperiods * nepochs, local = nepochs;
    printf("Global: %d Local: %d\n", (int)global, (int)local);
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0,
                                 NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to run kernel (%d)\n", err);
        return EXIT_FAILURE;
    }
    clFinish(queue);

    // Read the data back in.
    float *depths = malloc(nepochs * nperiods * sizeof(float));
    for (i = 0; i < nperiods; ++i)
        depths[i] = (float)i;
    err = clEnqueueReadBuffer(queue, results_buffer, CL_TRUE, 0,
                              nperiods * nepochs * sizeof(float), depths, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to read results (%d)\n", err);
        return EXIT_FAILURE;
    }

    for (i = 0; i < nperiods; ++i)
        for (j = 0; j < nepochs; ++j)
            printf("%f %f\n", periods[i], depths[i * nepochs + j]);

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
    free(depths);

    return 0;
}

int main()
{
    const char *filename = "data.fits";
    int val = 0;

    dataset *data = read_kepler_lc(filename);
    mask_dataset(data);
    /* val = fit_dataset(data); */
    /* int val = fit_gpu(data); */
    free_dataset(data);

    return val;
}
