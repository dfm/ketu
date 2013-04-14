#include <libc.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif

#define MAX_GROUPS      (64)
#define MAX_WORK_ITEMS  (64)

void create_reduction_pass_counts(int count, int max_group_size,
                                  int max_groups, int max_work_items,
                                  int *pass_count, size_t **group_counts,
                                  size_t **work_item_counts,
                                  int **operation_counts, int **entry_counts)
{
    int work_items = (count < max_work_items * 2) ? count / 2 : max_work_items;
    if(count < 1)
        work_items = 1;

    int groups = count / (work_items * 2);
    groups = max_groups < groups ? max_groups : groups;

    int max_levels = 1;
    int s = groups;

    while(s > 1) {
        int work_items = (s < max_work_items * 2) ? s / 2 : max_work_items;
        s = s / (work_items*2);
        max_levels++;
    }

    *group_counts = (size_t*)malloc(max_levels * sizeof(size_t));
    *work_item_counts = (size_t*)malloc(max_levels * sizeof(size_t));
    *operation_counts = (int*)malloc(max_levels * sizeof(int));
    *entry_counts = (int*)malloc(max_levels * sizeof(int));

    (*pass_count) = max_levels;
    (*group_counts)[0] = groups;
    (*work_item_counts)[0] = work_items;
    (*operation_counts)[0] = 1;
    (*entry_counts)[0] = count;
    if(max_group_size < work_items) {
        (*operation_counts)[0] = work_items;
        (*work_item_counts)[0] = max_group_size;
    }

    s = groups;
    int level = 1;

    while (s > 1) {
        int work_items = (s < max_work_items * 2) ? s / 2 : max_work_items;
        int groups = s / (work_items * 2);
        groups = (max_groups < groups) ? max_groups : groups;

        (*group_counts)[level] = groups;
        (*work_item_counts)[level] = work_items;
        (*operation_counts)[level] = 1;
        (*entry_counts)[level] = s;
        if(max_group_size < work_items) {
            (*operation_counts)[level] = work_items;
            (*work_item_counts)[level] = max_group_size;
        }

        s = s / (work_items*2);
        level++;
    }
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

int main()
{
    int i, err;

    // Initialize the data.
    int ndata = 1024;
    float *data = (float*)malloc(ndata * sizeof(float)),
          truth = 0.0;
    for (i = 0; i < ndata; ++i) {
        data[i] = (float)i;
        truth += data[i];
    }

    cl_device_id device_id;
    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to connect to GPU\n");
        return EXIT_FAILURE;

        // Fall back on the CPU.
        // err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
        // if (err != CL_SUCCESS) {
        //     fprintf(stderr, "Error: Failed to connect to CPU\n");
        //     return EXIT_FAILURE;
        // }
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

    // Copy over the data.
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                         ndata * sizeof(float), NULL, NULL);
    if (!input_buffer) {
        fprintf(stderr, "Error: Failed to allocate input buffer on device\n");
        return EXIT_FAILURE;
    }
    err = clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0,
                               ndata * sizeof(float), data, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to copy data\n");
        return EXIT_FAILURE;
    }

    // Set up the scratch and results buffers.
    cl_mem partials_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                            ndata * sizeof(float), NULL, NULL);
    if (!partials_buffer) {
        fprintf(stderr, "Error: Failed to allocate partial sum buffer\n");
        return EXIT_FAILURE;
    }

    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          ndata * sizeof(float), NULL, NULL);
    if (!output_buffer) {
        fprintf(stderr, "Error: Failed to allocate output buffer\n");
        return EXIT_FAILURE;
    }

    // Figure out the reduction tree configuration.
    int pass_count;
    size_t *group_counts, *work_item_counts;
    int *operation_counts, *entry_counts;
    create_reduction_pass_counts(ndata, wg_size, MAX_GROUPS,
                                 MAX_WORK_ITEMS, &pass_count, &group_counts,
                                 &work_item_counts, &operation_counts,
                                 &entry_counts);

    // Each level of the reduction should have a custom program.
    cl_program *programs = (cl_program*)malloc(pass_count * sizeof(cl_program));
    cl_kernel *kernels = (cl_kernel*)malloc(pass_count * sizeof(cl_kernel));

    for (i = 0; i < pass_count; ++i) {
        char *block_source = malloc(strlen(source) + 1024);
        size_t source_length = strlen(source) + 1024;

        const char group_size_macro[] = "#define GROUP_SIZE";
        const char operations_macro[] = "#define OPERATIONS";
        sprintf(block_source, "%s (%d) \n%s (%d)\n\n%s\n",
                group_size_macro, (int)group_counts[i],
                operations_macro, (int)operation_counts[i],
                source);

        // Initialize the program.
        programs[i] = clCreateProgramWithSource(context, 1,
                                                (const char**)&block_source,
                                                NULL, &err);
        if (!programs[i] || err != CL_SUCCESS) {
            fprintf(stderr, "%s\n", block_source);
            fprintf(stderr, "Error: Failed to create compute program\n");
            return EXIT_FAILURE;
        }

        // Build the program.
        err = clBuildProgram(programs[i], 0, NULL, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t length;
            char build_log[2048];
            fprintf(stderr, "%s\n", block_source);
            fprintf(stderr, "Error: Failed to build program\n");
            clGetProgramBuildInfo(programs[i], device_id, CL_PROGRAM_BUILD_LOG,
                                  sizeof(build_log), build_log, &length);
            fprintf(stderr, "%s\n", build_log);
            return EXIT_FAILURE;
        }

        // Compile the kernel.
        kernels[i] = clCreateKernel(programs[i], "reduce", &err);
        if (!kernels[i] || err != CL_SUCCESS) {
            fprintf(stderr, "Error: Failed to compile kernel\n");
            return EXIT_FAILURE;
        }

        free(block_source);
    }

    // Execute the reduction.
    cl_mem pass_swap, pass_input = output_buffer, pass_output = input_buffer;
    for (i = 0; i < pass_count; ++i) {
        size_t global = group_counts[i] * work_item_counts[i];
        size_t local = work_item_counts[i];
        unsigned int operations = operation_counts[i];
        unsigned int entries = entry_counts[i];
        size_t shared_size = sizeof(float) * local * operations;

        fprintf(stdout, "Pass[%4d] Global[%4d] Local[%4d] Groups[%4d] ",
                i, (int)global, (int)local, (int)group_counts[i]);
        fprintf(stdout, "WorkItems[%4d] Operations[%d] Entries[%d]\n",
                (int)work_item_counts[i], operations, entries);

        // Swap the inputs and outputs for each pass
        pass_swap = pass_input;
        pass_input = pass_output;
        pass_output = pass_swap;

        // Set the kernel arguments.
        err = CL_SUCCESS;
        err |= clSetKernelArg(kernels[i], 0, sizeof(cl_mem), &pass_output);
        err |= clSetKernelArg(kernels[i], 1, sizeof(cl_mem), &pass_input);
        err |= clSetKernelArg(kernels[i], 2, shared_size,    NULL);
        err |= clSetKernelArg(kernels[i], 3, sizeof(int),    &entries);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error: Failed to set kernel arguments\n");
            return EXIT_FAILURE;
        }

        // After the first pass, use the partial sums for the next input
        // values.
        if(pass_input == input_buffer)
            pass_input = partials_buffer;

        err = CL_SUCCESS;
        err |= clEnqueueNDRangeKernel(queue, kernels[i], 1, NULL, &global,
                                      &local, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error: Failed to execute kernel\n");
            return EXIT_FAILURE;
        }
    }

    // Wait for computation to finish.
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        fprintf(stderr,
                "Error: Failed to wait for command queue to finish %d\n", err);
        return EXIT_FAILURE;
    }

    float result;
    err = clEnqueueReadBuffer(queue, pass_output, CL_TRUE, 0,
                              sizeof(float), &result, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to read back result from the device\n");
        return EXIT_FAILURE;
    }

    fprintf(stdout, "Result: %f\nTruth: %f\n", result, truth);

    // Clean up.
    for (i = 0; i < pass_count; ++i) {
        clReleaseKernel(kernels[i]);
        clReleaseProgram(programs[i]);
    }
    free(kernels);
    free(programs);

    free(group_counts);
    free(work_item_counts);
    free(operation_counts);
    free(entry_counts);

    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(partials_buffer);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
