#include <stdio.h>
#include <string.h>
#include "kepler.h"

dataset *init_dataset(int length)
{
    dataset *self = (dataset*)malloc(length * sizeof(dataset));
    self->length = length;
    self->time = (float*)malloc(length * sizeof(float));
    self->flux = (float*)malloc(length * sizeof(float));
    self->ferr = (float*)malloc(length * sizeof(float));
    self->ivar = (float*)malloc(length * sizeof(float));
    return self;
}

void free_dataset(dataset *self)
{
    free(self->time);
    free(self->flux);
    free(self->ferr);
    free(self->ivar);
    free(self);
}

void print_fits_error(int status)
{
    if (status) {
        fits_report_error(stderr, status);
        exit(status);
    }
}

dataset *read_kepler_lc (const char *filename)
{
    fitsfile *f;
    int status = 0;

    // Open the file.
    if (fits_open_file(&f, filename, READONLY, &status))
        print_fits_error(status);

    // Choose the HDU.
    int hdutype;
    if (fits_movabs_hdu(f, 2, &hdutype, &status))
        print_fits_error(status);

    // Check the type of the HDU.
    if (hdutype != ASCII_TBL && hdutype != BINARY_TBL) {
        printf("Error: this HDU is not an ASCII or binary table\n");
        exit(1);
    }

    // Get the number of columns.
    int ncols;
    if (fits_get_num_cols(f, &ncols, &status)) print_fits_error(status);

    // Load the column names.
    int i, nfound;
    char **labels = (char**)malloc(ncols * sizeof(char*));
    for (i = 0; i < ncols; ++i)
        labels[i] = (char*)malloc(FLEN_VALUE);
    fits_read_keys_str(f, "TTYPE", 1, ncols, labels, &nfound, &status);

    // Figure out which columns to use.
    int time_col = -1, flux_col = -1, ferr_col = -1;
    for (i = 0; i < ncols; ++i) {
        if (strcmp(labels[i], "TIME") == 0)
            time_col = i;
        else if (strcmp(labels[i], "FLUX") == 0)
            flux_col = i;
        else if (strcmp(labels[i], "FERR") == 0)
            ferr_col = i;
    }

    // Clean up the labels.
    for (i = 0; i < ncols; ++i)
        free(labels[i]);
    free(labels);

    // Check that all the columns exist.
    if (time_col < 0 || flux_col < 0 || ferr_col < 0) {
        fprintf(stderr, "Couldn't find the right columns\n");
        exit(2);
    }

    // Figure out the number of rows.
    long nrows;
    if (fits_get_num_rows(f, &nrows, &status)) print_fits_error(status);

    // Load the data.
    dataset *data = init_dataset(nrows);
    int anynull;
    float fill = -1.0;

    fits_read_col(f, TFLOAT, time_col + 1, 1, 1, nrows, &fill, data->time,
                  &anynull, &status);
    fits_read_col(f, TFLOAT, flux_col + 1, 1, 1, nrows, &fill, data->flux,
                  &anynull, &status);
    fits_read_col(f, TFLOAT, ferr_col + 1, 1, 1, nrows, &fill, data->ferr,
                  &anynull, &status);

    // Compute the inverse variances and mask bad data.
    for (i = 0; i < nrows; ++i) {
        float sig = data->ferr[i];
        if (sig <= 0.0) data->ivar[i] = 0.0;
        else data->ivar[i] = 1.0 / sig / sig;
    }

    // Close the file.
    if (fits_close_file(f, &status)) print_fits_error(status);

    // Return the dataset.
    return data;
}
