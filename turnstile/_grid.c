#include <Python.h>
#include <numpy/arrayobject.h>
#include "turnstile.h"

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

#define PARSE_ARRAY(o) (PyArrayObject*) PyArray_FROM_OTF(o, NPY_DOUBLE, \
        NPY_IN_ARRAY)

#define FAILURE {Py_DECREF(hyperpars_array); \
    for (j = 0; j <= i; ++j) { \
        Py_XDECREF(time_arrays[j]); \
        Py_XDECREF(flux_arrays[j]); \
        Py_XDECREF(ferr_arrays[j]); \
        Py_XDECREF(time_objs[j]); \
        Py_XDECREF(flux_objs[j]); \
        Py_XDECREF(ferr_objs[j]); \
    } \
    free(time_arrays); \
    free(flux_arrays); \
    free(ferr_arrays); \
    free(time_objs); \
    free(flux_objs); \
    free(ferr_objs); \
    free(ndata); \
    free(time); \
    free(flux); \
    free(ferr); \
    return NULL;}

#define GP_FAILURE {Py_DECREF(hyperpars_array); \
    Py_DECREF(periods_array); \
    Py_DECREF(durations_array); \
    for (j = 0; j < ndatasets; ++j) { \
        Py_DECREF(time_arrays[j]); \
        Py_DECREF(flux_arrays[j]); \
        Py_DECREF(ferr_arrays[j]); \
        Py_DECREF(time_objs[j]); \
        Py_DECREF(flux_objs[j]); \
        Py_DECREF(ferr_objs[j]); \
    } \
    free(time_arrays); \
    free(flux_arrays); \
    free(ferr_arrays); \
    free(time_objs); \
    free(flux_objs); \
    free(ferr_objs); \
    free(ndata); \
    free(time); \
    free(flux); \
    free(ferr); \
    return NULL;}

static char search_doc[] =
    "Do a freakin' grid search\n";

static PyObject
*grid_search (PyObject *self, PyObject *args)
{
    double duration, depth, fphase;
    PyObject *datasets, *periods_obj, *hyperpars_obj;
    if (!PyArg_ParseTuple(args, "OOdddO",
                          &datasets, &periods_obj, &duration, &depth,
                          &fphase, &hyperpars_obj))
        return NULL;

    // Pull out the hyperparameters object.
    PyArrayObject *hyperpars_array = PARSE_ARRAY(hyperpars_obj);
    if (hyperpars_array == NULL || PyArray_DIM(hyperpars_array, 0) != 3) {
        Py_XDECREF(hyperpars_array);
        return NULL;
    }
    double *hyperpars = (double*)PyArray_DATA(hyperpars_array);

    // Parse the periods.
    PyArrayObject *periods_array = PARSE_ARRAY(periods_obj);
    if (periods_array == NULL) {
        Py_DECREF(hyperpars_array);
        Py_XDECREF(periods_array);
        return NULL;
    }
    double *periods = (double*)PyArray_DATA(periods_array);
    int nperiods = PyArray_DIM(periods_array, 0);

    // Access all the data in the list of datasets.
    int i, j, ndatasets = (int)PyList_Size(datasets);
    int *ndata = malloc(ndatasets * sizeof(int));
    double **time = malloc(ndatasets * sizeof(double*)),
           **flux = malloc(ndatasets * sizeof(double*)),
           **ferr = malloc(ndatasets * sizeof(double*));
    PyObject **time_objs = malloc(ndatasets * sizeof(PyObject*)),
             **flux_objs = malloc(ndatasets * sizeof(PyObject*)),
             **ferr_objs = malloc(ndatasets * sizeof(PyObject*));
    PyArrayObject **time_arrays = malloc(ndatasets * sizeof(PyArrayObject)),
                  **flux_arrays = malloc(ndatasets * sizeof(PyArrayObject)),
                  **ferr_arrays = malloc(ndatasets * sizeof(PyArrayObject));

    for (i = 0; i < ndatasets; ++i) {
        // Parse the data arrays from the dataset object.
        PyObject *dataset = PyList_GetItem(datasets, i);

        // Pull out the attributes.
        time_objs[i] = PyObject_GetAttrString(dataset, "time");
        flux_objs[i] = PyObject_GetAttrString(dataset, "flux");
        ferr_objs[i] = PyObject_GetAttrString(dataset, "ferr");
        if (time_objs[i] == NULL || flux_objs[i] == NULL ||
            ferr_objs[i] == NULL) FAILURE;

        // Parse as numpy arrays.
        time_arrays[i] = PARSE_ARRAY(time_objs[i]);
        flux_arrays[i] = PARSE_ARRAY(flux_objs[i]);
        ferr_arrays[i] = PARSE_ARRAY(ferr_objs[i]);
        if (time_arrays[i] == NULL || flux_arrays[i] == NULL ||
            ferr_arrays[i] == NULL) FAILURE;

        // Compute the dimension.
        ndata[i] = (int)PyArray_DIM(time_arrays[i], 0);

        // Pull out the pointers to the data.
        time[i] = (double*)PyArray_DATA(time_arrays[i]);
        flux[i] = (double*)PyArray_DATA(flux_arrays[i]);
        ferr[i] = (double*)PyArray_DATA(ferr_arrays[i]);
    }

    // Set up turnstile.
    printf("Initializing %d Gaussian processes\n", ndatasets);
    turnstile *grid = turnstile_allocate (duration, depth, ndatasets, ndata,
                                          time, flux, ferr, hyperpars);

    // Allocate memory for the output.
    PyObject *ret,
             *ll_list = PyList_New(nperiods),
             *phase_list = PyList_New(nperiods);
    if (ll_list == NULL || phase_list == NULL) goto fail;

    // Loop over periods.
    int ind, nphases;
    npy_intp dim[1];
    double period;
    printf("Looping over %d periods\n", nperiods);
    for (ind = 0; ind < nperiods; ++ind) {
        // Compute the period.
        period = periods[ind];

        // Figure out the number of phases that we need to compute and
        // allocate the output array.
        nphases = period / (fphase * duration);
        dim[0] = nphases;
        PyArrayObject *ll_array =
            (PyArrayObject*)PyArray_SimpleNew(1, dim, NPY_DOUBLE),
                      *phase_array =
            (PyArrayObject*)PyArray_SimpleNew(1, dim, NPY_DOUBLE);
        if (ll_array == NULL || phase_array == NULL) {
            Py_XDECREF(ll_array);
            Py_XDECREF(phase_array);
            goto fail;
        }
        double *lnlike = (double*)PyArray_DATA(ll_array),
               *phases = (double*)PyArray_DATA(phase_array);

        // Fill the phases array.
        for (i = 0; i < nphases; ++i)
            phases[i] = i * period / nphases;

        turnstile_evaluate (period, nphases, phases, lnlike, grid);

        // Append the array to the output list.
        PyList_SetItem(ll_list, ind, (PyObject*)ll_array);
        PyList_SetItem(phase_list, ind, (PyObject*)phase_array);
    }

    turnstile_free(grid);

    // De-reference numpy objects.
    Py_DECREF(hyperpars_array);
    Py_DECREF(periods_array);
    for (j = 0; j < ndatasets; ++j) {
        Py_DECREF(time_arrays[j]);
        Py_DECREF(flux_arrays[j]);
        Py_DECREF(ferr_arrays[j]);
        Py_DECREF(time_objs[j]);
        Py_DECREF(flux_objs[j]);
        Py_DECREF(ferr_objs[j]);
    }
    free(time_arrays);
    free(flux_arrays);
    free(ferr_arrays);
    free(time_objs);
    free(flux_objs);
    free(ferr_objs);
    free(ndata);
    free(time);
    free(flux);
    free(ferr);

    ret = Py_BuildValue("OO", phase_list, ll_list);
    Py_DECREF(phase_list);
    Py_DECREF(ll_list);

    return ret;

fail:

    turnstile_free(grid);

    Py_DECREF(hyperpars_array);
    Py_XDECREF(periods_array);

    for (j = 0; j < ndatasets; ++j) {
        Py_DECREF(time_arrays[j]);
        Py_DECREF(flux_arrays[j]);
        Py_DECREF(ferr_arrays[j]);
        Py_DECREF(time_objs[j]);
        Py_DECREF(flux_objs[j]);
        Py_DECREF(ferr_objs[j]);
    }
    free(time_arrays);
    free(flux_arrays);
    free(ferr_arrays);
    free(time_objs);
    free(flux_objs);
    free(ferr_objs);
    free(ndata);
    free(time);
    free(flux);
    free(ferr);

    return NULL;
}

static PyMethodDef grid_methods[] = {
    {"search",
     (PyCFunction) grid_search,
     METH_VARARGS,
     search_doc},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int grid_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int grid_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_grid",
    NULL,
    sizeof(struct module_state),
    grid_methods,
    NULL,
    grid_traverse,
    grid_clear,
    NULL
};

#define INITERROR return NULL

PyObject *PyInit__grid(void)
#else
#define INITERROR return

void init_grid(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("_grid", grid_methods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    char err[] = "_grid.Error";
    st->error = PyErr_NewException(err, NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    import_array();

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
