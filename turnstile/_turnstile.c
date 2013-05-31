#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

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

static PyObject
*turnstile_find_periods(PyObject *self, PyObject *args)
{
    PyObject *time_obj, *flux_obj, *ivar_obj;
    double min_period, max_period, dperiod, tau;

    if (!PyArg_ParseTuple(args, "OOOdddd", &time_obj, &flux_obj, &ivar_obj,
                          &min_period, &max_period, &dperiod, &tau))
        return NULL;

    PyArrayObject *time_array = (PyArrayObject*)PyArray_FROM_OTF(time_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *flux_array = (PyArrayObject*)PyArray_FROM_OTF(flux_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ivar_array = (PyArrayObject*)PyArray_FROM_OTF(ivar_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (time_array == NULL || flux_array == NULL || ivar_array == NULL) {
        Py_XDECREF(time_array);
        Py_XDECREF(flux_array);
        Py_XDECREF(ivar_array);
        return NULL;
    }

    int n = (int)PyArray_DIM(time_array, 0);
    if ((int)PyArray_DIM(flux_array, 0) != n || (int)PyArray_DIM(ivar_array, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch");
        Py_DECREF(time_array);
        Py_DECREF(flux_array);
        Py_DECREF(ivar_array);
        return NULL;
    }

    // Wrap the data in a lightcurve object.
    lightcurve *data = malloc(sizeof(lightcurve));
    data->length = n;
    data->time = (double*)PyArray_DATA(time_array);
    data->flux = (double*)PyArray_DATA(flux_array);
    data->ivar = (double*)PyArray_DATA(ivar_array);

    // Loop over period and compute the best depth and epoch.
    int nperiods = (int)((max_period - min_period) / dperiod) + 1;
    npy_intp pdim[1] = {nperiods};

    PyArrayObject *period_array = (PyArrayObject*)PyArray_SimpleNew(1, pdim, NPY_DOUBLE),
                  *depth_array = (PyArrayObject*)PyArray_SimpleNew(1, pdim, NPY_DOUBLE),
                  *epoch_array = (PyArrayObject*)PyArray_SimpleNew(1, pdim, NPY_DOUBLE),
                  *chi2_array = (PyArrayObject*)PyArray_SimpleNew(1, pdim, NPY_DOUBLE);
    double *periods = (double*)PyArray_DATA(period_array),
           *depths = (double*)PyArray_DATA(depth_array),
           *epochs = (double*)PyArray_DATA(epoch_array),
           *chi2 = (double*)PyArray_DATA(chi2_array);

    double period;
    int i, eind, factor = 2;
    for (i = 0; i < nperiods; ++i) {
        period = periods[i] = min_period + i * dperiod;
        tau = 0.5 * exp(0.44 * log(period) - 2.97);
        lightcurve *folded = lightcurve_fold_and_bin(data, period,
                                                     tau / factor, 1);
        test_epoch(folded, factor, &(depths[i]), &eind);
        epochs[i] = (tau / factor) * eind;
        free(folded);

        // Compute chi^2 at the best depth + epoch.
        chi2[i] = compute_chi2(data, period, depths[i], epochs[i], tau);
    }

    Py_DECREF(time_array);
    Py_DECREF(flux_array);
    Py_DECREF(ivar_array);
    free(data);

    PyObject *ret = Py_BuildValue("OOOO", period_array, depth_array, epoch_array,
                                  chi2_array);
    return ret;
}

static PyMethodDef turnstile_methods[] = {
    {"find_periods",
     (PyCFunction)turnstile_find_periods,
     METH_VARARGS,
     "Find periods in a time series."},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int turnstile_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int turnstile_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_turnstile",
        NULL,
        sizeof(struct module_state),
        turnstile_methods,
        NULL,
        turnstile_traverse,
        turnstile_clear,
        NULL
};

#define INITERROR return NULL

PyObject *PyInit__turnstile(void)

#else
#define INITERROR return

void init_turnstile(void)

#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("_turnstile", turnstile_methods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("_turnstile.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    import_array();

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
