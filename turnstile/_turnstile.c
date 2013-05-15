#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

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

    double *time = (double*)PyArray_DATA(time_array),
           *flux = (double*)PyArray_DATA(flux_array),
           *ivar = (double*)PyArray_DATA(ivar_array);

    printf("%f\n", time[0]);

    Py_DECREF(time_array);
    Py_DECREF(flux_array);
    Py_DECREF(ivar_array);

    PyObject *ret = Py_BuildValue("i", 0);
    return ret;
}

static PyMethodDef turnstile_methods[] = {
    {"find_discontinuities",
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
