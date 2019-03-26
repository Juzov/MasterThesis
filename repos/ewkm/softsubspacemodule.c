#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

#include "ewkm.h"

static char softsubspace_docstring[] =
    "Soft Subspace Weighing Methods";
static char ewkm_docstring[] =
    "Entropy-Weighted-K-means";

static PyObject*
softsubspace_ewkm(PyObject * self, PyObject *args)
{

    /* inputs */

    PyObject 	*x_ptr = NULL;

    int 	nr,
		nc,
		k,
		maxiter,
		maxrestart,
		init;

    double 	lambda,
	   	delta;

    /* outputs */

    PyObject 	*cluster_ptr = NULL,
	     	*centers_ptr = NULL,
	     	*weights_ptr = NULL;

    int 	iterations,
		restarts,
		totiters;


    /* parse objects from args */

    if(!PyArg_ParseTuple(
			args,
			"OiiididiiiOOOii",
			&x_ptr,
			&nr,
			&nc,
			&k,
			&lambda,
			&maxiter,
			&delta,
			&maxrestart,
			&init,
			&iterations,
			&cluster_ptr,
			&centers_ptr,
			&weights_ptr,
			&restarts,
			&totiters
			)
	)
    {
	PyErr_BadArgument();
	return NULL;
    }

    /* Interpret the objects as numpy arrays */
    /* 3rd option tells if we want to write or only read to array */

    PyObject *np_x = PyArray_FROM_OTF(x_ptr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    PyObject *np_cluster = PyArray_FROM_OTF(cluster_ptr, NPY_INT, NPY_ARRAY_INOUT_ARRAY);
    PyObject *np_centers = PyArray_FROM_OTF(centers_ptr, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    PyObject *np_weights = PyArray_FROM_OTF(weights_ptr, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);

    if( np_x == NULL || np_cluster == NULL || np_centers == NULL || np_weights == NULL ){
        Py_XDECREF(np_x);
        Py_XDECREF(np_cluster);
        Py_XDECREF(np_centers);
        Py_XDECREF(np_weights);
        return NULL;
    }

    /* treat the arrays as c arrays */

    double *x = (double*)PyArray_DATA(np_x);

    int *cluster = (int*)PyArray_DATA(np_cluster);
    double *centers = (double*)PyArray_DATA(np_centers);
    double *weights = (double*)PyArray_DATA(np_weights);

    /* do the actual work */

    ewkm(
		x, 		// Numeric matrix as vector by col (nr*nc)
		&nr, 		// Number of rows (points)
		&nc, 		// Number of columns (attributes/variables)
		&k, 		// Number of clusters
		&lambda, 	// Learning rate
		&maxiter, 	// Maximum number of iterations
		&delta, 	// Minimum change below which iteration stops
		&maxrestart,      // Maximum number of restarts
		&init,            // Initial k prototypes.
		/* intputs --------------------------------------------------------- */
		&iterations,	// Number of iterations
		cluster, 	// Cluster assignment for each obs (nr) ()
		centers, 	// Cluster centers (k*nc) (each cluster has a vector of feature dim)
		weights, 	// Variable weights (k*nc) (clusters x variables)
		&restarts,	// Number of restarts (meh why) --- irrelevant
		&totiters	// Number of iterations including restarts --- irrelevant
    );

    /* release arrays */
    Py_DECREF(np_x);
    Py_DECREF(np_cluster);
    Py_DECREF(np_centers);
    Py_DECREF(np_weights);

    /* do not return anything explicitly */

    Py_INCREF(Py_None);
    return Py_None;
    /* return NULL; */
}

static PyMethodDef SoftSubspaceMethods[] = {
    //METH_VARARGS tells the function should expect Python-level parameters
    //0 means that an variant of PyArg_ParseTuple is used
    {"ewkm", softsubspace_ewkm, METH_VARARGS, ewkm_docstring},

    {NULL, NULL, 0, NULL} //Sentinel
};

//module definition structure
static struct PyModuleDef softsubspacemodule = {
    PyModuleDef_HEAD_INIT,
    "softsubspace", //name of module
    softsubspace_docstring, // module documentation
    -1, // keeps state in global variables
	SoftSubspaceMethods
};

// PyMODINIT_FUNC declares the function as PyObject *
PyMODINIT_FUNC
PyInit_softsubspace(void)
{
    import_array();
    return PyModule_Create(&softsubspacemodule);
}


