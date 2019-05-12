#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

#include "lekm.h"
#include "utils.h"

static char softsubspace_docstring[] =
    "Soft Subspace Weighing Methods";
static char distances_docstring[] =
    "Get the distance for each point to the center of the cluster";
static char point_distances_docstring[] =
    "Get the point distance matrix calculated with respect to the subpace weights";
static char lekm_docstring[] =
    "Entropy-Weighted-K-means";

static PyObject*
softsubspace_lekm(PyObject * self, PyObject *args)
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
	     	*weights_ptr = NULL,
		*ret = NULL;

    int 	iterations,
		restarts,
		totiters;


    double	dispersion;
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

    double *x = (double*)PyArray_DATA((PyArrayObject *)np_x);

    int *cluster = (int*)PyArray_DATA((PyArrayObject *)np_cluster);
    double *centers = (double*)PyArray_DATA((PyArrayObject *)np_centers);
    double *weights = (double*)PyArray_DATA((PyArrayObject *)np_weights);

    /* do the actual work */

    dispersion = lekm(
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

    ret = Py_BuildValue("diii", dispersion, iterations, restarts, totiters);
    /* printf("iterations: %d totiters: %d", iterations, totiters); */
    return ret;
    /* return NULL; */
}

static PyObject*
softsubspace_distances(PyObject * self, PyObject *args)
{

    /* inputs */

    PyObject 	*x_ptr = NULL,
		*cluster_ptr = NULL,
	     	*centers_ptr = NULL,
	     	*weights_ptr = NULL;

    int 	nr,
		nc,
		k;

    /* outputs */

    PyObject	*ret = NULL;

    if(!PyArg_ParseTuple(
			args,
			"OiiiOOO",
			&x_ptr,
			&nr,
			&nc,
			&k,
			&cluster_ptr,
			&centers_ptr,
			&weights_ptr
			)
	)
    {
	PyErr_BadArgument();
	return NULL;
    }

    npy_intp dims[1];
    dims[0] = nr;
    // 2 dimensions / dimension sizes / type
    PyObject *np_distances = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    /* Interpret the objects as numpy arrays */
    /* 3rd option tells if we want to write or only read to array */

    PyObject *np_x = PyArray_FROM_OTF(x_ptr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    PyObject *np_cluster = PyArray_FROM_OTF(cluster_ptr, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyObject *np_centers = PyArray_FROM_OTF(centers_ptr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyObject *np_weights = PyArray_FROM_OTF(weights_ptr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if( np_x == NULL || np_cluster == NULL || np_centers == NULL || np_weights == NULL ){
        Py_XDECREF(np_x);
        Py_XDECREF(np_cluster);
        Py_XDECREF(np_centers);
        Py_XDECREF(np_weights);
        return NULL;
    }

    /* treat the arrays as c arrays */

    double *x = (double*)PyArray_DATA((PyArrayObject *)np_x);

    int *cluster = (int*)PyArray_DATA((PyArrayObject *)np_cluster);
    double *centers = (double*)PyArray_DATA((PyArrayObject *)np_centers);
    double *weights = (double*)PyArray_DATA((PyArrayObject *)np_weights);
    double *distance = (double*)PyArray_DATA((PyArrayObject *)np_distances);

    /* do the actual work */
    distances(
		x, 		// Numeric matrix as vector by col (nr*nc)
		&nr, 		// Number of rows (points)
		&nc, 		// Number of columns (attributes/variables)
		&k, 		// Number of clusters
		cluster, 	// Cluster assignment for each obs (nr) ()
		centers, 	// Cluster assignment for each obs (nr) ()
		weights, 	// Variable weights (k*nc) (clusters x variables)
		distance
    );

    ret = (PyObject *) np_distances;
    np_distances = NULL;
    /* release arrays */
    Py_DECREF(np_x);
    Py_DECREF(np_cluster);
    Py_DECREF(np_centers);
    Py_DECREF(np_weights);
    Py_XDECREF(np_distances);

    /* printf("iterations: %d totiters: %d", iterations, totiters); */
    return ret;
    /* return NULL; */
}

static PyObject*
softsubspace_point_distances(PyObject * self, PyObject *args)
{

    /* inputs */

    PyObject 	*x_ptr = NULL,
		*cluster_ptr = NULL,
	     	*weights_ptr = NULL;

    int 	nr,
		nc,
		k;

    /* outputs */

    PyObject	*ret = NULL;

    if(!PyArg_ParseTuple(
			args,
			"OiiiOO",
			&x_ptr,
			&nr,
			&nc,
			&k,
			&cluster_ptr,
			&weights_ptr
			)
	)
    {
	PyErr_BadArgument();
	return NULL;
    }

    npy_intp dims[2];
    dims[0] = nr;
    dims[1] = nr;
    // 2 dimensions / dimension sizes / type
    PyObject *np_point_distances = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    /* Interpret the objects as numpy arrays */
    /* 3rd option tells if we want to write or only read to array */

    PyObject *np_x = PyArray_FROM_OTF(x_ptr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    PyObject *np_cluster = PyArray_FROM_OTF(cluster_ptr, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyObject *np_weights = PyArray_FROM_OTF(weights_ptr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if( np_x == NULL || np_cluster == NULL || np_weights == NULL ){
        Py_XDECREF(np_x);
        Py_XDECREF(np_cluster);
        Py_XDECREF(np_weights);
        return NULL;
    }

    /* treat the arrays as c arrays */

    double *x = (double*)PyArray_DATA((PyArrayObject *)np_x);

    int *cluster = (int*)PyArray_DATA((PyArrayObject *)np_cluster);
    double *weights = (double*)PyArray_DATA((PyArrayObject *)np_weights);
    double *point_distance = (double*)PyArray_DATA((PyArrayObject *)np_point_distances);

    /* do the actual work */
    point_distances(
		x, 		// Numeric matrix as vector by col (nr*nc)
		&nr, 		// Number of rows (points)
		&nc, 		// Number of columns (attributes/variables)
		&k, 		//
		cluster, 	// Cluster assignment for each obs (nr) ()
		weights, 	// Variable weights (k*nc) (clusters x variables)
		point_distance
    );

    ret = (PyObject *) np_point_distances;
    np_point_distances = NULL;
    /* release arrays */
    Py_DECREF(np_x);
    Py_DECREF(np_cluster);
    Py_DECREF(np_weights);
    Py_XDECREF(np_point_distances);

    /* printf("iterations: %d totiters: %d", iterations, totiters); */
    return ret;
    /* return NULL; */
}
static PyMethodDef SoftSubspaceMethods[] = {
    //METH_VARARGS tells the function should expect Python-level parameters
    //0 means that an variant of PyArg_ParseTuple is used
    {"lekm", softsubspace_lekm, METH_VARARGS, lekm_docstring},
    {"distances", softsubspace_distances, METH_VARARGS, distances_docstring},
    {"point_distances", softsubspace_point_distances, METH_VARARGS, point_distances_docstring},

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


