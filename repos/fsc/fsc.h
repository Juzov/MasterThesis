// Fuzzy Subspace Clustering

// Based on EKWM in CRAN's wskm package

/* #include "Utils.h" */

// ----- double *x---
// COLUMN MAJOR 2D ARRAY
// Represented as a 1D array
// ---- VAR   j -----
// +  ______________
// + |				|
// P | 0	3	6	|
//   |				|
// i | 1	4	7	|
// + |				|
// i | 2	5	8	|
// + |______________|
//
// ----- double *O_prototype---
// COLUMN MAJOR 2D ARRAY
// Represented as a 1D array
// ---- VAR   j -----
// +  ______________
// + |				|
// k | 0	3	6	|
//   |				|
// i | 1	4	7	|
// + |				|
// i | 2	5	8	|
// + |______________|

// ***** Support functions *****

// ----- Create the initial prototype (k centroids) -----
// prototype is a bad name, would literally mean that we are talking about mixed data
//

double unif_rand();

void initPrototypesPlusPlus(
		double *x,	// Numeric matrix as vector by col (nr*nc)
		int *nr, 	// Number of rows (points)
		int *nc, 	// Number of columns (attributes/variables)
		int *k,	// Number of clusters
		// Output ---------------------------------------------
		double *o_prototype); // Numeric prototype matrix (k*nc)

void initPrototypes( // Inputs ---------------------------------------------
		double *x,	// Numeric matrix as vector by col (nr*nc)
		int *nr, 	// Number of rows
		int *nc, 	// Number of columns
		int *k,	// Number of clusters
		// Output ---------------------------------------------
		double *o_prototype); // Numeric prototype matrix (k*nc)

// ----- Calculate the cluster dispersion (objective function) -----
double calcCost(double *x, 	// Numeric matrix as vector by col (nr*nc)
		int *nr, 	// Number of rows
		int *nc, 	// Number of columns
		int *k, 		// Number of clusters
		double *alpha,	// fuzziness
		double *epsilon,	// small constant
		int *partition, 	// Partition matrix (nr)
		double *o_prototype, // Numeric prototype matrix (k*nc)
		double *subspace_weights); // Weights for variable/cluster (k*nc)

void updPartition(  // Inputs
		double *x, 	// Numeric matrix as vector by col (nr*nc)
		int *nr, 	// Number of rows
		int *nc, 	// Number of columns
		int *k, 	// Number of clusters
		double *alpha, 	// fuzziness
		double *o_prototype, // Numeric prototype matrix (k*nc)
		double *subspace_weights, // Weights for variable/cluster (k*nc)
		// Output
		int *partition);	// Partition matrix (nr)

// --- Update the prototypes -----

int updPrototypes(  // Inputs ---------------------------------------------
		double *x, 	// Numeric matrix as vector by col (nr*nc)
		int *nr, 		// Number of rows
		int *nc, 		// Number of columns
		int *k, 		// Number of clusters
		int *partition, 	// Partition matrix (nr)
		// Output ---------------------------------------------
		double *o_prototype); // Numeric prototype matrix (k*nc)

// ----- Update subspace weights. -----

void updWeights( // Inputs -------------------------------------------------------
		double *x, 	// Numeric matrix as vector by col (nr*nc)
		int *nr, 	// Number of rows
		int *nc, 	// Number of columns
		int *k, 	// Number of clusters
		double *alpha, //fuzzyness index
		double *epsilon, //small constant
		int *partition,	// Partition matrix (nr)
		double *o_prototype, // Numeric prototype matrix (k*nc)
		// Output -------------------------------------------------------
		double *subspace_weights); // Weights for variable/cluster (k*nc)


double fsc( // Inputs ----------------------------------------------------------
		double *x, 		// Numeric matrix as vector by col (nr*nc)
		int *nr, 		// Number of rows
		int *nc, 		// Number of columns
		int *k, 		// Number of clusters
		double *alpha, 	// fuzziness
		int *maxiter, 	// Maximum number of iterations
		double *delta, 	// Minimum change below which iteration stops
		int *maxrestart,      // Maximum number of restarts
		int *init,            // Initial k prototypes.
		// Outputs ---------------------------------------------------------
		int *iterations,	// Number of iterations
		int *cluster, 	// Cluster assignment for each obs (nr)
		double *centers, 	// Cluster centers (k*nc)
		double *weights, 	// Variable weights (k*nc)
		int *restarts,	// Number of restarts
		int *totiters);	// Number of iterations including restarts
