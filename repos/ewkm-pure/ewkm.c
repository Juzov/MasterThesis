// Entropy Weighted K-Means

// Original C code for k-means by Joshua Huang. Qiang Wang implemented
// the entropy weighted algorithm. Xiaojun prepared the code for
// packaging, and Graham Williams finalised the code for release and
// maintains the package.

// Copyright (c) 2011 Shenzhen Institutes of Advanced Technology
// Chinese Academny of Sciences

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

//TODO enable it for R
/* #include <R.h> */
#include <math.h>
#include <float.h>
#include <ctype.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <stdio.h>
#include <stdio.h>
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

double unif_rand(){
	return (double)rand() / (double)RAND_MAX;
}

void initPrototypes( // Inputs ---------------------------------------------
		double *x,	// Numeric matrix as vector by col (nr*nc)
		int *nr, 	// Number of rows (points)
		int *nc, 	// Number of columns (attributes/variables)
		int *k,	// Number of clusters
		// Output ---------------------------------------------
		double *o_prototype) // Numeric prototype matrix (k*nc)
{
	int i, j, l; // i, 0< j <nc ? 0 < l < k
	int flag = 0;
	int index;

	int *random_obj_num; 	// Array for randomly selected objects (k)

	// Memory for array of randomly selected objects size k

	random_obj_num = (int *) malloc(sizeof(int) * (*k));
	if (!random_obj_num) {
		fprintf(stderr, "Can't allocate memory for random_obj_num matrix\n");
		exit(-1);
	}

	// Memory for array of randomly selected objects size k
	for (l = 0; l < *k; l++)
		random_obj_num[l] = -1;

	// Randomly select k objects.

	// for 0:k
	for (l = 0; l < *k; l++) {
		flag = 1;

		while (flag) {
		  //    index = (int) (rand() % (*nr));
		  //    R function
		  //    value range [0,1]
		  //    * number of 
		  //    random point index
		  //    *nr-1
		  //    i.e. nr-1 = 9 unif_rand 0.5 9*0.5 = 4.5 = 4?
			index = (int) (*nr-1) * unif_rand();
			flag = 0;
			// for 0:l
			// check that previous randomly selected  are not the same as the randomly chosen one
			// if it is find a new random point
			for (i = 0; i < l; i++)
				if (random_obj_num[i] == index)
					flag = 1;
		}

		// random obj l = index
		random_obj_num[l] = index;
		// go through all points?
		// take all attibutes from for index
		// put them in the lth row of o_prototype
		for (j = 0; j < (*nc); j++)
			// go through
			// jth attribute times number of points + picked point
			// 3 * 10 + 5
			// 5
			// 15
			// 25
			// 35
			// it has to be a column major "2-d array"
			// the lth random sample / row of o_prototype
			o_prototype[j * (*k) + l] = x[j * (*nr) + index];
	}

	free(random_obj_num);
}

// ----- Calculate the cluster dispersion (objective function) -----

float calcCost(double *x, 	// Numeric matrix as vector by col (nr*nc)
		int *nr, 	// Number of rows (points)
		int *nc, 	// Number of columns (attributes/variables)
		int *k, 		// Number of clusters
		double *lambda,	// Learning rate
		int *partition, 	// Partition matrix (nr)
		double *o_prototype, // Numeric prototype matrix (k*nc)
		double *subspace_weights) // Weights for variable/cluster (k*nc)
{
	float dispersion = 0.0,  // Dispersion of current cluster
			entropy = 0.0;

	int i, j, l, index;

	for (i = 0; i < *nr; i++)
		for (j = 0; j < *nc; j++) {
			index = j * (*k) + partition[i];
			dispersion += subspace_weights[index]
					* pow(x[j * (*nr) + i] - o_prototype[index], 2);
		}

	for (l = 0; l < (*k) * (*nc); l++)
		entropy += subspace_weights[l] * log(subspace_weights[l]);

	dispersion += *lambda * entropy;

	return dispersion;
}

// ----- Partition objects into clusters -----

void updPartition(  // Inputs
		double *x, 	// Numeric matrix as vector by col (nr*nc)
		int *nr, 	// Number of rows (points)
		int *nc, 	// Number of columns (attributes/variables)
		int *k, 	// Number of clusters
		double *o_prototype, // Numeric prototype matrix (k*nc)
		double *subspace_weights, // Weights for variable/cluster (k*nc)
		// Output
		int *partition)	// Partition matrix (nr)
{
	int i, j, l;

	// We record the cluster number with the smallest distance to a
	// certain object and store the smallest distence between clusers.

	double o_dist, min_dist;

	for (i = 0; i < *nr; i++) {
		min_dist = 1.79769e+308;
		partition[i] = 0;
		for (l = 0; l < *k; l++) {
			o_dist = 0.0;

			for (j = 0; j < *nc; j++) {
				o_dist += subspace_weights[j * (*k) + l]
						* pow(x[j * (*nr) + i] - o_prototype[j * (*k) + l], 2);
			}

			if (min_dist >= o_dist) {
				min_dist = o_dist;
				partition[i] = l;
			}
		}
	}
}

// --- Update the prototypes -----

int updPrototypes(  // Inputs ---------------------------------------------
		double *x, 	// Numeric matrix as vector by col (nr*nc)
		int *nr, 		// Number of rows (points)
		int *nc, 		// Number of columns (attributes/variables)
		int *k, 		// Number of clusters
		int *partition, 	// Partition matrix (nr)
		// Output ---------------------------------------------
		double *o_prototype) // Numeric prototype matrix (k*nc)
{
	int i, j, l;
	int *no_clusters;

	no_clusters = (int *) calloc(*k, sizeof(int));

	//	clean
	for (l = 0; l < (*k) * (*nc); l++) {
		o_prototype[l] = 0;
	}

	//add all values of the cluster
	for (i = 0; i < *nr; i++) {
		//size variable
		no_clusters[partition[i]]++;
		for (j = 0; j < *nc; j++)
			o_prototype[j * (*k) + partition[i]] += x[j * (*nr) + i];
	}

	//divide it by the size
	int flag = 1;
	for (l = 0; l < *k; l++) {
		//if one cluster has zero points restart
		//but I don't care though
		if (no_clusters[l] == 0) {
			flag = 0;
			break;
		}
		for (j = 0; j < *nc; j++)
			o_prototype[j * (*k) + l] /= (double) no_clusters[l];
	}
	free(no_clusters);
	return flag;
}

// ----- Update subspace weights. -----

void updWeights( // Inputs -------------------------------------------------------
		double *x, 	// Numeric matrix as vector by col (nr*nc)
		int *nr, 	// Number of rows (points)
		int *nc, 	// Number of columns (attributes/variables)
		int *k, 	// Number of clusters
		double *lambda,	// Learning rate
		int *partition,	// Partition matrix (nr)
		double *o_prototype, // Numeric prototype matrix (k*nc)
		// Output -------------------------------------------------------
		double *subspace_weights) // Weights for variable/cluster (k*nc)
{
	int i, j, l, index;

	for (l = 0; l < (*k) * (*nc); l++) {
		subspace_weights[l] = 0;
	}

	//	total distance of attribute j in subspace index
	for (i = 0; i < *nr; i++) {
		for (j = 0; j < *nc; j++) {
			index = j * (*k) + partition[i];
			subspace_weights[index] += pow(
					(x[j * (*nr) + i] - o_prototype[index]), 2);
		}
	}

	double minWeight = 0.0001 / (*nc);
	double *max, *sum, *sum2;
	max = (double*) malloc(sizeof(double));
	sum = (double*) malloc(sizeof(double));
	sum2 = (double*) malloc(sizeof(double));

	for (l = 0; l < *k; l++) {
		*max = -1.79769e+308;
		*sum = 0;
		*sum2 = 0;
		//find maximum
		//	Normalization purposes ?
		for (j = 0; j < *nc; j++) {
			index = j * (*k) + l;
			subspace_weights[index] = -subspace_weights[index] / (*lambda);
			if (*max < subspace_weights[index]) {
				*max = subspace_weights[index];
			}
		}
		//compute exp()
		for (j = 0; j < *nc; j++) {
			index = j * (*k) + l;
			subspace_weights[index] = exp(subspace_weights[index] - *max);
			*sum += subspace_weights[index];
		}
		//first normalize
		for (j = 0; j < *nc; j++) {
			index = j * (*k) + l;
			subspace_weights[index] /= *sum; //??
			if (subspace_weights[index] < minWeight) {
				subspace_weights[index] = minWeight;
			}
			//???
			*sum2 += subspace_weights[index];
		}
		//final normalize
		for (j = 0; j < *nc; j++) {
			index = j * (*k) + l;
			subspace_weights[index] /= *sum2;
		}
	}

	free(max);
	free(sum);
	free(sum2);
}

// ***** Primary Interface *****

// This is oriented toward interfacing with R, though a
// separate main.c can be used for stand alone testing.

void ewkm( // Inputs ----------------------------------------------------------
		double *x, 		// Numeric matrix as vector by col (nr*nc)
		int *nr, 		// Number of rows (points)
		int *nc, 		// Number of columns (attributes/variables)
		int *k, 		// Number of clusters
		double *lambda, 	// Learning rate
		int *maxiter, 	// Maximum number of iterations
		double *delta, 	// Minimum change below which iteration stops
		int *maxrestart,      // Maximum number of restarts
		int *init,            // Initial k prototypes.
		// Outputs ---------------------------------------------------------
		int *iterations,	// Number of iterations
		int *cluster, 	// Cluster assignment for each obs (nr) ()
		double *centers, 	// Cluster centers (k*nc) (each cluster has a vector of feature dim)
		double *weights, 	// Variable weights (k*nc) (clusters x variables)
		int *restarts,	// Number of restarts (meh why) --- irrelevant
		int *totiters)	// Number of iterations including restarts --- irrelevant
{
	int l, full;

	int iteration; // Count of iterations.

	// Objective function value. cost function
	float dispersion = FLT_MAX, dispersion1 = FLT_MAX;

	//TODO enable it for R
	// Read in (or create) .Random.seed, the R random number data, and
	// then initialise the random sequence.

	/* GetRNGstate(); */

	//TODO enable it for R
	// Initialise a rand sequence.
	/* srand(unif_rand() * RAND_MAX); */
	srand(123456);

	// Initialize the prototypes. The user can pass in a list of k
	// indicies as the row indicies for the initial protoypes. A
	// single 0 indicates to use random initialisation.

	if (*init == 0)
	  initPrototypes(x, nr, nc, k, centers);

	// Initialize the feature weights of a cluster.

	for (l = 0; l < (*k) * (*nc); l++)
		weights[l] = 1.0 / *nc;

	// Now cluster

	iteration = 0;
	*totiters = 0;
	*restarts = 0;

	while (++iteration <= *maxiter) {
		dispersion = dispersion1;

		updPartition(x, nr, nc, k, centers, weights, cluster);

		// Check if any prototypes are empty, and if so we have to
		// initiate a new search if we have restarts left

		full = updPrototypes(x, nr, nc, k, cluster, centers);

		//restart essenatially
		if (!full && *maxrestart != 0) {
			*restarts += 1;
			*maxrestart -= 1;
			*totiters += iteration;
			// printf("Restarted %d times for %d iterations.\n", *restarts, *totiters);
			iteration = 0;

			// Initialize the prototypes

			initPrototypes(x, nr, nc, k, centers);

			// Initialize the feature weights of a cluster.

			for (l = 0; l < (*k) * (*nc); l++)
				weights[l] = 1.0 / *nc;
		}

		// Update weights of attibutes of each cluster

		updWeights(x, nr, nc, k, lambda, cluster, centers, weights);

		// Compute objective function value

		dispersion1 = calcCost(x, nr, nc, k, lambda, cluster, centers, weights);

		// Check for convergence

		if (fabs(dispersion - dispersion1) / dispersion1 < *delta)
			break;
	}

	// Record results in output variables for passing back to R.

	iterations[0] = iteration - 1;

	*totiters += iteration;
	// If we have reached the maximum iterations, the count was already
	// increased.
	if (iteration == *maxiter + 1)
		*totiters = *totiters - 1;

	//TODO enable it for R
	// Write out the R random number data.
	/* PutRNGstate(); */

}

int main(){
	int nr = 6000; //points
	int nc = 100;
	int k = 50;
	double lambda = 0.5;
	int maxiter = 100;
	double delta = 0.05;
	int maxrestart = 1;
	int iterations = 0;

	double abc[nr*nc];
	double *x = &abc[0];
	int init[k*nc];
	int *initptr = &init[0];
	srand(123456);

	for(int i=0; i<(nr*nc); i++) {
		abc[i] = unif_rand();
		/* printf("%f",abc[i]); */
	}

	for(int i=0; i<(k*nc); i++) {
		init[i] = 0;
	}

	int cluster[nr];
	double centers[k*nc];
	double weights[k*nc];
	int restarts = 0;
	int totiters = 0;

	ewkm(x, &nr,&nc, &k, &lambda, &maxiter, &delta, &maxrestart, initptr, &iterations, &cluster[0], &centers[0], &weights[0],&restarts,&totiters);

	/* for(int i=0; i<(k*nc); i++) { */
	/* 	printf("WEEE"); */
	/* 	printf("%f",weights[i]); */
	/* } */
	/* for (int j = 0; j < (nc); j++){ */
	/* 	printf("%f",centers[j * (k) + 2]); */
	/* } */
	for (int i = 0; i < (nr); i++){
		printf("%d ",cluster[i]);
	}


}

