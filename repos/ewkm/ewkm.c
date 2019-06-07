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

void initPrototypesPlusPlus(
		double *x,	// Numeric matrix as vector by col (nr*nc)
		int *nr, 	// Number of rows (points)
		int *nc, 	// Number of columns (attributes/variables)
		int *k,	// Number of clusters
		// Output ---------------------------------------------
		double *o_prototype) // Numeric prototype matrix (k*nc)
{

	int i, j, l;
	int index;

	double *min_cluster_dis;

	// Minimum distance for point i to any already chosen cluster center
	// a distance of zero represents a cluster center
	min_cluster_dis = (double *) malloc(sizeof(double) * (*nr));

	if (!min_cluster_dis) {
		fprintf(stderr, "Can't allocate memory for min_cluster_dis matrix\n");
		exit(-1);
	}

	index = (int) (((*nr)-1) * unif_rand());
	for (j = 0; j < (*nc); j++){
		o_prototype[j * (*k) + 0] = x[j * (*nr) + index];
	}

	double o_dist, sum;

	// check min dist to 0
	for (l = 1; l < *k; l++) {
		sum = 0.0;
		for (i = 0; i < *nr; i++) {
			o_dist = 0.0;

			for (j = 0; j < *nc; j++) {
				o_dist += pow(x[j * (*nr) + i] - o_prototype[j * (*k) + (l-1)], 2);
			}

			if( l == 1 ){
				min_cluster_dis[i] = o_dist;
				/* printf("min %f, %d\n", min_cluster_dis[i], i); */
			}
			else if (min_cluster_dis[i] > o_dist) {
				min_cluster_dis[i] = o_dist;
			}
			sum += min_cluster_dis[i];
		}

		//random value
		sum *= (double) unif_rand();
		/* printf("sum %f, %d\n", sum, l); */
		/* printf("%f\n",sum); */

		// since the min dist to the already chosen cluster centers is zero
		for (i = 0; i < *nr; i++) {
			sum -= min_cluster_dis[i];
			if(sum <= 0){
				index = i;
				break;
			}
		}
		for (j = 0; j < (*nc); j++){
			o_prototype[j * (*k) + l] = x[j * (*nr) + index];
		}
	}

	free(min_cluster_dis);
}



void initPrototypes( // Inputs ---------------------------------------------
		double *x,	// Numeric matrix as vector by col (nr*nc)
		int *nr, 	// Number of rows
		int *nc, 	// Number of columns
		int *k,	// Number of clusters
		// Output ---------------------------------------------
		double *o_prototype) // Numeric prototype matrix (k*nc)
{
	int i, j, l;
	int flag = 0;
	int index;

	int *random_obj_num; 	// Array for randomly selected objects (k)

	// Memory for array of randomly selected objects

	random_obj_num = (int *) malloc(sizeof(int) * (*k));
	if (!random_obj_num) {
		perror("Can't allocate memory for random_obj_num matrix\n");
	}

	for (l = 0; l < *k; l++)
		random_obj_num[l] = -1;

	// Randomly select k objects.

	for (l = 0; l < *k; l++) {
		flag = 1;

		while (flag) {
		  //    index = (int) (rand() % (*nr));
			index = (int) (*nr-1) * unif_rand();
			flag = 0;
			for (i = 0; i < l; i++)
				if (random_obj_num[i] == index	)
					flag = 1;
		}

		random_obj_num[l] = index;
		for (j = 0; j < (*nc); j++)
			o_prototype[j * (*k) + l] = x[j * (*nr) + index];
	}

	free(random_obj_num);
}

// ----- Calculate the cluster dispersion (objective function) -----
double calcCost(double *x, 	// Numeric matrix as vector by col (nr*nc)
		int *nr, 	// Number of rows
		int *nc, 	// Number of columns
		int *k, 		// Number of clusters
		double *lambda,	// Learning rate
		int *partition, 	// Partition matrix (nr)
		double *o_prototype, // Numeric prototype matrix (k*nc)
		double *subspace_weights) // Weights for variable/cluster (k*nc)
{
	double dispersion = 0.0,  // Dispersion of current cluster
			entropy = 0.0,
			curEntrop = 0.0;

	int i, j, l, index;

	for (i = 0; i < *nr; i++){
		for (j = 0; j < *nc; j++) {
			index = j * (*k) + partition[i];
			dispersion += subspace_weights[index]
					* pow(x[j * (*nr) + i] - o_prototype[index], 2);
		}
	}

	for (l = 0; l < (*k); l++){
		for (j = 0; j < *nc; j++) {
			index = j * (*k) + l;
			if(subspace_weights[l] > 0){
				entropy += (subspace_weights[l] * log(subspace_weights[l]));
			}
		}
	}
	printf("Disp vs Lambda * Entropy (#): %11.10f - %11.10f \n", dispersion, (*lambda) * entropy);

	dispersion += *lambda * entropy;
	printf("^^^^^^^^^^^^^^^^^\n");
	printf("  cost funct %f \n", dispersion);
	printf("^^^^^^^^^^^^^^^^^\n");
	return dispersion;
}

/* double calcCost(double *x, 	// Numeric matrix as vector by col (nr*nc) */
/* 		int *nr, 	// Number of rows */
/* 		int *nc, 	// Number of columns */
/* 		int *k, 		// Number of clusters */
/* 		double *lambda,	// Learning rate */
/* 		int *partition, 	// Partition matrix (nr) */
/* 		double *o_prototype, // Numeric prototype matrix (k*nc) */
/* 		double *subspace_weights) // Weights for variable/cluster (k*nc) */
/* { */
/* 	double dispersion = 0.0,  // Dispersion of current cluster */
/* 			curDispersion; */

/* 	int i, j, l, index; */
/* 	double *entropies, */
/* 		  *dispersions; */

/* 	entropies = (double *) malloc(sizeof(double) * (*k)); */
/* 	dispersions = (double *) malloc(sizeof(double) * (*k)); */

/* 	for (l = 0; l < *k; l++) { */
/* 		entropies[l] = 0.0; */
/* 		dispersions[l] = 0.0; */
/* 	} */


/* 	for (i = 0; i < *nr; i++){ */
/* 		curDispersion = 0.0; */
/* 		for (j = 0; j < *nc; j++) { */
/* 			index = j * (*k) + partition[i]; */

/* 			dispersions[partition[i]] += subspace_weights[index] */
/* 					* pow(x[j * (*nr) + i] - o_prototype[index], 2); */
/* 		} */

/* 	for (l = 0; l < *k; l++) { */
/* 		for (j = 0; j < *nc; j++) { */
/* 			index = j * (*k) + l; */
/* 			entropies[l] += subspace_weights[index] * log(subspace_weights[index]) / log(subspace_weights[index]); */
/* 		} */
/* 		curDispersion = dispersions[l] + *lambda * entropies[l]; */
/* 		if(l % (int)(*k * 0.1) == 0) */
/* 			printf("Disp vs Lambda * Entropy (#): %11.10f - %11.10f - %d \n", dispersions[l], *lambda * entropies[l], l); */
/* 		dispersion += curDispersion; */
/* 	} */
/* 	/1* dispersion += *lambda * entropy; *1/ */
/* 	printf("^^^^^^^^^^^^^^^^^\n"); */
/* 	printf("  cost funct %f \n", dispersion); */
/* 	printf("^^^^^^^^^^^^^^^^^\n"); */

/* 	free(dispersions); */
/* 	free(entropies); */
/* 	return dispersion; */
/* } */

// ----- Partition objects into clusters -----

void updPartition(  // Inputs
		double *x, 	// Numeric matrix as vector by col (nr*nc)
		int *nr, 	// Number of rows
		int *nc, 	// Number of columns
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
		int *nr, 		// Number of rows
		int *nc, 		// Number of columns
		int *k, 		// Number of clusters
		int *partition, 	// Partition matrix (nr)
		// Output ---------------------------------------------
		double *o_prototype) // Numeric prototype matrix (k*nc)
{
	int i, j, l;
	int *no_clusters;

	no_clusters = (int *) calloc(*k, sizeof(int));

	for (l = 0; l < (*k) * (*nc); l++) {
		o_prototype[l] = 0;
	}

	for (i = 0; i < *nr; i++) {
		no_clusters[partition[i]]++;
		for (j = 0; j < *nc; j++)
			o_prototype[j * (*k) + partition[i]] += x[j * (*nr) + i];
	}

	int flag = 1;
	for (l = 0; l < *k; l++) {
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
		int *nr, 	// Number of rows
		int *nc, 	// Number of columns
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
			subspace_weights[index] /= *sum; //?
			if (subspace_weights[index] < minWeight) {
				subspace_weights[index] = minWeight;
			}
			/* else{ */
			/* 	printf("check\n"); */
			/* } */
			*sum2 += subspace_weights[index];
		}
		//final normalize
		for (j = 0; j < *nc; j++) {
			index = j * (*k) + l;
			subspace_weights[index] /= *sum2;
		}
		printf("%f\n", *sum2);
	}

	free(max);
	free(sum);
	free(sum2);
}


double ewkm( // Inputs ----------------------------------------------------------
		double *x, 		// Numeric matrix as vector by col (nr*nc)
		int *nr, 		// Number of rows
		int *nc, 		// Number of columns
		int *k, 		// Number of clusters
		double *lambda, 	// Learning rate
		int *maxiter, 	// Maximum number of iterations;
		double *delta, 	// Minimum change below which iteration stops
		int *maxrestart,      // Maximum number of restarts
		int *init,            // Initial k prototypes.
		// Outputs ---------------------------------------------------------
		int *iterations,	// Number of iterations
		int *cluster, 	// Cluster assignment for each obs (nr)
		double *centers, 	// Cluster centers (k*nc)
		double *weights, 	// Variable weights (k*nc)
		int *restarts,	// Number of restarts
		int *totiters)	// Number of iterations including restarts
{
	int l, full;

	int iteration; // Count of iterations.

	int initType = *init;

	double dispersion = DBL_MAX, dispersion1 = DBL_MAX;

	//TODO enable it for R
	// Read in (or create) .Random.seed, the R random number data, and
	// then initialise the random sequence.

	/* GetRNGstate(); */

	//TODO enable it for R
	// Initialise a rand sequence.
	srand(time(NULL));

	// Initialize the prototypes. The user can pass in a list of k
	// indicies as the row indicies for the initial protoypes. A
	// single 0 indicates to use random initialisation.

	printf("start weights %f\n",weights[0]);
	if (initType == 0)
	  	initPrototypes(x, nr, nc, k, centers);
	else
		initPrototypesPlusPlus(x, nr, nc, k, centers);
	// Initialize the feature weights of a cluster.

	for (l = 0; l < (*k) * (*nc); l++)
		weights[l] = 1.0 / *nc;

	// Now cluster

	iteration = 0;
	*totiters = 0;
	*restarts = 0;
	printf("re-set weights %f\n",weights[0]);

	printf("----- k lamb: %d, %f ------ \n", *k, *lambda);
	while (++iteration <= *maxiter) {
		printf("> iteration: %d, restart: %d, tot iterations: %d \n", iteration, *restarts, *totiters + iteration);

		dispersion = dispersion1;

		updPartition(x, nr, nc, k, centers, weights, cluster);

		// Check if any prototypes are empty, and if so we have to
		// initiate a new search if we have restarts left

		full = updPrototypes(x, nr, nc, k, cluster, centers);

		if (!full && *maxrestart != 0) {
			*restarts += 1;
			*maxrestart -= 1;
			*totiters += iteration;
			// printf("Restarted %d times for %d iterations.\n", *restarts, *totiters);
			iteration = 0;

			// Initialize the prototypes

			if (initType == 0)
				initPrototypes(x, nr, nc, k, centers);
			else
				initPrototypesPlusPlus(x, nr, nc, k, centers);

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
	printf("Final disp: %f,%d\n",dispersion, iteration);
	printf("-----------:");
	printf("final weights %f\n",weights[0]);

	return dispersion;

}
