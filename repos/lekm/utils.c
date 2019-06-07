#include <math.h>
#include <stdio.h>
#include <ctype.h>

void distances( // Inputs ----------------------------------------------------------
		double *x, 		// Numeric matrix as vector by col (nr*nc)
		int *nr, 		// Number of rows (points)
		int *nc, 		// Number of columns (attributes/variables)
		int *k, 		// Number of clusters
		double *lambda, // smoothness
		int *cluster, 	// Cluster assignment for each obs (nr) ()
		double *o_prototype, // Numeric prototype matrix (k*nc)
		double *subspace_weights,
		// Outputs ---------------------------------------------
		double *distances 	// Distances to centers (nr)
		)
{
	int i, j, count, index;


	double o_dist;

	count = 0;

	for (i = 0; i < *nr; i++) {
		o_dist = 0.0;
		/* o_reg = 0.0; */

		for (j = 0; j < *nc; j++) {
			index = j * (*k) + cluster[i];

			o_dist += subspace_weights[index] * (log(1 + pow(x[j * (*nr) + i] - o_prototype[index], 2)));
			/* o_reg += subspace_weights[index] * log(subspace_weights[index]) / *nc; */
		}

		distances[i] = o_dist;
		if(distances[i] < 0)
			count++;
	}
	printf("distances f: %d\n", count);
}

void point_distances( // Inputs ----------------------------------------------------------
		double *x, 		// Numeric matrix as vector by col (nr*nc)
		int *nr, 		// Number of rows (points)
		int *nc, 		// Number of columns (attributes/variables)
		int *k, 		// Number of clusters
		double *lambda, // smoothness
		int *partition, 	// Cluster assignment for each obs (nr) ()
		double *subspace_weights,
		// Outputs ---------------------------------------------
		double *point_distances		//Distances between points (nr*nr)
		)
{
	/* Return the distances between points with respect to the subspaces  */
	/* Returns a non-symmetric distance measure. */
	/* Based on updPartition */
	int i1, i2, j;

	double o_dist, o_reg;
	int index;

	for (i1 = 0; i1 < *nr; i1++) {
		for (i2 = 0; i2 < *nr; i2++) {
			if(i1 == i2)
				continue;
			o_reg = 0.0;
			o_dist = 0.0;
			for (j = 0; j < *nc; j++) {
				index = j * (*k) + partition[i2];
				o_dist += subspace_weights[index] * log(1 + pow(x[j * (*nr) + i1] - x[j * (*nr) + i1], 2));
				o_reg += subspace_weights[index] * log(subspace_weights[index]);
			}

			o_dist = o_dist + o_reg * *lambda;
			// from the perspective of i1 the distance of i2
			// hence, weights from the i2 cluster.
			point_distances[i2 * (*nr) + i1] = o_dist;
		}
	}
}
