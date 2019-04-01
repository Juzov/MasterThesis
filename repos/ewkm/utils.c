#include <math.h>

void distances( // Inputs ----------------------------------------------------------
		double *x, 		// Numeric matrix as vector by col (nr*nc)
		int *nr, 		// Number of rows (points)
		int *nc, 		// Number of columns (attributes/variables)
		int *k, 		// Number of clusters
		double *lambda, 	// Learning rate
		int *cluster, 	// Cluster assignment for each obs (nr) ()
		double *centers, 	// Cluster centers (k*nc) (each cluster has a vector of feature dim)
		double *weights, 	// Variable weights (k*nc) (clusters x variables)
		double *o_prototype, // Numeric prototype matrix (k*nc)
		double *subspace_weights,
		// Outputs ---------------------------------------------
		double *distances 	// Distances to centers (nr)
		)
{
	/* Return the distances to cluster center */
		int i, j, l;

	// We record the cluster number with the smallest distance to a
	// certain object and store the smallest distence between clusers.

	double o_dist;
	int partition;

	for (i = 0; i < *nr; i++) {
		o_dist = 0.0;
		partition = cluster[i];

		for (j = 0; j < *nc; j++) {

			o_dist += subspace_weights[j * (*k) + partition]
					* pow(x[j * (*nr) + i] - o_prototype[j * (*k) + partition], 2);
		}
		distances[i] = o_dist;
	}
}
