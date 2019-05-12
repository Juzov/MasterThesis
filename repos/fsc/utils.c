#include <math.h>

void distances( // Inputs ----------------------------------------------------------
		double *x, 		// Numeric matrix as vector by col (nr*nc)
		int *nr, 		// Number of rows (points)
		int *nc, 		// Number of columns (attributes/variables)
		int *k, 		// Number of clusters
		int *cluster, 	// Cluster assignment for each obs (nr) ()
		double *o_prototype, // Numeric prototype matrix (k*nc)
		double *subspace_weights,
		// Outputs ---------------------------------------------
		double *distances 	// Distances to centers (nr)
		)
{
	/* Return the distances to cluster center */
	int i, j;

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

void point_distances( // Inputs ----------------------------------------------------------
		double *x, 		// Numeric matrix as vector by col (nr*nc)
		int *nr, 		// Number of rows (points)
		int *nc, 		// Number of columns (attributes/variables)
		int *k, 		// Number of clusters
		int *cluster, 	// Cluster assignment for each obs (nr) ()
		double *subspace_weights,
		// Outputs ---------------------------------------------
		double *point_distances		//Distances between points (nr*nr)
		)
{
	/* Return the distances between points with respect to the subspaces  */
	/* Returns a non-symmetric distance measure. */
	int i1, i2, j;

	double o_dist;
	int partition_i2;

	for (i1 = 0; i1 < *nr; i1++) {
		for (i2 = 0; i2 < *nr; i2++) {
			partition_i2 = cluster[i2];
			o_dist = 0.0;
			for (j = 0; j < *nc; j++) {

				o_dist += subspace_weights[j * (*k) + partition_i2]
						* pow(x[j * (*nr) + i1] - x[j * (*nr) + i2], 2);
			}
			// from the perspective of i1 the distance of i2
			// hence, weights from the i2 cluster.
			point_distances[i2 * (*nr) + i1] = o_dist;
		}
	}
}
