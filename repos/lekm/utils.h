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
		);

void point_distances( // Inputs ----------------------------------------------------------
		double *x, 		// Numeric matrix as vector by col (nr*nc)
		int *nr, 		// Number of rows (points)
		int *nc, 		// Number of columns (attributes/variables)
		int *k, 		// Number of clusters
		double *lambda, 		// Number of clusters
		int *partition, 	// Cluster assignment for each obs (nr) ()
		double *subspace_weights,
		// Outputs ---------------------------------------------
		double *point_distances		//Distances between points (nr*nr)
		);

