double unif_rand();

void initPrototypes( // Inputs ---------------------------------------------
		double *x,	// Numeric matrix as vector by col (nr*nc);
		int *nr, 	// Number of rows (points);
		int *nc, 	// Number of columns (attributes/variables);
		int *k,	// Number of clusters
		// Output ---------------------------------------------
		double *o_prototype); // Numeric prototype matrix (k*nc)

double calcCost(double *x, 	// Numeric matrix as vector by col (nr*nc);
		int *nr, 	// Number of rows (points);
		int *nc, 	// Number of columns (attributes/variables);
		int *k, 		// Number of clusters
		double *lambda,	// Learning rate
		int *partition, 	// Partition matrix (nr);
		double *o_prototype, // Numeric prototype matrix (k*nc);
		double *subspace_weights); // Weights for variable/cluster (k*nc)

void updPartition(  // Inputs
		double *x, 	// Numeric matrix as vector by col (nr*nc);
		int *nr, 	// Number of rows (points);
		int *nc, 	// Number of columns (attributes/variables);
		int *k, 	// Number of clusters
		double *o_prototype, // Numeric prototype matrix (k*nc);
		double *subspace_weights, // Weights for variable/cluster (k*nc);
		// Output
		int *partition);	// Partition matrix (nr)

int updPrototypes(  // Inputs ---------------------------------------------
		double *x, 	// Numeric matrix as vector by col (nr*nc);
		int *nr, 		// Number of rows (points);
		int *nc, 		// Number of columns (attributes/variables);
		int *k, 		// Number of clusters
		int *partition, 	// Partition matrix (nr);
		// Output ---------------------------------------------
		double *o_prototype); // Numeric prototype matrix (k*nc)

void updWeights( // Inputs -------------------------------------------------------
		double *x, 	// Numeric matrix as vector by col (nr*nc);
		int *nr, 	// Number of rows (points);
		int *nc, 	// Number of columns (attributes/variables);
		int *k, 	// Number of clusters
		double *lambda,	// Learning rate
		int *partition,	// Partition matrix (nr);
		double *o_prototype, // Numeric prototype matrix (k*nc);
		// Output -------------------------------------------------------
		double *subspace_weights); // Weights for variable/cluster (k*nc)

double ewkm( // Inputs ----------------------------------------------------------
		double *x, 		// Numeric matrix as vector by col (nr*nc);
		int *nr, 		// Number of rows (points);
		int *nc, 		// Number of columns (attributes/variables);
		int *k, 		// Number of clusters
		double *lambda, 	// Learning rate
		int *maxiter, 	// Maximum number of iterations
		double *delta, 	// Minimum change below which iteration stops
		int *maxrestart,      // Maximum number of restarts
		int *init,            // Initial k prototypes.
		// Outputs ---------------------------------------------------------
		int *iterations,	// Number of iterations
		int *cluster, 	// Cluster assignment for each obs (nr); ()
		double *centers, 	// Cluster centers (k*nc); (each cluster has a vector of feature dim)
		double *weights, 	// Variable weights (k*nc); (clusters x variables)
		int *restarts,	// Number of restarts (meh why); --- irrelevant
		int *totiters);	// Number of iterations including restarts --- irrelevant


