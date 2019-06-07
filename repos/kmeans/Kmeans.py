import sys

import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd
import pandas_profiling as pp
import random


import softsubspace as ss


class Kmeans:
    """ A class used for clustering analysis using the
    Entropy-Weighted-K-means algorithm
    ...

    Attributes
    -------


    Methods
    -------


    """
    # _k = 50
    # _lamb = 1
    # _max_iter = 100
    # _iterations = 100
    # _max_restart = 0
    # _delta = 0.05

    def __init__(self, data):
        # get numpy 2d array from dataframe
        X = data.values
        # self._X = X.ravel(order='F') * 100
        self._orig_X = X
        # ravel it, column-wise
        # The c-code is structured to not have any 2d-arrays instead an column-wise array
        self._X = X.ravel(order='F')
        self._nr, self._nc = X.shape

        # initilize empty array with dimensions dependent on X and k
    def predict(self, k, lamb, max_iter, \
                delta, max_restart, init=0):
        """Run the algorithm"""

        self._k = k
        self._lamb = lamb
        self._max_iter = max_iter
        self._max_restart = max_restart
        self._delta = delta
        self._init = init

        self._cluster = np.empty(self._nr, dtype='int32')
        self._cluster.fill(-1)

        self._centers = np.zeros((k*self._nc))
        # self._weights = np.empty((k*self._nc))
        # uniform = 1/self._nc
        # self._weights.fill(uniform)
        self._weights = np.zeros((k*self._nc))

        iterations = 0
        restarts = 0
        totiters = 0

        dispersion, iterations, restarts, totiters = ss.kmeans(
            self._X,
            self._nr,
            self._nc,
            self._k,
            self._lamb,
            self._max_iter,
            self._delta,
            self._max_restart,
            self._init,
            iterations,
            self._cluster,
            self._centers,
            self._weights,
            restarts,
            totiters
        )

        self._distances = ss.distances(
            self._X,
            self._nr,
            self._nc,
            self._k,
            self._cluster,
            self._centers,
            self._weights
        )


        unraveled_weights = np.reshape(
            a=self._weights,
            newshape=(k, self._nc),
            order='F'
        )

        unraveled_centers = np.reshape(
            a=self._centers,
            newshape=(k, self._nc),
            order='F'
        )

        print(unraveled_weights.shape)
        print(self._orig_X.shape)

        # for i in range(0, self._nr):
        #     partition = self._cluster[i]
        #     for j in range(0, self._nc):
        #         distances[i] += pow((self._orig_X[i,j] - unraveled_centers[partition,j]),2)


        return dispersion, self._cluster, unraveled_centers, \
            unraveled_weights, iterations, restarts, totiters, self._distances

    def silhouette(self, sample_size=1000):
        """ Returns the silhouette score
        through sklearn with a precomputed
        point-2-point distance matrix
        calculated in c function point_distances
        """
        if sample_size > self._nr:
            sample_size = int((self._nr) * 0.1)
            print(f're-set sample size to {sample_size}')

        idx = np.random.choice(
            self._orig_X.shape[0],
            sample_size,
            replace=False
        )

        sample_X = self._orig_X[idx]
        sample_cluster = self._cluster[idx]

        self._point_distances = ss.point_distances(
            sample_X.ravel(order='F'),
            sample_size,
            self._nc,
            self._k,
            sample_cluster,
            self._weights
        )

        self._point_distances = np.reshape(
            a=self._point_distances,
            newshape=(sample_size, sample_size),
            order='F'
        )


        return silhouette_score(
            self._point_distances,
            sample_cluster,
            metric="precomputed"
        )
