import sys

import numpy as np
import pandas as pd
import pandas_profiling as pp
import random

import softsubspace as ss


class EWKM:
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
        # read it as a 1d array which is column major
        # remove negative dispersion problems by scaling it by a 100
        X = data.values
        # self._X = X.ravel(order='F') * 100
        self._X = X.ravel(order='F')
        self._nr, self._nc = X.shape

        # initilize empty array with dimensions dependent on X and k
    def predict(self, k, lamb, max_iter, delta, max_restart, init=1):

        self._k = k
        self._lamb = lamb
        self._max_iter = max_iter
        self._max_restart = max_restart
        self._delta = delta
        self._init = init


        self._cluster = np.empty(self._nr, dtype='int32')
        self._cluster.fill(-1)

        self._centers = np.zeros((k*self._nr))
        # self._weights = np.empty((k*self._nc))
        # uniform = 1/self._nc
        # self._weights.fill(uniform)
        self._weights = np.zeros((k*self._nc))
        # print(self._weights)

        iterations = 0
        restarts = 0
        totiters = 0

        ss.ewkm(
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
        print(iterations)
        print(self._X.shape)

        return self._cluster, self._centers, self._weights, iterations, totiters
