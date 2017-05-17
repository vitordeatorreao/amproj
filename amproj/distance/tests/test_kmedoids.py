"""Tests the k-Medoids algorithms"""

import os
import random
import unittest

from amproj.datasets import read_from_data_file_with_headers
from amproj.distance import euclidean, get_dist_matrix
from amproj.distance import FuzzyKMedoids


def prod(x):
    return reduce(lambda a, b: a*b, x)


class TestFuzzyKMedoids(unittest.TestCase):

    def setUp(self):
        filepath = os.path.join("resources", "example.data")
        ds = read_from_data_file_with_headers(filepath)
        self.v1 = get_dist_matrix(
            ds,  # the dataset
            ["REGION-CENTROID-COL",
             "REGION-CENTROID-ROW",
             "REGION-PIXEL-COUNT",
             "SHORT-LINE-DENSITY-5",
             "SHORT-LINE-DENSITY-2",
             "VEDGE-MEAN",
             "VEDGE-SD",
             "HEDGE-MEAN",
             "HEDGE-SD"],
            euclidean  # the metric
        )
        self.v2 = get_dist_matrix(
            ds,  # the dataset
            ["INTENSITY-MEAN",
             "RAWRED-MEAN",
             "RAWBLUE-MEAN",
             "RAWGREEN-MEAN",
             "EXRED-MEAN",
             "EXBLUE-MEAN",
             "EXGREEN-MEAN",
             "VALUE-MEAN",
             "SATURATION-MEAN",
             "HUE-MEAN"],
            euclidean  # the metric
        )
        self.assertEqual(len(self.v1), len(self.v2))
        self.assertEqual(len(self.v1[0]), len(self.v2[0]))

    def __prepare_fuzzy_instance__(self):
        K = 2
        m = 1.6
        s = 1.0
        q = 2
        p = 2
        n = len(self.v1[0])
        # initialize the "view relevance weights per cluster" matrix
        lambs = []  # should have K lines and p columns
        for k in range(K):
            if s == 1.0:  # if this is MFCMdd-RWL-P
                row = [1.0] * p
            else:
                row = [1.0/p] * p
            lambs.append(row)
        # initialize cluster prototypes
        pool = range(n)  # we need to shuffle the pool of objects
        try:
            import numpy as np
            random.shuffle(pool, np.random.uniform)
        except ImportError:
            random.shuffle(pool)
        G = []  # K lines and q columns
        i = 0  # index of the next element we should take for the prototype
        for k in range(K):
            # get the next q elements for the k-th prototype
            G.append(pool[i:i+q])
            i += q
        # compute the membership degree; u has n lines and k columns
        fuzzyk = FuzzyKMedoids(K, m, s, 1000, 0.01, q)  # we do this just...
        # ...because we need an instance of FuzzyKMedoids
        u = fuzzyk.__membership_degress__(
            [self.v1, self.v2], K, m, s, q, p, n, lambs, G)
        return fuzzyk, K, m, s, q, p, n, lambs, G, u

    def test_membership_degress(self):
        fuzzyk, K, m, s, q, p, n, lambs, G, u = \
            self.__prepare_fuzzy_instance__()
        for i in range(n):
            sk = 0
            for k in range(K):
                sk += u[i][k]
                self.assertTrue(u[i][k] > 0.0)
                self.assertTrue(u[i][k] < 1.0)
            self.assertTrue(sk > 0.99 and sk < 1.01)

    def test_cost_function(self):
        K = 2
        m = 1.6
        s = 1.0
        q = 2
        p = 2
        n = len(self.v1[0])
        # initialize the "view relevance weights per cluster" matrix
        lambs = []  # should have K lines and p columns
        for k in range(K):
            if s == 1.0:  # if this is MFCMdd-RWL-P
                row = [1.0] * p
            else:
                row = [1.0/p] * p
            lambs.append(row)
        # initialize cluster prototypes
        pool = range(n)  # we need to shuffle the pool of objects
        try:
            import numpy as np
            np.random.seed(0)
            random.shuffle(pool, np.random.uniform)
        except ImportError:
            random.seed(0)
            random.shuffle(pool)
        G = []  # K lines and q columns
        i = 0  # index of the next element we should take for the prototype
        for k in range(K):
            # get the next q elements for the k-th prototype
            G.append(pool[i:i+q])
            i += q
        # compute the membership degree; u has n lines and k columns
        fuzzyk = FuzzyKMedoids(K, m, s, 100, 0.01, q)  # we do this just...
        # ...because we need an instance of FuzzyKMedoids
        u = fuzzyk.__membership_degress__(
            [self.v1, self.v2], K, m, s, q, p, n, lambs, G)
        print("Distance Matrix for View 1")
        for i in range(n):
            print("[\t" + ",\t".join(map(str, self.v1[i])) + "\t]")
        print("Distance Matrix for View 2")
        for i in range(n):
            print("[\t" + ",\t".join(map(str, self.v2[i])) + "\t]")
        print("Lambdas:")
        for k in range(K):
            print("[\t" + ",\t".join(map(str, lambs[k])) + "\t]")
        for k in range(K):
            print("Prototype of Cluster " + str(k) + " is [" +
                  ", ".join(map(str, G[k])) + "]")
        print("U matrix:")
        for k in range(n):
            print("[\t" + ",\t".join(map(str, u[k])) + "\t]")
        J = fuzzyk.__cost_function__(
            [self.v1, self.v2], K, m, s, q, p, n, u, lambs, G)
        self.assertTrue(J > 334.816776)
        self.assertTrue(J < 334.816777)

    def test_update_prototypes(self):
        for b in range(1000):
            fuzzyk, K, m, s, q, p, n, lambs, G, u = \
                self.__prepare_fuzzy_instance__()
            J_0 = fuzzyk.__cost_function__(
                [self.v1, self.v2], K, m, s, q, p, n, u, lambs, G)
            G = fuzzyk.__update_prototypes__(
                    [self.v1, self.v2], K, m, s, q, p, n, u, lambs)
            J_1 = fuzzyk.__cost_function__(
                [self.v1, self.v2], K, m, s, q, p, n, u, lambs, G)
            print("J_0 = " + str(J_0) + ";\tJ_1 = " + str(J_1))
            self.assertTrue(J_1 <= J_0)

    def test_update_lambs(self):
        for b in range(1000):
            fuzzyk, K, m, s, q, p, n, lambs, G, u = \
                self.__prepare_fuzzy_instance__()
            J_0 = fuzzyk.__cost_function__(
                [self.v1, self.v2], K, m, s, q, p, n, u, lambs, G)
            lambs = fuzzyk.__update_lambs__(
                    [self.v1, self.v2], K, m, s, q, p, n, u, G)
            J_1 = fuzzyk.__cost_function__(
                [self.v1, self.v2], K, m, s, q, p, n, u, lambs, G)
            print("Lambdas:")
            for k in range(K):
                self.assertTrue(prod(lambs[k]) < 1.00001)
                self.assertTrue(prod(lambs[k]) > 0.99999)
                print("[\t" + ",\t".join(map(str, lambs[k])) + "\t]")
            print("J_0 = " + str(J_0) + ";\tJ_1 = " + str(J_1))
            self.assertTrue(J_1 <= J_0)

    def check_Js(self, j1, j2):
        print("Before = " + str(j1) + ";\tafter = " + str(j2))
        # Because of rounding errors common to float types, to check if the
        # new adequacy criterion is lower than the new one, we must check if
        # their difference (old - new) is higher than some very low number.
        self.assertTrue(j1 - j2 > 0.00000001)

    def test_FuzzyKMedoids(self):
        fuzzyk = FuzzyKMedoids(2, 1.6, 1.0, 1000000, 0.01, 2)
        lambs, G, u, J = fuzzyk.fit(self.v1, self.v2, updated=self.check_Js)
