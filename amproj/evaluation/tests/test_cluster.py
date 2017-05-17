"""Tests the cluster metrics"""

import os
import unittest

from amproj.datasets import read_from_data_file_with_headers
from amproj.distance import euclidean, get_dist_matrix
from amproj.distance import FuzzyKMedoids
from amproj.evaluation import rand_index, adjusted_rand_index


class TestCluster(unittest.TestCase):

    def test_rand(self):
        filepath = os.path.join("resources", "example14.data")
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
        expected_partition = []
        expected_groups = []
        class_idx = {}
        c = 0
        e = 0
        for data in ds:
            if data['CLASS'] not in class_idx:
                class_idx[data['CLASS']] = c
                expected_groups.append([])
                c += 1
            expected_partition.append(class_idx[data['CLASS']])
            expected_groups[class_idx[data['CLASS']]].append(e)
            e += 1
        rand_expected = rand_index(expected_partition, expected_partition)
        adj_rand_expected = adjusted_rand_index(
                                expected_groups, expected_groups, len(ds))
        self.assertTrue(rand_expected > 0.99999)
        self.assertTrue(rand_expected < 1.00001)
        self.assertTrue(adj_rand_expected > 0.99999)
        self.assertTrue(adj_rand_expected < 1.00001)
        for i in range(100):
            fuzzyk = FuzzyKMedoids(7, 1.6, 1.0, 1000000, 0.01, 2)
            lambs, G, u, J = fuzzyk.fit(self.v1, self.v2)
            actual_partition = []
            actual_groups = []  # list of elements in each group
            for k in range(len(G)):
                actual_groups.append([])
            for e in range(len(u)):
                max_group = -1
                max_val = 0.0
                for k in range(len(u[e])):
                    if u[e][k] > max_val:
                        max_val = u[e][k]
                        max_group = k
                actual_groups[max_group].append(e)
                actual_partition.append(max_group)
            r = rand_index(expected_partition, actual_partition)
            ar = adjusted_rand_index(expected_groups, actual_groups, len(ds))
            self.assertTrue(r > 0.00000)
            self.assertTrue(r < 1.00001)
            self.assertTrue(ar > 0.00000)
            self.assertTrue(ar < 1.00001)
            ri = rand_index(actual_partition, expected_partition)
            self.assertTrue(abs(r - ri) < 0.00001)
            ari = adjusted_rand_index(actual_groups, expected_groups, len(ds))
            self.assertTrue(abs(ar - ari) < 0.0001)
            r2 = rand_index(actual_partition, actual_partition)
            self.assertTrue(r2 > 0.99999)
            self.assertTrue(r2 < 1.00001)
