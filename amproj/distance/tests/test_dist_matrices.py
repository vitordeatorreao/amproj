"""Tests the methods for calculating distance matrices"""

import os
import unittest

from amproj.datasets import read_from_data_file_with_headers
from amproj.distance import euclidean, get_dist_matrix


class TestDistMatrices(unittest.TestCase):

    def test_get_dist_matrix(self):
        filepath = os.path.join("resources", "example.data")
        ds = read_from_data_file_with_headers(filepath)
        d = get_dist_matrix(
            ds,  # the dataset
            ['REGION-CENTROID-COL',  # the features forming the view
             'REGION-CENTROID-ROW',
             'REGION-PIXEL-COUNT'],
            euclidean  # the metric
        )
        self.assertEqual(len(ds), len(d))
        for row in d:
            self.assertEqual(len(ds), len(row))
        for i in range(len(d)):
            self.assertEqual(0.0, d[i][i])
        for i in range(len(d)):
            for j in range(len(d)):
                self.assertEqual(d[i][j], d[j][i])
