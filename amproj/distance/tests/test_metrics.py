"""Tests the distance/dissimilarity metrics"""

import unittest

from amproj.distance import euclidean


class TestMetrics(unittest.TestCase):

    def test_euclidean(self):
        x = [1, 2, 3, 4]
        y = [3, 4, 5, 6]
        z = [1, 2, 3]
        self.assertEqual(4.0, euclidean(x, y))
        self.assertEqual(4.0, euclidean(y, x))
        self.assertEqual(0.0, euclidean(x, x))
        self.assertEqual(0.0, euclidean(y, y))
        with self.assertRaises(ValueError):
            euclidean(x, z)
        with self.assertRaises(ValueError):
            euclidean(z, x)
        try:
            import numpy as np
            xa = np.array(x)
            ya = np.array(y)
            self.assertEqual(4.0, euclidean(xa, ya))
            self.assertEqual(4.0, euclidean(ya, xa))
            self.assertEqual(0.0, euclidean(xa, xa))
            self.assertEqual(0.0, euclidean(ya, ya))
        except ImportError:
            pass
