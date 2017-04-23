import os
import unittest

from amproj.datasets import read_from_data_file_with_headers


class TestBase(unittest.TestCase):

    def test_read_from_data_file_with_headers(self):
        filepath = os.path.join("resources", "example.data")
        ds = read_from_data_file_with_headers(filepath)
        self.assertEqual(20, len(ds.features))
        self.assertEqual(5, len(ds.data))

        # test a couple of datapoints
        self.assertEqual("GRASS", ds.data[0]["CLASS"])
        self.assertEqual("GRASS", ds.data[4]["CLASS"])
        self.assertEqual(9, ds.data[0]["REGION-PIXEL-COUNT"])
        self.assertEqual(9, ds.data[4]["REGION-PIXEL-COUNT"])

        # can't test a float value by comparing it directly:
        self.assertTrue(ds.data[0]["HUE-MEAN"] > 1.91086)
        self.assertTrue(ds.data[0]["HUE-MEAN"] < 1.91087)
        self.assertTrue(ds.data[4]["HUE-MEAN"] > 1.86365)
        self.assertTrue(ds.data[4]["HUE-MEAN"] < 1.86366)
