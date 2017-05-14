import unittest

from amproj.datasets import Dataset


class TestDatasetMethods(unittest.TestCase):

    def test_tryparse(self):
        ds = Dataset()
        self.assertEqual("", ds.__tryparse__(""))
        self.assertEqual(2, ds.__tryparse__("2"))
        self.assertEqual(3.4, ds.__tryparse__("3.4"))
        self.assertEqual(2.0, ds.__tryparse__("2.0"))
        self.assertEqual(2, ds.__tryparse__(" 2 "))
        self.assertEqual(2, ds.__tryparse__(" 2"))
        self.assertEqual(2, ds.__tryparse__("2 "))
        self.assertEqual("GRASS", ds.__tryparse__("GRASS"))
        self.assertEqual("GRASS", ds.__tryparse__("GRASS "))
        self.assertEqual("GRASS", ds.__tryparse__(" GRASS"))

    def test_add_datapoint(self):
        ds = Dataset()
        ds.add_datapoint(["1.0", "2.0"])
        self.assertEqual({"feature0": 1.0, "feature1": 2.0}, ds.data[0])
        self.assertEqual({"feature0": 1.0, "feature1": 2.0}, ds[0])
        ds.add_datapoint(["2", "3"])
        self.assertEqual(2, len(ds.data))
        self.assertEqual(2, len(ds))
        self.assertEqual({"feature0": 2, "feature1": 3}, ds.data[1])
        self.assertEqual({"feature0": 2, "feature1": 3}, ds[1])
        with self.assertRaises(TypeError):
            # can't add a datapoint with more features than the others
            ds.add_datapoint(["2", "3", "4"])

        ds = Dataset(["Class", "X1"])
        ds.add_datapoint(["GRASS", "2.0"])
        ds.add_datapoint(["POISON", "3.0"])
        with self.assertRaises(TypeError):
            # can't add a datapoint with more features than the others
            ds.add_datapoint(["POISON", "4.0", "1.0"])
        self.assertEqual({"Class": "GRASS", "X1": 2.0}, ds.data[0])
        self.assertEqual({"Class": "POISON", "X1": 3.0}, ds.data[1])
        self.assertEqual(2, len(ds.data))
        self.assertEqual(2, len(ds))

        # test iteration
        for dp in ds:
            self.assertTrue("Class" in dp)
            self.assertTrue("X1" in dp)
