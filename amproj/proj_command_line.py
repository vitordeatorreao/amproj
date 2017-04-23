import argparse
import sys

from amproj.datasets import read_from_data_file_with_headers


def main():
    parser = argparse.ArgumentParser(
        description="Trains four classifiers according to the given dataset " +
        "and outputs their metrics")
    parser.add_argument("file",
                        help="the data from this file will be read and used " +
                        "as the input dataset")
    args = parser.parse_args()
    dataset = read_from_data_file_with_headers(args.file)
    print("Read {} examples with {} features.".format(
        len(dataset.data), len(dataset.features)))
