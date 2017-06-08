import argparse
import datetime
import sys
import traceback

from amproj.datasets import read_from_data_file_with_headers
from amproj.distance import FuzzyKMedoids, get_dist_matrix, euclidean
from amproj.evaluation import rand_index, adjusted_rand_index


def check_Js(j1, j2):
    print("Before = " + str(j1) + ";\tafter = " + str(j2) + " at " + str(datetime.datetime.now()))


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
    expected_partition = []
    expected_groups = []
    class_idx = {}
    c = 0
    e = 0
    for data in dataset:
        if data['CLASS'] not in class_idx:
            class_idx[data['CLASS']] = c
            expected_groups.append([])
            c += 1
        expected_partition.append(class_idx[data['CLASS']])
        expected_groups[class_idx[data['CLASS']]].append(e)
        e += 1
    shapeview = get_dist_matrix(
        dataset,  # the dataset
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
    rgbview = get_dist_matrix(
        dataset,  # the dataset
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
    best_lambs = None
    best_G = None
    best_u = None
    best_value = float("inf")  # infinity
    print("Started at " + str(datetime.datetime.now()))
    for i in range(100):
        try:
            fuzzyk = FuzzyKMedoids(7, 1.6, 1.0, 300, 0.0000000001, 3)
            lambs, G, u, J = fuzzyk.fit(shapeview, rgbview, updated=check_Js)
            if best_value > J:
                best_lambs = lambs
                best_G = G
                best_u = u
                best_value = J
            print("Iteration " + str(i+1) + "/100 at " + str(datetime.datetime.now()))
        except (KeyboardInterrupt, SystemExit, Exception):
            print("Interrupted!")
            ex_type, ex, tb = sys.exc_info()
            print("".join(traceback.format_exception(ex_type, ex, tb)))
            break
    print("Best adequacy criterion = " + str(best_value))
    for k in range(len(best_G)):
        print("Representantes grupo " + str(k) + " = [" +
              ",\t".join(map(str, best_G[k])) + "]")
    print("Matriz de pesos de relevancia das views:")
    groups = []  # list of elements in each group
    for k in range(len(best_G)):
        print("[\t" + ",\t".join(map(str, best_lambs[k])) + "\t]")
        groups.append([])
    actual_partition = []
    for e in range(len(best_u)):
        max_group = -1
        max_val = 0.0
        for k in range(len(best_u[e])):
            if best_u[e][k] > max_val:
                max_val = best_u[e][k]
                max_group = k
        groups[max_group].append(e)
        actual_partition.append(max_group)
    print("A particao exclusiva:")
    for k in range(len(best_G)):
        print("Grupo " + str(k) + " = [" + ",".join(map(str, groups[k])) + "]")
    rand = rand_index(expected_partition, actual_partition)
    print("Indice de rand = " + str(rand))
    adj_rand = adjusted_rand_index(expected_groups, groups, len(dataset))
    print("Indice de rand corrijido = " + str(adj_rand))
