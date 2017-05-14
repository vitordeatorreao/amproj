"""Methods for calculating dissimilarity matrices from datasets"""


def get_dist_matrix(ds, view, metric):
    """Calculates the distance matrix for the objects in the given dataset
    taking into consideration the features in the view array

    Parameters
    ----------
    ds : Dataset
        The Dataset instance with in-memory representation of a dataset
    view : list<string>
        A list containing the labels of the features that should be considered
        for the calculation of the matrix
    metric : callable
        The metric to use for the calculation of distances between the objects

    Returns
    -------
    dist : array-like
        The distance matrix. Where dist[i][j] is the distance between the
        objects i and j
    """
    dist = [] * len(ds)  # the distance matrix
    for i in range(len(ds)):
        row = [0.0] * len(ds)
        dist.append(row)
    objects = {}  # the projection of the elements in the view
    for i in range(len(ds)):
        objects[i] = []
        for attr in view:
            objects[i].append(ds[i][attr])
    for i in range(len(ds)):
        for j in range(len(ds)):
            dist[i][j] = metric(objects[i], objects[j])
    return dist
