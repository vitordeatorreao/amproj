"""Some metrics to calculate the distance or dissimilarity between objects"""

from math import sqrt


def euclidean(x, y):
    """Calculates the dissimiliraty between two objects
    using the euclidean distance

    Parameters
    ----------
    x : an array-like object
        An array where each element is the value of a feature

    y : an array-like object
        An array where each element is the value of a feature

    Returns
    -------
    distance : float
        The distance or dissimiliraty between objects x and y
    """
    if len(x) != len(y):
        raise ValueError("The objects must have the same number of features")
    s = 0  # the sum of squared distances
    for i in range(len(x)):
        s += (x[i] - y[i]) ** 2
    return sqrt(s)
