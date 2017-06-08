"""
Provides functions for calculating clustering metrics, such as the Rand Index
"""

import math
from decimal import Decimal


def rand_index(pt1, pt2):
    """Calculates the rand index between the two given partitions.

    Parameters
    ----------
    pt1 : list
        A list where the index corresponds to an object in some dataset and the
        value corresponds to the cluster to which the object belongs.
    pt2 : list
        A list where the index corresponds to an object in some dataset and the
        value corresponds to the cluster to which the object belongs.

    Returns
    -------
    rand : Decimal
        The rand index comparing the two given partitions.
    """
    a = 0
    b = 0
    c = 0
    d = 0
    pairs = set()
    for e1 in range(len(pt1)):
        for e2 in range(len(pt1)):
            if e1 == e2 or (e1, e2) in pairs or (e2, e1) in pairs:
                continue
            pairs.add((e1, e2))
            if pt1[e1] == pt1[e2] and pt2[e1] == pt2[e2]:
                a += 1
            elif pt1[e1] != pt1[e2] and pt2[e1] != pt2[e2]:
                b += 1
            elif pt1[e1] == pt1[e2] and pt2[e1] != pt2[e2]:
                c += 1
            elif pt1[e1] != pt1[e2] and pt2[e1] == pt2[e2]:
                d += 1
    return Decimal(a + b)/Decimal(a + b + c + d)


def binomial(a, b):
    """Calculates the binomial given the two parameters"""
    if a == b:
        return Decimal(1.0)
    elif b == 1.0:
        return a
    elif b > a:
        return Decimal(0.0)
    else:
        x = math.factorial(a)
        y = math.factorial(b)
        z = math.factorial(a - b)
        return Decimal(x) / Decimal(y * z)


def adjusted_rand_index(pt1, pt2, l):
    """Calculates the adjusted rand index between the two given partitions.

    Parameters
    ----------
    pt1 : list<list>
        A list of lists where each element of the list correspond to a cluster
        and is a list of elements that belong to that cluster.
    pt2 : list<list>
        A list of lists where each element of the list correspond to a cluster
        and is a list of elements that belong to that cluster.
    l : int
        The length of the entire dataset (or the number of examples).

    Returns
    -------
    rand : Decimal
        The adjusted rand index comparing the two given partitions.
    """
    r = len(pt1)
    s = len(pt2)
    if r == s == 1 or r == s == 0 or r == s == l:
        return 1.0
    n = []
    for ctr in range(r):
        n.append([0] * s)
    a = [0] * r
    for i in range(r):
        for j in range(s):
            n[i][j] = len(filter(set(pt1[i]).__contains__, pt2[j]))  # |X n Y|
            a[i] += n[i][j]
    b = [0] * s
    for j in range(s):
        for i in range(r):
            b[j] += n[i][j]
    sum_comb = sum([binomial(n[i][j], 2) for i in range(r) for j in range(s)])
    sum_comb_a = sum([binomial(a[i], 2) for i in range(r)])
    sum_comb_b = sum([binomial(b[j], 2) for j in range(s)])
    prod_comb = Decimal(sum_comb_a * sum_comb_b) / binomial(l, 2)
    mean_comb = Decimal(sum_comb_a + sum_comb_b) / 2
    return Decimal(sum_comb - prod_comb) / Decimal(mean_comb - prod_comb)
