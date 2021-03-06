"""
The `amproj.distance` module includes utilities to calculate dissimilarity
between objects.
"""

from .metrics import euclidean
from .dist_matrices import get_dist_matrix
from .kmedoids import FuzzyKMedoids

__all__ = [
    'euclidean',
    'get_dist_matrix',
    'FuzzyKMedoids',
]
