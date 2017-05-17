"""
The `amproj.evaluation` module includes utilities to calculate different
metrics for evaluating the algorithms implemented in the project.
"""

from .cluster import rand_index
from .cluster import adjusted_rand_index

__all__ = [
    'rand_index',
    'adjusted_rand_index',
]
