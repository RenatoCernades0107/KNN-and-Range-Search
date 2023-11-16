import numpy as np
from collections.abc import Callable
from FixedMaxMinHeap import FixedMaxMinHeap
from distances import euclidian_dist, DistanceFunc


def KNN(data: np.ndarray, target: np.ndarray, k: int = 1, distance: str | Callable[..., int | float] = euclidian_dist):
    distance_fun = DistanceFunc(distance)
    heap = FixedMaxMinHeap(k, lambda y: y[0])
    for id, obj in enumerate(data):
        dist = distance_fun(obj, target)
        heap.push((dist, id))  # push the distance and the identifier of the object

    result = [heap.pop()[1] for _ in range(k)]
    result.reverse()

    return result
