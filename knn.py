import numpy as np
from collections.abc import Callable
from FixedMaxMinHeap import FixedMaxMinHeap
from numbers import Number

def euclidian_dist(x: np.ndarray, y: np.ndarray):
    return np.linalg.norm(x - y)


def manhattan_dist(x: np.ndarray, y: np.ndarray):
    return np.sum(np.abs(x - y))


def chebyshev_dist(x: np.ndarray, y: np.ndarray):
    return np.max(np.abs(x - y))


class DistanceFunc:
    def __init__(self, name_or_func: str | Callable[..., int | float]):
        self.name_or_func = name_or_func
        self.func: Callable = lambda x: x
        if type(name_or_func) is str:
            if name_or_func == "euclidian":
                self.func = euclidian_dist
            elif name_or_func == "manhattan":
                self.func = manhattan_dist
            elif name_or_func == "inf":
                self.func = chebyshev_dist
            else:
                raise Exception(f" {name_or_func} distance does not supported")
        else:
            self.func = name_or_func

    def __call__(self, *args, **kwargs) -> int | float:
        if len(args) != 2:
            raise Exception(f"Any distance must be have two paramenters, not {len(args)}")
        dist = self.func(args[0], args[1])
        if dist < 0:
            raise Exception(f"Any distance must be positive")

        return dist


def KNN(data: np.ndarray, target: np.ndarray, k: int = 1, distance: str | Callable[..., int | float] = euclidian_dist):
    distance_fun = DistanceFunc(distance)
    heap = FixedMaxMinHeap(k, lambda y: y[0])
    for id, obj in enumerate(data):
        dist = distance_fun(obj, target)
        heap.push((dist, id))  # push the distance and the identifier of the object

    result = [heap.pop()[1] for _ in range(k)]
    result.reverse()

    return result
