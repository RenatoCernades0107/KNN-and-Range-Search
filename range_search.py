import numpy as np
from collections.abc import Callable
from distances import euclidian_dist, DistanceFunc


def range_search(data: np.ndarray, target: np.ndarray, radius: int | float, distance: str | Callable[..., int | float] = euclidian_dist) -> list[int]:
    distance_fun = DistanceFunc(distance)
    result: list[int] = []
    for id, obj in enumerate(data):
        dist = distance_fun(target, obj)
        if dist <= radius:
            result.append(id)

    return result
