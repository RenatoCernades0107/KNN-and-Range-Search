import numpy as np


def euclidian_dist(x: np.ndarray, y: np.ndarray):
    return np.linalg.norm(x - y)

def range_search(data: np.ndarray, target: np.ndarray, radius: int | float) -> list[int]:
    result: list[int] = []
    for id, obj in enumerate(data):
        dist = euclidian_dist(target, obj)
        if dist <= radius:
            result.append(id)

    return result
