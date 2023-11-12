import heapq

class FixedMaxMinHeap:
    def __init__(self, max_size: int, comp_fun=lambda x: x):
        self.data: list[tuple[..., ...]] = []
        self.max_size = max_size
        self.size = 0
        self.comp = comp_fun

    def push(self, obj) -> None:
        if self.size < self.max_size:
            heapq.heappush(self.data, (-self.comp(obj), obj))
            self.size += 1
        else:
            if -self.comp(obj) > self.comp(self.data[0]):
                heapq.heappushpop(self.data, (-self.comp(obj), obj))

    def pop(self) -> tuple[..., ...] | None:
        if self.size == 0:
            return None
        key, value = self.data[0]
        heapq.heappop(self.data)
        self.size -= 1
        return value

    def __repr__(self):
        return [value for _, value in self.data].__repr__()

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter < self.size:
            _, value = self.data[self.iter]
            self.iter += 1
            return value

        raise StopIteration