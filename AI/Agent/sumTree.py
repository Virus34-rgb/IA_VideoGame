import numpy


class SumTree:

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)

        self.write = 0
        self.size = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, value):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if value <= self.tree[left]:
            return self._retrieve(left, value)

        return self._retrieve(right, value - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        tree_index = self.write + self.capacity - 1

        self.data[self.write] = data

        self.update(tree_index, priority)

        self.write += 1

        if self.write >= self.capacity:
            self.write = 0

        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]

        self.tree[tree_index] = priority

        self._propagate(tree_index, change)

    def get(self, value):
        tree_index = self._retrieve(0, value)

        data_index = tree_index - self.capacity + 1

        return (
            tree_index,
            self.tree[tree_index],
            self.data[data_index]
        )

    def __len__(self):
        return self.size