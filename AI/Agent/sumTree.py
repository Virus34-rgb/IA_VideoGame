from collections import namedtuple

import numpy



class SumTree:

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)

        self.write = 0
        self.size = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _propagate_batch(self, idxs, changes):
        idxs = numpy.asarray(idxs)
        changes = numpy.asarray(changes)
        while len(idxs) > 0:
            parents = (idxs - 1) // 2
            unique_parents, inverse = numpy.unique(parents, return_inverse=True)
            parent_changes = numpy.zeros(len(unique_parents))
            numpy.add.at(parent_changes, inverse, changes)
            self.tree[unique_parents] += parent_changes
            mask = unique_parents > 0
            idxs = unique_parents[mask]
            changes = parent_changes[mask]

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
            
    def add_batch(self, priorities):
        priorities_len = len(priorities)
        if priorities_len  == 0:
            return
        indexes = numpy.arange(priorities_len )
        priority_indexes = (self.write + indexes) % self.capacity
        leaf_indexes = priority_indexes + self.capacity - 1
        
        unique_leaves, unique_indices = numpy.unique(leaf_indexes[::-1], return_index=True)
        unique_indices = len(leaf_indexes) - 1 - unique_indices
        filtered_leaves = leaf_indexes[unique_indices]
        
        change = priorities[unique_indices] - self.tree[filtered_leaves]
        
        self.tree[filtered_leaves] = priorities[unique_indices]
        self._propagate_batch(filtered_leaves, change)
        self.write = (self.write + priorities_len ) % self.capacity
        if self.size < self.capacity:
            self.size = min(self.size + priorities_len , self.capacity)
        return priority_indexes
        
    def update_batch(self, tree_indices, priorities):
        tree_indices = numpy.asarray(tree_indices, dtype=numpy.int64)
        priorities = numpy.asarray(priorities)

        changes = priorities - self.tree[tree_indices]
        self.tree[tree_indices] = priorities

        self._propagate_batch(tree_indices, changes)

    def get_batch(self, values):
        values = numpy.asarray(values, dtype=numpy.float32)
        if len(values) == 0:
            return (numpy.empty(0, dtype=numpy.int64),numpy.empty(0, dtype=numpy.float32))

        indices = numpy.zeros(len(values), dtype=numpy.int64)
        remaining = values.copy()
        active = numpy.ones(len(values), dtype=bool)

        while active.any():
            active_indices = numpy.flatnonzero(active)
            current = indices[active_indices]
            left = 2 * current + 1

            is_leaf = left >= len(self.tree)

            leaf_indices = active_indices[is_leaf]
            active[leaf_indices] = False

            non_leaf_indices = active_indices[~is_leaf]

            if len(non_leaf_indices) == 0:
                continue

            non_leaf_left = left[~is_leaf]
            left_values = self.tree[non_leaf_left]

            go_left = remaining[non_leaf_indices] <= left_values
            go_right = ~go_left

            indices[non_leaf_indices[go_left]] = non_leaf_left[go_left]

            right = non_leaf_left[go_right] + 1
            indices[non_leaf_indices[go_right]] = right

            remaining[non_leaf_indices[go_right]] -= left_values[go_right]
        priorities = self.tree[indices]
        return indices, priorities

    def __len__(self):
        return self.size
