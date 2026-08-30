"""
SumTree para Prioritized Experience Replay (PER).
Mantiene un árbol binario donde las hojas almacenan prioridades y los nodos internos
la suma de prioridades de sus hijos. Permite muestreo eficiente con O(log n).
"""
import numpy


class SumTree:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)  # índices 0..2*cap-2
        self.write = 0   # próxima posición de escritura (circular)
        self.size = 0    # número de elementos insertados (máx capacity)

    def _propagate_batch(self, idxs: numpy.ndarray, changes: numpy.ndarray) -> None:
        """
        Propaga los cambios hacia arriba en el árbol para un conjunto de hojas.
        Vectorizado: en cada nivel, agrupa por nodo padre y acumula los cambios
        usando bincount (más rápido que unique+add.at para índices densos y acotados).
        """
        idxs = numpy.asarray(idxs, dtype=numpy.int64)
        changes = numpy.asarray(changes, dtype=numpy.float32)
        tree_len = len(self.tree)
        while len(idxs) > 0:
            parents = (idxs - 1) // 2
            parent_changes_full = numpy.bincount(parents, weights=changes, minlength=tree_len)
            unique_parents = numpy.unique(parents)
            parent_changes = parent_changes_full[unique_parents]
            self.tree[unique_parents] += parent_changes
            # Subir al siguiente nivel (nodos que no son raíz)
            mask = unique_parents > 0
            idxs = unique_parents[mask]
            changes = parent_changes[mask]

    def total(self) -> float:
        """Suma total de prioridades (raíz del árbol)."""
        return self.tree[0]

    def add_batch(self, priorities: numpy.ndarray) -> numpy.ndarray:
        """
        Añade un lote de prioridades en las hojas correspondientes (orden circular).
        Devuelve los índices de las hojas (en el espacio de datos) para escribir después.
        """
        n = len(priorities)
        if n == 0:
            return numpy.array([], dtype=numpy.int64)
        indexes = numpy.arange(n)
        priority_indexes = (self.write + indexes) % self.capacity
        leaf_indexes = priority_indexes + self.capacity - 1

        # Actualizar hojas y propagar
        # Para evitar duplicados si write + n supera capacity, agrupamos por leaf
        unique_leaves, unique_indices = numpy.unique(leaf_indexes[::-1], return_index=True)
        unique_indices = n - 1 - unique_indices  # revertir orden
        filtered_leaves = leaf_indexes[unique_indices]

        change = priorities[unique_indices] - self.tree[filtered_leaves]
        self.tree[filtered_leaves] = priorities[unique_indices]
        self._propagate_batch(filtered_leaves, change)

        self.write = (self.write + n) % self.capacity
        self.size = min(self.size + n, self.capacity)
        return priority_indexes

    def update_batch(self, tree_indices: numpy.ndarray, priorities: numpy.ndarray) -> None:
        """Actualiza las prioridades de un conjunto de hojas (índices de árbol)."""
        tree_indices = numpy.asarray(tree_indices, dtype=numpy.int64)
        priorities = numpy.asarray(priorities, dtype=numpy.float32)
        changes = priorities - self.tree[tree_indices]
        self.tree[tree_indices] = priorities
        self._propagate_batch(tree_indices, changes)

    def get_batch(self, values: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Dado un conjunto de valores uniformes en [0, total), devuelve los índices de
        las hojas correspondientes y sus prioridades.
        """
        values = numpy.asarray(values, dtype=numpy.float32)
        if len(values) == 0:
            return numpy.empty(0, dtype=numpy.int64), numpy.empty(0, dtype=numpy.float32)

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

    def __len__(self) -> int:
        return self.size