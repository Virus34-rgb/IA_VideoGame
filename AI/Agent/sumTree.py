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
        Usa np.add.at directamente, sin deduplicar índices por nivel: add.at ya
        acumula correctamente sobre índices repetidos, evitando las llamadas
        extra a unique/bincount/flatnonzero que dominaban el coste con lotes
        pequeños (overhead de NumPy por llamada, no por volumen de datos).
        """
        idxs = numpy.asarray(idxs, dtype=numpy.int64)
        changes = numpy.asarray(changes, dtype=numpy.float32)
        while len(idxs) > 0 and not (len(idxs) == 1 and idxs[0] == 0):
            parents = (idxs - 1) // 2
            numpy.add.at(self.tree, parents, changes)
            mask = parents > 0
            idxs = parents[mask]
            changes = changes[mask]

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

        Vectorizado sin flatnonzero/máscaras dinámicas: la profundidad del árbol es
        estructural (depende solo de self.capacity, no de los datos), así que se itera
        un número fijo de pasos y se usa numpy.where para "congelar" los índices que
        ya llegaron a una hoja, evitando el overhead de filtrar activos en cada nivel.
        """
        values = numpy.asarray(values, dtype=numpy.float32)
        if len(values) == 0:
            return numpy.empty(0, dtype=numpy.int64), numpy.empty(0, dtype=numpy.float32)

        indices = numpy.zeros(len(values), dtype=numpy.int64)
        remaining = values.copy()
        tree_len = len(self.tree)
        depth = int(numpy.ceil(numpy.log2(max(self.capacity, 1)))) + 1

        for _ in range(depth):
            left = 2 * indices + 1
            is_leaf = left >= tree_len

            left_safe = numpy.where(is_leaf, 0, left)
            left_values = self.tree[left_safe]

            go_left = remaining <= left_values
            next_indices = numpy.where(go_left, left, left + 1)

            indices = numpy.where(is_leaf, indices, next_indices)
            remaining = numpy.where(is_leaf | go_left, remaining, remaining - left_values)

        priorities = self.tree[indices]
        return indices, priorities

    def __len__(self) -> int:
        return self.size