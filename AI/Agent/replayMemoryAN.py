"""
Memoria de replay con priorización (PER) para transiciones de turno.

Almacena experiencias de turno (estado, acción, recompensa, siguiente estado, etc.)
y permite muestreo con prioridad basado en error TD.
"""
from typing import Tuple, Any
import numpy as np
import torch

from AI.Agent.replayStorage import ReplayStorage
from AI.Agent.sumTree import SumTree
import constants


class ReplayMemoryAN:
    """
    Replay memory para transiciones de turno (acción por slot).

    Usa SumTree para priorización y ReplayStorage para almacenamiento eficiente.
    """

    def __init__(self, capacity: int, state_dim: int) -> None:
        """
        Args:
            capacity: Número máximo de transiciones.
            state_dim: Dimensión del vector de observación.
        """
        self.capacity = capacity
        self.memory = SumTree(capacity)
        self.storage = ReplayStorage(capacity, state_dim, action_shape=(3,), is_turn_storage=True)

        self.alpha = constants.ALPHA
        self.beta = constants.BETA_START
        self.beta_end = constants.BETA_END
        self.epsilon = constants.PER_EPSILON
        self.max_priority = 1.0

    def push_batch(
            self, states, actions, rewards, next_states, dones,
            alive, types, cooldowns, opp_types,
            next_types, next_alive, next_cooldowns, next_opp_types,
            instance_abilities, next_instance_abilities,
            action_mask, next_action_mask,
        ) -> None:
            n = len(states)
            priorities = np.full(n, self.max_priority, dtype=np.float32)
            indices = self.memory.add_batch(priorities)

            self.storage.states[indices] = states.half()
            self.storage.actions[indices] = actions
            self.storage.rewards[indices] = rewards
            self.storage.next_states[indices] = next_states.half()
            self.storage.dones[indices] = dones

            self.storage.alive[indices] = alive
            self.storage.types[indices] = types
            self.storage.cooldowns[indices] = cooldowns
            self.storage.opp_types[indices] = opp_types

            self.storage.next_types[indices] = next_types
            self.storage.next_alive[indices] = next_alive
            self.storage.next_cooldowns[indices] = next_cooldowns
            self.storage.next_opp_types[indices] = next_opp_types

            self.storage.instance_abilities[indices] = instance_abilities
            self.storage.next_instance_abilities[indices] = next_instance_abilities
            self.storage.action_mask[indices] = action_mask
            self.storage.next_action_mask[indices] = next_action_mask  

    def sample(self, batch_size: int) -> Tuple[Any, np.ndarray, np.ndarray]:
        """
        Muestrea un lote de transiciones con prioridad.

        Returns:
            batch: ReplayBatch (namedtuple) con los campos.
            tree_indices: índices en el árbol para actualizar prioridades después.
            weights: pesos de importancia para la pérdida ponderada.
        """
        total = self.memory.total()
        if not np.isfinite(total) or total == 0:
            self.memory.tree.fill(1.0)
            self.max_priority = 1.0
            total = self.memory.total()
        segment = total / batch_size

        starts = np.arange(batch_size) * segment
        ends = (np.arange(batch_size) + 1) * segment
        values = np.random.uniform(starts, ends)
        values = np.minimum(values, total - 1e-6)

        tree_indices, priorities = self.memory.get_batch(values)

        probabilities = np.maximum(priorities, 1e-8) / total
        weights = (total * probabilities) ** (-self.beta)
        weights /= weights.max()

        data_indices = tree_indices - self.capacity + 1  # convertir a índices de almacenamiento
        batch = self.storage.get_batch(data_indices)

        return batch, tree_indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Actualiza las prioridades de las transiciones muestreadas."""
        priorities = np.abs(priorities) + self.epsilon
        priorities = priorities ** self.alpha
        self.memory.update_batch(indices, priorities)
        self.max_priority = max(self.max_priority, np.max(priorities))

    def update_beta(self, replayed_count: int) -> None:
        """Decae el factor de importancia (beta) según el número de repeticiones."""
        self.beta = constants.BETA_END - (constants.BETA_END - constants.BETA_START) * (constants.BETA_DECAY_RATE ** replayed_count)

    def __len__(self) -> int:
        return len(self.memory)

    def state_dict(self) -> dict:
        """Devuelve el estado interno para guardar checkpoint."""
        return {
            "tree": self.memory.tree.copy(),
            "write": self.memory.write,
            "size": self.memory.size,
            "max_priority": self.max_priority,
            "storage": self.storage.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Carga el estado interno desde un checkpoint."""
        self.memory.tree = state["tree"].copy()
        self.memory.write = state["write"]
        self.memory.size = state["size"]
        self.max_priority = state["max_priority"]
        self.storage.load_state_dict(state["storage"])