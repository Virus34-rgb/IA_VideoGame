"""
Memoria de replay con priorización (PER) para transiciones de selección.

Almacena experiencias de selección de equipo (estado, acción, recompensa, siguiente estado)
y permite muestreo con prioridad.
"""
from typing import Tuple, Any
import numpy as np
import torch

from AI.Agent.replayStorage import ReplayStorage
from AI.Agent.sumTree import SumTree
from constants import ALPHA, BETA_DECAY_RATE, BETA_END, BETA_START, PER_EPSILON


class ReplayMemoryPM:
    """
    Replay memory para transiciones de selección (acción escalar).
    """

    def __init__(self, capacity: int, state_dim: int) -> None:
        """
        Args:
            capacity: Número máximo de transiciones.
            state_dim: Dimensión del vector de estado de selección.
        """
        self.capacity = capacity
        self.memory = SumTree(capacity)
        self.storage = ReplayStorage(capacity, state_dim, action_shape=(), is_turn_storage=False)

        self.alpha = ALPHA
        self.beta = BETA_START
        self.beta_end = BETA_END
        self.epsilon = PER_EPSILON
        self.max_priority = 1.0

    def push_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor | None,
        dones: torch.Tensor,
    ) -> None:
        """
        Almacena un lote de transiciones.

        Args:
            states: (batch, state_dim)
            actions: (batch,) acciones seleccionadas.
            rewards: (batch,) recompensas.
            next_states: (batch, state_dim) o None (para estados terminales).
            dones: (batch,) bool, True si es terminal.
        """
        n = len(states)
        if not np.isfinite(self.max_priority):
            self.max_priority = 1.0
        priorities = np.full(n, self.max_priority, dtype=np.float32)
        indices = self.memory.add_batch(priorities)

        self.storage.states[indices] = states.half()
        self.storage.actions[indices] = actions
        self.storage.rewards[indices] = rewards
        if next_states is not None:
            self.storage.next_states[indices] = next_states.half()
        else:
            self.storage.next_states[indices] = torch.zeros(states.shape, dtype=torch.float16)
        self.storage.dones[indices] = dones

    def sample(self, batch_size: int) -> Tuple[Any, np.ndarray, np.ndarray]:
        """Muestrea un lote con prioridad."""
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

        data_indices = tree_indices - self.capacity + 1
        batch = self.storage.get_batch(data_indices)

        return batch, tree_indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.abs(priorities) + self.epsilon
        priorities = priorities ** self.alpha
        self.memory.update_batch(indices, priorities)
        self.max_priority = max(self.max_priority, np.max(priorities))

    def update_beta(self, replayed_count: int) -> None:
        self.beta = BETA_END - (BETA_END - BETA_START) * (BETA_DECAY_RATE ** replayed_count)

    def __len__(self) -> int:
        return len(self.memory)

    def state_dict(self) -> dict:
        return {
            "tree": self.memory.tree.copy(),
            "write": self.memory.write,
            "size": self.memory.size,
            "max_priority": self.max_priority,
            "storage": self.storage.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.memory.tree = state["tree"].copy()
        self.memory.write = state["write"]
        self.memory.size = state["size"]
        self.max_priority = state["max_priority"]
        self.storage.load_state_dict(state["storage"])