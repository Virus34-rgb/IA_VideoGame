"""
Red neuronal para la selección de acciones en el turno (TurnNetwork).

Soporta arquitectura Dueling DQN (separación de valor de estado y ventaja)
y enmascaramiento de acciones inválidas.
"""
import torch
from torch import nn
from constants import USE_DUELING_DQN


class TurnNetwork(nn.Module):
    """
    Red para predecir Q-values de acciones por slot (3 guerreros, 6 acciones cada uno).

    Arquitectura:
        - Tronco compartido (3 capas ocultas).
        - Si USE_DUELING_DQN=True: dos cabezas (valor y ventaja) que se combinan.
        - Si no: una sola cabeza de salida.
    """

    def __init__(
        self,
        input_size: int = 58,
        output_size: int = 18,
        num_slots: int = 3,
        actions_per_slot: int = 6,
    ) -> None:
        """
        Args:
            input_size: Dimensión de la observación.
            output_size: Total de acciones (num_slots * actions_per_slot).
            num_slots: Número de guerreros en el equipo.
            actions_per_slot: Acciones disponibles por guerrero.
        """
        super().__init__()
        assert output_size == num_slots * actions_per_slot, (
            f"output_size ({output_size}) debe ser num_slots*actions_per_slot "
            f"({num_slots}*{actions_per_slot})"
        )
        self.num_slots = num_slots
        self.actions_per_slot = actions_per_slot

        # Tronco compartido
        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        if USE_DUELING_DQN:
            # Dueling: valor del estado (por slot) + ventaja (por acción)
            self.value = nn.Linear(32, num_slots)
            self.advantage = nn.Linear(32, output_size)
        else:
            # Salida directa
            self.value = nn.Linear(32, output_size)

    def forward(self, x: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, input_size) observación.
            action_mask: (batch, num_slots, actions_per_slot) booleano o float,
                True/1.0 = acción válida, False/0.0 = inválida.
                Si se proporciona, la media de ventaja se calcula solo sobre válidas.

        Returns:
            (batch, output_size) logits (Q-values) de cada acción.
        """
        trunk = self.shared(x)

        if USE_DUELING_DQN:
            # Dueling: Q = V(s) + (A(s,a) - mean(A(s,·)))
            val = self.value(trunk).view(-1, self.num_slots, 1)          # (B, 3, 1)
            adv = self.advantage(trunk).view(-1, self.num_slots, self.actions_per_slot)  # (B, 3, 6)

            if action_mask is not None:
                mask = action_mask.float()  # (B, 3, 6), 1.0 = válida, 0.0 = inválida
                valid_count = mask.sum(dim=2, keepdim=True).clamp(min=1.0)  # (B, 3, 1)
                adv_masked_sum = (adv * mask).sum(dim=2, keepdim=True)       # (B, 3, 1)
                adv_mean = adv_masked_sum / valid_count                     # media solo de válidas
            else:
                adv_mean = adv.mean(dim=2, keepdim=True)

            logits = val + (adv - adv_mean)
            logits = logits.view(-1, self.num_slots * self.actions_per_slot)  # (B, 18)
        else:
            logits = self.value(trunk)

        return logits