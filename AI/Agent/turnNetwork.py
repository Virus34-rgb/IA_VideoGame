"""
Red neuronal para la selección de acciones en el turno (TurnNetwork).

Soporta arquitectura Dueling DQN (separación de valor de estado y ventaja)
y enmascaramiento de acciones inválidas.
"""
import torch
from torch import nn
from AI.Agent.noisy_linear import NoisyLinear
import constants


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
        sigma_init=0.5,
    ) -> None:
        """
        Args:
            input_size: Dimensión de la observación.
            output_size: Total de acciones (num_slots * actions_per_slot).
            num_slots: Número de guerreros en el equipo.
            actions_per_slot: Acciones disponibles por guerrero.
        """
        super().__init__()
        self.num_slots = num_slots
        self.actions_per_slot = actions_per_slot
        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        if constants.USE_DUELING_DQN:
            self.value = NoisyLinear(32, self.num_slots, std_init=sigma_init)        # ← Ruido aquí
            self.advantage = NoisyLinear(32, output_size, std_init=sigma_init)       # ← Y aquí
        else:
            self.value = NoisyLinear(32, output_size, std_init=sigma_init)           # ← O aquí

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
        if constants.USE_DUELING_DQN:
            val = self.value(trunk).view(-1, self.num_slots, 1)
            adv = self.advantage(trunk).view(-1, self.num_slots, self.actions_per_slot)

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
    
    def reset_noise(self):
        if constants.USE_DUELING_DQN:
            self.value.reset_noise()
            self.advantage.reset_noise()
        else:
            self.value.reset_noise()