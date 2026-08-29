"""
Red neuronal para la selección de acciones en el turno (TurnNetwork).
"""
import torch
from torch import nn
from AI.Agent.noisy_linear import NoisyLinear
import constants


class TurnNetwork(nn.Module):
    def __init__(
        self,
        input_size: int = constants.TURN_STATE_DIM,   # CAMBIADO: antes 58, fijo
        output_size: int = 18,
        num_slots: int = 3,
        actions_per_slot: int = 6,
        sigma_init=0.5,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.actions_per_slot = actions_per_slot
        self.shared = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        if constants.USE_DUELING_DQN:
            self.value = NoisyLinear(32, self.num_slots, std_init=sigma_init)
            self.advantage = NoisyLinear(32, output_size, std_init=sigma_init)
        else:
            self.value = NoisyLinear(32, output_size, std_init=sigma_init)

    def forward(self, x: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        trunk = self.shared(x)
        if constants.USE_DUELING_DQN:
            val = self.value(trunk).view(-1, self.num_slots, 1)
            adv = self.advantage(trunk).view(-1, self.num_slots, self.actions_per_slot)
            if action_mask is not None:
                mask = action_mask.float()
                valid_count = mask.sum(dim=2, keepdim=True).clamp(min=1.0)
                adv_masked_sum = (adv * mask).sum(dim=2, keepdim=True)
                adv_mean = adv_masked_sum / valid_count
            else:
                adv_mean = adv.mean(dim=2, keepdim=True)
            logits = val + (adv - adv_mean)
            logits = logits.view(-1, self.num_slots * self.actions_per_slot)
        else:
            logits = self.value(trunk)
        return logits

    def reset_noise(self):
        if constants.USE_DUELING_DQN:
            self.value.reset_noise()
            self.advantage.reset_noise()
        else:
            self.value.reset_noise()