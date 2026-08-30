"""
Red neuronal para la selección de equipo (SelectionNetwork).
"""
from torch import nn
import torch

from AI.Agent.noisy_linear import NoisyLinear
import constants


class SelectionNetwork(nn.Module):
    def __init__(self, input_size: int = constants.SELECTION_STATE_DIM, output_size: int = None, sigma_init=0.5) -> None:
        super().__init__()
        if output_size is None:
            output_size = constants.MAX_CASTLE_SIZE * 3 if constants.USE_META_GAME else constants.WARRIOR_QUANTITY * 3
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = NoisyLinear(32, output_size, std_init=sigma_init)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def reset_noise(self):
        self.fc4.reset_noise()