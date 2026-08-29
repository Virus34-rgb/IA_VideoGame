"""
Red neuronal para la selección de equipo (SelectionNetwork).

Entrada: estado de selección codificado (disposición actual + catálogo + oponente).
Salida: logits para cada acción (guerrero * 3 posiciones).
"""
from torch import nn
import torch

from AI.Agent.noisy_linear import NoisyLinear


class SelectionNetwork(nn.Module):
    """
    Red para elegir el siguiente guerrero y su posición en la fase de draft.

    Arquitectura simple: 3 capas ocultas (128, 64, 32) y salida lineal.
    """

    def __init__(self, input_size: int = 46, output_size: int = 15, sigma_init=0.5) -> None:
        """
        Args:
            input_size: Dimensión del estado de selección.
            output_size: Número de acciones posibles (WARRIOR_QUANTITY * 3).
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = NoisyLinear(32, output_size, std_init=sigma_init)  # ¡Única capa con ruido!
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_size) estado de selección.

        Returns:
            (batch, output_size) logits por acción.
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)   # El ruido se aplica aquí automáticamente (si training=True)
        return x

    def reset_noise(self):
        self.fc4.reset_noise()   # Solo resetear el ruido de la capa con ruido