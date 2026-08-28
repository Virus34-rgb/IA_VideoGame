"""
Red neuronal para la selección de equipo (SelectionNetwork).

Entrada: estado de selección codificado (disposición actual + catálogo + oponente).
Salida: logits para cada acción (guerrero * 3 posiciones).
"""
from torch import nn
import torch


class SelectionNetwork(nn.Module):
    """
    Red para elegir el siguiente guerrero y su posición en la fase de draft.

    Arquitectura simple: 3 capas ocultas (128, 64, 32) y salida lineal.
    """

    def __init__(self, input_size: int = 46, output_size: int = 15) -> None:
        """
        Args:
            input_size: Dimensión del estado de selección.
            output_size: Número de acciones posibles (WARRIOR_QUANTITY * 3).
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_size) estado de selección.

        Returns:
            (batch, output_size) logits por acción.
        """
        return self.network(x)