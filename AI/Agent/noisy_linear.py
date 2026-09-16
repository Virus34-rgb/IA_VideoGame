"""
Noisy Linear Layer with parametric noise.

Implementación basada en:
  https://github.com/ray-project/ray/blob/ray-2.22.0/rllib/algorithms/dqn/torch/torch_noisy_linear.py
  "Noisy Networks for Exploration", https://arxiv.org/abs/1706.10295v3

Esta versión NO cachea los pesos efectivos: `weight` y `bias` se computan
en cada forward. El cacheo previo tenía un bug crítico — el decorador
`@torch.no_grad()` en `_refresh_cache` sacaba los pesos efectivos del grafo
de cómputo y ningún gradiente llegaba a `weight_mu`/`weight_sigma`. La capa
final de la red nunca aprendía.

El coste de recomputar (una suma y un producto por forward) es despreciable
frente al matmul que viene después, así que no merece la pena cachear.
"""
import math
import torch
from torch import nn
from typing import Optional, Sequence, Union

DEVICE_TYPING = Union[torch.device, str, int]


class NoisyLinear(nn.Linear):
    """
    Capa lineal con ruido gaussiano factorizado añadido a los pesos.

    Añade estocasticidad paramétrica a la red. Los parámetros del ruido
    (mu y sigma) se aprenden por gradient descent junto al resto de pesos.
    El ruido se re-muestrea llamando a `reset_noise()` (típicamente una vez
    por decisión o por lote, según `constants.RESET_IN_DECISIONS`).

    Args:
        in_features: Dimensión de entrada.
        out_features: Dimensión de salida.
        bias: Si `True`, añade bias. Default `True`.
        device: Device del layer. Default `None` (CPU).
        dtype: dtype de los parámetros. Default `None` (default de torch).
        std_init: Desviación estándar inicial del ruido. Default `0.1`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[DEVICE_TYPING] = None,
        dtype: Optional[torch.dtype] = None,
        std_init: float = 0.1,
    ):
        nn.Module.__init__(self)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.std_init = std_init

        self.weight_mu = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.weight_sigma = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.register_buffer(
            "weight_epsilon",
            torch.empty(out_features, in_features, device=device, dtype=dtype),
        )
        if bias:
            self.bias_mu = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.bias_sigma = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.register_buffer(
                "bias_epsilon",
                torch.empty(out_features, device=device, dtype=dtype),
            )
        else:
            self.bias_mu = None
        self.reset_parameters()
        self.reset_noise()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Inicialización estándar de NoisyLinear (factorized Gaussian)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        if self.bias_mu is not None:
            self.bias_mu.data.zero_()
            self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @torch.no_grad()
    def reset_noise(self) -> None:
        """Re-muestrea los tensores epsilon (factorized Gaussian noise)."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        if self.bias_mu is not None:
            self.bias_epsilon.copy_(epsilon_out)

    @torch.no_grad()
    def _scale_noise(self, size: Union[int, torch.Size, Sequence]) -> torch.Tensor:
        """Ruido gaussiano escalado: sign(x) * sqrt(|x|)."""
        if isinstance(size, int):
            size = (size,)
        x = torch.randn(*size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    @property
    def weight(self) -> torch.Tensor:
        """
        Peso efectivo, computado en cada acceso para que el backward
        propague gradiente a weight_mu/weight_sigma.
        """
        if not self.training:
            return self.weight_mu
        return self.weight_mu + self.weight_sigma * self.weight_epsilon

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Bias efectivo, computado en cada acceso. Mismo criterio que weight."""
        if self.bias_mu is None:
            return None
        if not self.training:
            return self.bias_mu
        return self.bias_mu + self.bias_sigma * self.bias_epsilon

    @torch.no_grad()
    def clamp_sigma(self, min_sigma: float) -> None:
        """Recorta weight_sigma/bias_sigma a un mínimo absoluto para evitar
        que el optimizador los colapse hacia 0 (y con ello, la exploración)."""
        self.weight_sigma.data.clamp_(min=min_sigma)
        if self.bias_sigma is not None:
            self.bias_sigma.data.clamp_(min=min_sigma)

    @torch.no_grad()
    def mean_abs_sigma(self) -> float:
        """Media de |sigma| de esta capa, para logging/diagnóstico."""
        return self.weight_sigma.data.abs().mean().item()