"""
Fijado de semillas de aleatoriedad (torch/numpy/random) para reproducibilidad.
Extraído de mainV.py -- función única, sin estado ni dependencias del resto
del pipeline.
"""
from typing import Optional

import numpy
import random
import torch


def set_seed(seed: Optional[int]) -> None:
    """Fija las semillas de torch, numpy y random para reproducibilidad.
    Si seed es None, no hace nada (comportamiento no determinista, por defecto)."""
    if seed is None:
        return
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)