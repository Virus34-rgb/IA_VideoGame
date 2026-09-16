"""
Configuración común de tests.

Nota sobre determinismo:
  El proyecto entero usa CPU y no tiene GPU. Para que los tests de
  regresión dorada sean reproducibles, este conftest fuerza:
    - torch.set_num_threads(1) — evita reordenación de reducciones BLAS
    - numpy/random/torch seeds fijas — no deja basura entre tests
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic_environment():
    """
    Fixture autouse: se aplica a TODOS los tests del directorio. Fuerza
    single-thread y semillas por defecto antes/después de cada test, para
    que el orden de ejecución no afecte a los resultados.
    """
    prev_threads = torch.get_num_threads()
    prev_torch_state = torch.get_rng_state()
    prev_numpy_state = np.random.get_state()
    prev_random_state = random.getstate()

    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    try:
        yield
    finally:
        torch.set_num_threads(prev_threads)
        torch.set_rng_state(prev_torch_state)
        np.random.set_state(prev_numpy_state)
        random.setstate(prev_random_state)