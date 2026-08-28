"""
Módulo para codificar el estado de selección de equipo (fase de draft).
Proporciona versiones vectorizadas (batch) y no vectorizadas (single).
"""
import torch

from AI.Agent.observationV import ObservationV
from constants import WARRIOR_QUANTITY


class ChooseStateV:
    """
    Representa el estado visible para la selección de guerreros.
    Utilizado para la fase de draft donde se eligen 3 héroes de entre los disponibles.
    """

    def __init__(
        self,
        pl_disposition_ids: list[int],
        pl_warriors: list[int],
        opp_initial_warrior: int,
        opp_initial_position: int,
    ) -> None:
        """
        Args:
            pl_disposition_ids: IDs de los héroes ya colocados (3 posiciones, 0 si vacío).
            pl_warriors: IDs de todos los héroes disponibles para seleccionar.
            opp_initial_warrior: ID del primer héroe que el oponente ha seleccionado (0 si ninguno).
            opp_initial_position: Posición donde el oponente colocó su primer héroe (0-2).
        """
        self.pl_disposition = pl_disposition_ids
        self.pl_warriors = pl_warriors
        self.opp_initial_warrior = opp_initial_warrior
        self.opp_initialPosition = opp_initial_position

    @staticmethod
    def encode_choose_state_batch(
        pl_disposition: torch.Tensor,        # (N, 3)
        pl_warriors_ids: torch.Tensor,       # (N, WARRIOR_QUANTITY)  catálogo completo (one‑hot o IDs)
        opp_initial_warrior: torch.Tensor,   # (N,)
        opp_initial_position: torch.Tensor,  # (N,)
    ) -> torch.Tensor:
        """
        Devuelve un tensor de forma (N, dim_estado) con la codificación concatenada.

        El estado codificado contiene:
        - One‑hot de la disposición actual (3 * WARRIOR_QUANTITY)
        - One‑hot de todos los guerreros disponibles (WARRIOR_QUANTITY * WARRIOR_QUANTITY)
        - One‑hot del primer guerrero del oponente (WARRIOR_QUANTITY)
        - Posición del oponente normalizada (1)
        """
        # One‑hot de la disposición actual
        idx_disp = (pl_disposition - 1).clamp(min=0)  # (N,3)
        one_hot_disp = torch.nn.functional.one_hot(idx_disp, num_classes=WARRIOR_QUANTITY).float()
        one_hot_disp = one_hot_disp * (pl_disposition > 0).unsqueeze(-1).float()
        one_hot_disp = one_hot_disp.flatten(start_dim=1)  # (N, 3*WQ)

        # One‑hot de los guerreros disponibles (catálogo)
        # pl_warriors_ids se asume que es un tensor (N, WQ) con IDs o 0 para no disponible
        idx_w = (pl_warriors_ids - 1).clamp(min=0)  # (N, WQ)
        one_hot_w = torch.nn.functional.one_hot(idx_w, num_classes=WARRIOR_QUANTITY).float()
        one_hot_w = one_hot_w * (pl_warriors_ids > 0).unsqueeze(-1).float()
        one_hot_w = one_hot_w.flatten(start_dim=1)  # (N, WQ*WQ)

        # One‑hot del primer guerrero del oponente
        idx_opp = (opp_initial_warrior - 1).clamp(min=0)
        one_hot_opp = torch.nn.functional.one_hot(idx_opp, num_classes=WARRIOR_QUANTITY).float()
        one_hot_opp = one_hot_opp * (opp_initial_warrior > 0).unsqueeze(-1).float()

        # Normalizar posición (0-2) a [0,1]
        pos_norm = (opp_initial_position / 3.0).unsqueeze(-1)  # (N,1)

        return torch.cat([one_hot_disp, one_hot_w, one_hot_opp, pos_norm], dim=-1)