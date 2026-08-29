"""
Módulo para codificar el estado de selección de equipo (fase de draft).
"""
import torch

from AI.Agent.observationV import ObservationV
from constants import WARRIOR_QUANTITY, MAX_POOL_SIZE   # AÑADIDO MAX_POOL_SIZE


class ChooseStateV:
    def __init__(self, pl_disposition_ids, pl_warriors, opp_initial_warrior, opp_initial_position):
        self.pl_disposition = pl_disposition_ids
        self.pl_warriors = pl_warriors
        self.opp_initial_warrior = opp_initial_warrior
        self.opp_initialPosition = opp_initial_position

    @staticmethod
    def encode_choose_state_batch(
        pl_disposition: torch.Tensor,
        pl_warriors_ids: torch.Tensor,
        opp_initial_warrior: torch.Tensor,
        opp_initial_position: torch.Tensor,
        catalog_abilities: torch.Tensor,   # NUEVO (N, WARRIOR_QUANTITY, 4)
    ) -> torch.Tensor:
        idx_disp = (pl_disposition - 1).clamp(min=0)
        one_hot_disp = torch.nn.functional.one_hot(idx_disp, num_classes=WARRIOR_QUANTITY).float()
        one_hot_disp = one_hot_disp * (pl_disposition > 0).unsqueeze(-1).float()
        one_hot_disp = one_hot_disp.flatten(start_dim=1)

        idx_w = (pl_warriors_ids - 1).clamp(min=0)
        one_hot_w = torch.nn.functional.one_hot(idx_w, num_classes=WARRIOR_QUANTITY).float()
        one_hot_w = one_hot_w * (pl_warriors_ids > 0).unsqueeze(-1).float()
        one_hot_w = one_hot_w.flatten(start_dim=1)

        idx_opp = (opp_initial_warrior - 1).clamp(min=0)
        one_hot_opp = torch.nn.functional.one_hot(idx_opp, num_classes=WARRIOR_QUANTITY).float()
        one_hot_opp = one_hot_opp * (opp_initial_warrior > 0).unsqueeze(-1).float()

        pos_norm = (opp_initial_position / 3.0).unsqueeze(-1)

        catalog_onehot = torch.nn.functional.one_hot(catalog_abilities, num_classes=MAX_POOL_SIZE).float()
        catalog_onehot = catalog_onehot.flatten(start_dim=1)   # (N, WQ*4*POOL)

        return torch.cat([one_hot_disp, one_hot_w, one_hot_opp, pos_norm, catalog_onehot], dim=-1)