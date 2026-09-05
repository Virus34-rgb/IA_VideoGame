"""
Módulo para codificar el estado de selección de equipo (fase de draft).
Soporta dos modos, elegidos por constants.USE_META_GAME:
  - Catálogo: 5 tipos fijos por lote (modo histórico, sin meta-juego).
  - Castillo: hasta MAX_CASTLE_SIZE instancias propias del castillo de esa partida.
"""
import torch

import constants


class ChooseStateV:
    def __init__(self, pl_disposition_ids, pl_warriors, opp_initial_warrior, opp_initial_position):
        self.pl_disposition = pl_disposition_ids
        self.pl_warriors = pl_warriors
        self.opp_initial_warrior = opp_initial_warrior
        self.opp_initialPosition = opp_initial_position

    # ------------------------------------------------------------
    # Modo catálogo (histórico, sin meta-juego)
    # ------------------------------------------------------------
    @staticmethod
    def encode_choose_state_batch_catalog(
        pl_disposition: torch.Tensor,
        pl_warriors_ids: torch.Tensor,
        opp_initial_warrior: torch.Tensor,
        opp_initial_position: torch.Tensor,
        catalog_abilities: torch.Tensor,   # (N, WARRIOR_QUANTITY, 4)
    ) -> torch.Tensor:
        idx_disp = (pl_disposition - 1).clamp(min=0)
        one_hot_disp = torch.nn.functional.one_hot(idx_disp, num_classes=constants.WARRIOR_QUANTITY).float()
        one_hot_disp = one_hot_disp * (pl_disposition > 0).unsqueeze(-1).float()
        one_hot_disp = one_hot_disp.flatten(start_dim=1)

        idx_w = (pl_warriors_ids - 1).clamp(min=0)
        one_hot_w = torch.nn.functional.one_hot(idx_w, num_classes=constants.WARRIOR_QUANTITY).float()
        one_hot_w = one_hot_w * (pl_warriors_ids > 0).unsqueeze(-1).float()
        one_hot_w = one_hot_w.flatten(start_dim=1)

        idx_opp = (opp_initial_warrior - 1).clamp(min=0)
        one_hot_opp = torch.nn.functional.one_hot(idx_opp, num_classes=constants.WARRIOR_QUANTITY).float()
        one_hot_opp = one_hot_opp * (opp_initial_warrior > 0).unsqueeze(-1).float()

        pos_norm = (opp_initial_position / 3.0).unsqueeze(-1)

        catalog_onehot = torch.nn.functional.one_hot(catalog_abilities, num_classes=constants.MAX_POOL_SIZE).float()
        catalog_onehot = catalog_onehot.flatten(start_dim=1)

        return torch.cat([one_hot_disp, one_hot_w, one_hot_opp, pos_norm, catalog_onehot], dim=-1)

    # ------------------------------------------------------------
    # Modo castillo (meta-juego)
    # ------------------------------------------------------------
    @staticmethod
    def encode_choose_state_batch_castle(
        castle_types: torch.Tensor,             # (N, MAX_CASTLE_SIZE)
        castle_abilities: torch.Tensor,          # (N, MAX_CASTLE_SIZE, 4)
        castle_abilities_levels: torch.Tensor,   # (N, MAX_CASTLE_SIZE, 4)
        battle_fought: torch.Tensor,             # (N, MAX_CASTLE_SIZE)
        castle_alive: torch.Tensor,              # (N, MAX_CASTLE_SIZE)
        gold: torch.Tensor,                      # (N,)
        opp_initial_warrior: torch.Tensor,
        opp_initial_position: torch.Tensor,
    ) -> torch.Tensor:
        
        # Al inicio de encode_choose_state_batch_castle
        if not isinstance(opp_initial_warrior, torch.Tensor):
            opp_initial_warrior = torch.tensor(opp_initial_warrior, dtype=torch.long)
        if not isinstance(opp_initial_position, torch.Tensor):
            opp_initial_position = torch.tensor(opp_initial_position, dtype=torch.float)

        # Asegurar que tengan al menos dimensión 1 (batch)
        if opp_initial_warrior.dim() == 0:
            opp_initial_warrior = opp_initial_warrior.unsqueeze(0)
        if opp_initial_position.dim() == 0:
            opp_initial_position = opp_initial_position.unsqueeze(0)
            
        N = castle_types.shape[0]

        idx = (castle_types - 1).clamp(min=0)
        one_hot_types = torch.nn.functional.one_hot(idx, num_classes=constants.WARRIOR_QUANTITY).float()
        one_hot_types = one_hot_types * castle_alive.unsqueeze(-1).float()
        one_hot_types_flat = one_hot_types.flatten(start_dim=1)

        abilities_onehot = torch.nn.functional.one_hot(castle_abilities, num_classes=constants.MAX_POOL_SIZE).float()
        abilities_onehot = abilities_onehot * castle_alive.view(N, constants.MAX_CASTLE_SIZE, 1, 1).float()
        abilities_flat = abilities_onehot.flatten(start_dim=1)

        levels_norm = (castle_abilities_levels.float() / constants.MAX_ABILITY_LEVEL)
        levels_norm = levels_norm * castle_alive.view(N, constants.MAX_CASTLE_SIZE, 1).float()
        levels_flat = levels_norm.flatten(start_dim=1)

        age_norm = (battle_fought.float() / constants.MAX_BATALLAS) * castle_alive.float()

        gold_norm = (gold.float() / constants.GOLD_NORM_REF).unsqueeze(-1)

        idx_opp = (opp_initial_warrior - 1).clamp(min=0)
        one_hot_opp = torch.nn.functional.one_hot(idx_opp, num_classes=constants.WARRIOR_QUANTITY).float()
        one_hot_opp = one_hot_opp * (opp_initial_warrior > 0).unsqueeze(-1).float()
        pos_norm = (opp_initial_position / 3.0).unsqueeze(-1)

        return torch.cat(
            [one_hot_types_flat, abilities_flat, levels_flat, age_norm, gold_norm, one_hot_opp, pos_norm],
            dim=-1,
        )