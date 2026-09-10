"""
Decide qué tipo de oponente enfrenta cada partida del lote (self-play normal,
pool de checkpoints, o rusher). Extraído de TrainerV.

"""
import torch

import constants


class OpponentAssignmentService:
    def __init__(self, N: int, opponent_pool, playerRusher):
        self.N = N
        self.opponent_pool = opponent_pool   
        self.playerRusher = playerRusher    

        self._from_pool_mask = torch.zeros(N, dtype=torch.bool)
        self._rusher_mask = torch.zeros(N, dtype=torch.bool)
        self._grouped_opponents = {}

    @property
    def from_pool_mask(self) -> torch.Tensor:
        return self._from_pool_mask

    @property
    def rusher_mask(self) -> torch.Tensor:
        return self._rusher_mask

    @property
    def grouped_opponents(self) -> dict:
        return self._grouped_opponents

    def reset(self):
        self._from_pool_mask = torch.zeros(self.N, dtype=torch.bool)
        self._rusher_mask = torch.zeros(self.N, dtype=torch.bool)
        self._grouped_opponents = {}

    def refresh_pool_assignment(self, player1_elo: float, player_class, environment) -> None:
        from_pool, checkpoint_idx = self.opponent_pool.sample_assignment(self.N, constants.POOL_PORCENTAGE, player1_elo)
        self._from_pool_mask = from_pool
        self._grouped_opponents = (
            self.opponent_pool.build_grouped_opponents(checkpoint_idx, player_class, self.N, environment)
            if from_pool.any() else {}
        )

    def assign_rusher_for_training(self, sample_aggression_fn) -> None:
        """Uso durante entrenamiento normal (self-play): sortea qué partidas
        enfrentan al rusher este lote, evitando solaparse con la pool."""
        rand = torch.rand(self.N)
        self._rusher_mask = (rand < constants.RUSHER_OPPONENT_PERCENTAGE) & ~self._from_pool_mask
        if self._rusher_mask.any():
            self.playerRusher.set_aggression(sample_aggression_fn(self.N))

    def assign_rusher_fixed(self, aggression) -> None:
        """Uso durante evaluación fija contra rusher: TODAS las partidas lo enfrentan."""
        self._rusher_mask = torch.ones(self.N, dtype=torch.bool)
        self.playerRusher.set_aggression(aggression)