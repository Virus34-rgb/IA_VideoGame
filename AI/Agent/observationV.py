"""
Módulo para normalizar observaciones del entorno.
"""
import torch
from constants import MAX_TURNS, WARRIOR_QUANTITY


class ObservationV:
    """
    Codifica la observación del entorno para la toma de decisiones en un turno.
    """

    def __init__(self, pl_types, pl_alive, pl_speed_norm, pl_health_norm,
                 pl_cooldowns, opp_life, opp_disposition, turn):
        self.pl_types = pl_types
        self.pl_alive = pl_alive
        self.pl_speed_norm = pl_speed_norm
        self.pl_health_norm = pl_health_norm
        self.pl_cooldowns = pl_cooldowns
        self.opp_life = opp_life
        self.opp_disposition = opp_disposition
        self.turn = turn

    @staticmethod
    def id_to_one_hot(warrior_id: int, warrior_quantity: int) -> list[int]:
        """Convierte un ID de guerrero (1..WQ) en one-hot, o todo 0 si warrior_id=0."""
        result = [0] * warrior_quantity
        if warrior_id:
            result[warrior_id - 1] = 1
        return result

    @staticmethod
    def normalize_abilities(cooldowns_4: list[bool]) -> list[int]:
        """
        Codifica los cooldowns de 4 habilidades en 6 valores:
        - Los primeros 4 indican si la habilidad está disponible (1) o en cooldown (0).
        - Los últimos 2 son siempre 0 (movimientos, se manejan aparte).
        """
        encoded = [0, 0, 0, 0, 0, 0]
        for position, en_cooldown in enumerate(cooldowns_4):
            encoded[position] = 0 if en_cooldown else 1
        # Movimientos siempre disponibles en la observación, pero la máscara los maneja
        return encoded

    @staticmethod
    def normalize_batch(
        pl_types: torch.Tensor,           # (N, 3)
        pl_alive: torch.Tensor,           # (N, 3) bool
        pl_speed_norm: torch.Tensor,      # (N, 3)
        pl_health_norm: torch.Tensor,     # (N, 3)
        pl_cooldowns: torch.Tensor,       # (N, 3, 4) bool
        opp_life: torch.Tensor,           # (N, 3)
        opp_disposition: torch.Tensor,    # (N, 3)
        turn_norm: torch.Tensor,          # (N,)
    ) -> torch.Tensor:
        """
        Versión vectorizada para N partidas en paralelo.
        Retorna un tensor de forma (N, 3*(WQ+8) + 3*WQ + 4) aproximadamente.
        """
        N = pl_types.shape[0]

        # One-hot del tipo propio (por slot)
        idx = (pl_types - 1).clamp(min=0)  # (N,3)
        one_hot_own = torch.nn.functional.one_hot(idx, num_classes=WARRIOR_QUANTITY).float()
        one_hot_own = one_hot_own * pl_alive.unsqueeze(-1).float()

        # Velocidad y vida normalizadas (0 si muerto)
        speed = torch.where(pl_alive, pl_speed_norm, torch.zeros_like(pl_speed_norm)).unsqueeze(-1)
        health = torch.where(pl_alive, pl_health_norm, torch.zeros_like(pl_health_norm)).unsqueeze(-1)

        # Cooldowns: 1 = usable, 0 = en cooldown (o muerto)
        cd_usable = (~pl_cooldowns).float()
        cd_usable = torch.where(pl_alive.unsqueeze(-1), cd_usable, torch.zeros_like(cd_usable))
        extra_zeros = torch.zeros(N, 3, 2)  # para movimientos (se codifican siempre 0)

        # Concatenar por slot: (WQ, speed, health, cd_usable[4], extra[2]) -> WQ+2+6
        propio = torch.cat([one_hot_own, speed, health, cd_usable, extra_zeros], dim=-1)
        propio = propio.flatten(start_dim=1)  # (N, 3*(WQ+8))

        # One-hot del oponente (disposición)
        idx_opp = (opp_disposition - 1).clamp(min=0)
        one_hot_opp = torch.nn.functional.one_hot(idx_opp, num_classes=WARRIOR_QUANTITY).float()
        one_hot_opp = one_hot_opp * (opp_disposition > 0).unsqueeze(-1).float()
        one_hot_opp = one_hot_opp.flatten(start_dim=1)  # (N, 3*WQ)

        return torch.cat([propio, opp_life, one_hot_opp, turn_norm.unsqueeze(-1)], dim=-1)