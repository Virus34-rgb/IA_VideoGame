"""
Acumulador de métricas de batalla. Extraído de StatsV -- mantiene SOLO el
estado y las operaciones de acumulación; el formateo a texto vive en
AI/Logging/stats_report_writer.py (StatsReportWriter), que recibe una
instancia de esta clase ya poblada, nunca la posee.
"""
import torch
from dataclasses import dataclass
from typing import Dict, Any

import constants


@dataclass
class StatsSummary:
    """Resumen compacto de las estadísticas agregadas."""
    partidas: int
    p1_victories: int
    p2_victories: int
    empates: int
    partidas_por_muerte: int
    partidas_por_limite_turnos: int
    total_turns: int
    p1_damage: float
    p2_damage: float
    p1_succes_blocks: float
    p2_succes_blocks: float
    p1_tot_damage_evaded: float
    p2_tot_damage_evaded: float
    p1_tot_heal: float
    p2_tot_heal: float
    p1_total_deaths: int
    p2_total_deaths: int
    total_reward_p1: float
    total_reward_p2: float
    p1_movements: float
    p2_movements: float

    def to_dict(self) -> Dict[str, Any]:
        partidas = max(self.partidas, 1)
        decisive = self.p1_victories + self.p2_victories
        return {
            "partidas": self.partidas,
            "p1_winrate": self.p1_victories / partidas * 100,
            "p2_winrate": self.p2_victories / partidas * 100,
            "drawrate": self.empates / partidas * 100,
            "p1_win_ratio_excl_draws": (self.p1_victories / decisive * 100) if decisive else 0.0,
            "p2_win_ratio_excl_draws": (self.p2_victories / decisive * 100) if decisive else 0.0,
            "avg_turns": self.total_turns / partidas,
            "partidas_por_muerte_pct": self.partidas_por_muerte / partidas * 100,
            "partidas_por_limite_turnos_pct": self.partidas_por_limite_turnos / partidas * 100,
            "p1_damage_avg": self.p1_damage / partidas,
            "p2_damage_avg": self.p2_damage / partidas,
            "p1_deaths_avg": self.p1_total_deaths / partidas,
            "p2_deaths_avg": self.p2_total_deaths / partidas,
            "p1_success_blocks_avg": self.p1_succes_blocks / partidas,
            "p2_success_blocks_avg": self.p2_succes_blocks / partidas,
            "p1_damage_evaded_avg": self.p1_tot_damage_evaded / partidas,
            "p2_damage_evaded_avg": self.p2_tot_damage_evaded / partidas,
            "p1_reward_avg": self.total_reward_p1 / partidas,
            "p2_reward_avg": self.total_reward_p2 / partidas,
            "p1_tot_heal_avg": self.p1_tot_heal / partidas,
            "p2_tot_heal_avg": self.p2_tot_heal / partidas,
        }


class StatsAccumulator:
    """
    Recolector de estadísticas para el entrenamiento. SOLO acumulación --
    sin ningún método de formateo/escritura a disco.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.partidas: int = 0
        self.p1_victories: int = 0
        self.p2_victories: int = 0
        self.empates: int = 0
        self.partidas_por_muerte: int = 0
        self.partidas_por_limite_turnos: int = 0

        self.p1_damage: float = 0.0
        self.p2_damage: float = 0.0
        self.p1_succes_blocks: float = 0.0
        self.p2_succes_blocks: float = 0.0
        self.p1_tot_damage_evaded: float = 0.0
        self.p2_tot_damage_evaded: float = 0.0
        self.p1_overkill_damage: float = 0.0
        self.p2_overkill_damage: float = 0.0
        self.p1_kill_confirmed: float = 0.0
        self.p2_kill_confirmed: float = 0.0
        self.p1_tot_heal: float = 0.0
        self.p2_tot_heal: float = 0.0
        self.wasted_heal_p1 = 0.0
        self.wasted_heal_p2 = 0.0
        self.wasted_defense_p1 = 0.0
        self.wasted_defense_p2 = 0.0
        self.p1_total_deaths: int = 0
        self.p2_total_deaths: int = 0
        self.total_turns: int = 0
        self.total_reward_p1: float = 0.0
        self.total_reward_p2: float = 0.0

        self.p1_movements: float = 0.0
        self.p2_movements: float = 0.0
        self.p1_strategic_movement: float = 0.0
        self.p2_strategic_movement: float = 0.0

        self._p1_attacks_tensor = torch.zeros(constants.WARRIOR_QUANTITY + 1, constants.MAX_POOL_SIZE)
        self._p2_attacks_tensor = torch.zeros(constants.WARRIOR_QUANTITY + 1, constants.MAX_POOL_SIZE)
        self._p1_warrior_use_tensor = torch.zeros(constants.WARRIOR_QUANTITY)
        self._p2_warrior_use_tensor = torch.zeros(constants.WARRIOR_QUANTITY)
        self._p1_warrior_use_ema = torch.zeros(constants.WARRIOR_QUANTITY)
        self._p2_warrior_use_ema = torch.zeros(constants.WARRIOR_QUANTITY)

        self._p1_damage_batch: torch.Tensor | None = None
        self._p2_damage_batch: torch.Tensor | None = None
        self._p1_blocks_batch: torch.Tensor | None = None
        self._p2_blocks_batch: torch.Tensor | None = None
        self._p1_evaded_batch: torch.Tensor | None = None
        self._p2_evaded_batch: torch.Tensor | None = None
        self._p1_heal_batch: torch.Tensor | None = None
        self._p2_heal_batch: torch.Tensor | None = None

        self.partidas_vs_rusher: int = 0
        self.p1_victories_vs_rusher: int = 0
        self.p2_victories_vs_rusher: int = 0
        self.empates_vs_rusher: int = 0

    def start_batch(self, N: int) -> None:
        self._p1_damage_batch = torch.zeros(N)
        self._p2_damage_batch = torch.zeros(N)
        self._p1_blocks_batch = torch.zeros(N)
        self._p2_blocks_batch = torch.zeros(N)
        self._p1_evaded_batch = torch.zeros(N)
        self._p2_evaded_batch = torch.zeros(N)
        self._p1_heal_batch = torch.zeros(N)
        self._p2_heal_batch = torch.zeros(N)

    def accumulate_turn(
        self, damage_p1, damage_p2, blocks_p1, blocks_p2, avoided_p1, avoided_p2,
        heal_p1, heal_p2, wasted_heal_p1, wasted_heal_p2, wasted_defense_p1, wasted_defense_p2,
        strategic_movement_p1, strategic_movement_p2, overkill_damage_p1, overkill_damage_p2,
        kill_confirmed_p1, kill_confirmed_p2, ya_terminadas_antes,
    ) -> None:
        activa = ~ya_terminadas_antes
        self._p1_damage_batch += torch.where(activa, damage_p1, torch.zeros_like(damage_p1))
        self._p2_damage_batch += torch.where(activa, damage_p2, torch.zeros_like(damage_p2))
        self._p1_blocks_batch += torch.where(activa, blocks_p1, torch.zeros_like(blocks_p1))
        self._p2_blocks_batch += torch.where(activa, blocks_p2, torch.zeros_like(blocks_p2))
        self._p1_evaded_batch += torch.where(activa, avoided_p1, torch.zeros_like(avoided_p1))
        self._p2_evaded_batch += torch.where(activa, avoided_p2, torch.zeros_like(avoided_p2))
        self._p1_heal_batch += torch.where(activa, heal_p1, torch.zeros_like(heal_p1))
        self._p2_heal_batch += torch.where(activa, heal_p2, torch.zeros_like(heal_p2))
        self.wasted_heal_p1 += torch.where(activa, wasted_heal_p1, torch.zeros_like(wasted_heal_p1)).sum().item()
        self.wasted_heal_p2 += torch.where(activa, wasted_heal_p2, torch.zeros_like(wasted_heal_p2)).sum().item()
        self.wasted_defense_p1 += torch.where(activa, wasted_defense_p1, torch.zeros_like(wasted_defense_p1)).sum().item()
        self.wasted_defense_p2 += torch.where(activa, wasted_defense_p2, torch.zeros_like(wasted_defense_p2)).sum().item()
        self.p1_strategic_movement += torch.where(activa, strategic_movement_p1, torch.zeros_like(strategic_movement_p1)).sum().item()
        self.p2_strategic_movement += torch.where(activa, strategic_movement_p2, torch.zeros_like(strategic_movement_p2)).sum().item()
        self.p1_overkill_damage += torch.where(activa, overkill_damage_p1, torch.zeros_like(overkill_damage_p1)).sum().item()
        self.p2_overkill_damage += torch.where(activa, overkill_damage_p2, torch.zeros_like(overkill_damage_p2)).sum().item()
        self.p1_kill_confirmed += torch.where(activa, kill_confirmed_p1, torch.zeros_like(kill_confirmed_p1)).sum().item()
        self.p2_kill_confirmed += torch.where(activa, kill_confirmed_p2, torch.zeros_like(kill_confirmed_p2)).sum().item()

    def accumulate_movements(self, moved, es_p1, activa) -> None:
        mask = activa.float()
        self.p1_movements += (torch.where(es_p1, moved, torch.zeros_like(moved)) * mask).sum().item()
        self.p2_movements += (torch.where(~es_p1, moved, torch.zeros_like(moved)) * mask).sum().item()

    def accumulate_attacks(self, tipo_actor, accion_actor, es_p1, activa) -> None:
        es_habilidad = (accion_actor >= 0) & (accion_actor < constants.MAX_POOL_SIZE)
        mask = es_habilidad & activa
        mask_p1 = mask & es_p1
        mask_p2 = mask & ~es_p1

        if mask_p1.any():
            idx = tipo_actor[mask_p1] * constants.MAX_POOL_SIZE + accion_actor[mask_p1]
            counts = torch.bincount(idx, minlength=(constants.WARRIOR_QUANTITY + 1) * constants.MAX_POOL_SIZE)
            self._p1_attacks_tensor += counts.view(constants.WARRIOR_QUANTITY + 1, constants.MAX_POOL_SIZE).float()

        if mask_p2.any():
            idx = tipo_actor[mask_p2] * constants.MAX_POOL_SIZE + accion_actor[mask_p2]
            counts = torch.bincount(idx, minlength=(constants.WARRIOR_QUANTITY + 1) * constants.MAX_POOL_SIZE)
            self._p2_attacks_tensor += counts.view(constants.WARRIOR_QUANTITY + 1, constants.MAX_POOL_SIZE).float()

    def accumulate_warrior_use(self, warrior1, warrior2) -> None:
        c1 = torch.bincount(warrior1, minlength=constants.WARRIOR_QUANTITY + 1)[1:]
        c2 = torch.bincount(warrior2, minlength=constants.WARRIOR_QUANTITY + 1)[1:]
        self._p1_warrior_use_tensor += c1.float()
        self._p2_warrior_use_tensor += c2.float()

        decay = constants.WARRIOR_USE_EMA_DECAY
        prop1 = c1.float() / c1.sum().clamp(min=1.0)
        prop2 = c2.float() / c2.sum().clamp(min=1.0)
        self._p1_warrior_use_ema = decay * self._p1_warrior_use_ema + (1 - decay) * prop1
        self._p2_warrior_use_ema = decay * self._p2_warrior_use_ema + (1 - decay) * prop2

    def accumulate_rusher_stats(self, winner, rusher_mask) -> None:
        self.partidas_vs_rusher += torch.where(rusher_mask, 1, 0).sum()
        self.p1_victories_vs_rusher += (rusher_mask & (winner == 0)).sum().item()
        self.empates_vs_rusher += (rusher_mask & (winner == 2)).sum().item()
        self.p2_victories_vs_rusher = self.partidas_vs_rusher - self.p1_victories_vs_rusher - self.empates_vs_rusher

    def close_finished_games(
        self, termina_ahora, winner, p1_deaths, p2_deaths, turn_number, por_muerte_mask, por_turnos_mask,
    ) -> None:
        n_cerradas = termina_ahora.sum().item()
        if n_cerradas == 0:
            return

        idx = termina_ahora.nonzero(as_tuple=True)[0]

        self.partidas += n_cerradas
        self.p1_victories += (termina_ahora & (winner == 0)).sum().item()
        self.p2_victories += (termina_ahora & (winner == 1)).sum().item()
        self.empates += (termina_ahora & (winner == 2)).sum().item()
        self.partidas_por_muerte += (termina_ahora & por_muerte_mask).sum().item()
        self.partidas_por_limite_turnos += (termina_ahora & por_turnos_mask).sum().item()

        self.p1_total_deaths += p1_deaths[idx].sum().item()
        self.p2_total_deaths += p2_deaths[idx].sum().item()
        self.total_turns += turn_number[idx].sum().item()

        self.p1_damage += self._p1_damage_batch[idx].sum().item()
        self.p2_damage += self._p2_damage_batch[idx].sum().item()
        self.p1_succes_blocks += self._p1_blocks_batch[idx].sum().item()
        self.p2_succes_blocks += self._p2_blocks_batch[idx].sum().item()
        self.p1_tot_damage_evaded += self._p1_evaded_batch[idx].sum().item()
        self.p2_tot_damage_evaded += self._p2_evaded_batch[idx].sum().item()
        self.p1_tot_heal += self._p1_heal_batch[idx].sum().item()
        self.p2_tot_heal += self._p2_heal_batch[idx].sum().item()

    def build_summary(self) -> StatsSummary:
        """Público (antes _build_summary, privado en StatsV) -- StatsReportWriter
        necesita invocarlo desde fuera, así que pasa a formar parte del contrato
        público de este colaborador."""
        return StatsSummary(
            partidas=self.partidas, p1_victories=self.p1_victories, p2_victories=self.p2_victories,
            empates=self.empates, partidas_por_muerte=self.partidas_por_muerte,
            partidas_por_limite_turnos=self.partidas_por_limite_turnos, total_turns=self.total_turns,
            p1_damage=self.p1_damage, p2_damage=self.p2_damage,
            p1_succes_blocks=self.p1_succes_blocks, p2_succes_blocks=self.p2_succes_blocks,
            p1_tot_damage_evaded=self.p1_tot_damage_evaded, p2_tot_damage_evaded=self.p2_tot_damage_evaded,
            p1_tot_heal=self.p1_tot_heal, p2_tot_heal=self.p2_tot_heal,
            p1_total_deaths=self.p1_total_deaths, p2_total_deaths=self.p2_total_deaths,
            total_reward_p1=self.total_reward_p1, total_reward_p2=self.total_reward_p2,
            p1_movements=self.p1_movements, p2_movements=self.p2_movements,
        )

    def __len__(self) -> int:
        return self.partidas