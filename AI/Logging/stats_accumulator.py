"""
Módulo de estadísticas para el entorno vectorizado.

Acumula métricas de batalla (daño, cura, bloqueos, movimientos, selección de héroes, etc.)
y genera informes en texto plano.
"""
import torch
from dataclasses import dataclass
from typing import Dict, List, Any

import constants


@dataclass
class StatsAccumulator:
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
        """Convierte el resumen a un diccionario con métricas derivadas."""
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


class StatsV:
    """
    Recolector y generador de estadísticas para el entrenamiento.

    Acumula métricas por lote (N partidas en paralelo) y al final genera
    un informe de texto con resultados agregados.
    """

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------
    # Inicialización y reseteo
    # ------------------------------------------------------------

    def reset(self) -> None:
        """Reinicia todas las estadísticas acumuladas."""
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

        # Tensores acumuladores por tipo de guerrero y habilidad
        self._p1_attacks_tensor = torch.zeros(constants.WARRIOR_QUANTITY + 1, constants.MAX_POOL_SIZE)
        self._p2_attacks_tensor = torch.zeros(constants.WARRIOR_QUANTITY + 1, constants.MAX_POOL_SIZE)
        self._p1_warrior_use_tensor = torch.zeros(constants.WARRIOR_QUANTITY)
        self._p2_warrior_use_tensor = torch.zeros(constants.WARRIOR_QUANTITY)
        self._p1_warrior_use_ema = torch.zeros(constants.WARRIOR_QUANTITY) 
        self._p2_warrior_use_ema = torch.zeros(constants.WARRIOR_QUANTITY)

        # Buffers por batch (se reinician en start_batch)
        self._p1_damage_batch: torch.Tensor | None = None
        self._p2_damage_batch: torch.Tensor | None = None
        self._p1_blocks_batch: torch.Tensor | None = None
        self._p2_blocks_batch: torch.Tensor | None = None
        self._p1_evaded_batch: torch.Tensor | None = None
        self._p2_evaded_batch: torch.Tensor | None = None
        self._p1_heal_batch: torch.Tensor | None = None
        self._p2_heal_batch: torch.Tensor | None = None
        
        #Buffers rusher
        self.partidas_vs_rusher: int = 0
        self.p1_victories_vs_rusher: int = 0
        self.p2_victories_vs_rusher: int = 0
        self.empates_vs_rusher: int = 0

    def start_batch(self, N: int) -> None:
        """
        Inicializa los buffers para un nuevo lote de N partidas.
        Debe llamarse al comenzar cada lote en VectorizedEnvironment.reset().
        """
        self._p1_damage_batch = torch.zeros(N)
        self._p2_damage_batch = torch.zeros(N)
        self._p1_blocks_batch = torch.zeros(N)
        self._p2_blocks_batch = torch.zeros(N)
        self._p1_evaded_batch = torch.zeros(N)
        self._p2_evaded_batch = torch.zeros(N)
        self._p1_heal_batch = torch.zeros(N)
        self._p2_heal_batch = torch.zeros(N)

    # ------------------------------------------------------------
    # Acumulación de métricas por turno
    # ------------------------------------------------------------

    def accumulate_turn(
        self,
        damage_p1: torch.Tensor,
        damage_p2: torch.Tensor,
        blocks_p1: torch.Tensor,
        blocks_p2: torch.Tensor,
        avoided_p1: torch.Tensor,
        avoided_p2: torch.Tensor,
        heal_p1: torch.Tensor,
        heal_p2: torch.Tensor,
        wasted_heal_p1,
        wasted_heal_p2,
        wasted_defense_p1,
        wasted_defense_p2,
        strategic_movement_p1,
        strategic_movement_p2,
        overkill_damage_p1,
        overkill_damage_p2,
        kill_confirmed_p1,
        kill_confirmed_p2,
        ya_terminadas_antes: torch.Tensor,
    ) -> None:
        """
        Acumula las métricas de un turno para todas las partidas activas.

        Args:
            damage_p1, damage_p2: (N,) daño infligido por cada jugador.
            blocks_p1, blocks_p2: (N,) bloqueos exitosos.
            avoided_p1, avoided_p2: (N,) daño evitado (por bloqueo/defensa).
            heal_p1, heal_p2: (N,) curación realizada.
            ya_terminadas_antes: (N,) bool, True para partidas que ya habían terminado
                antes de este turno (sus métricas se ignoran).
        """
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

    def accumulate_movements(self, moved: torch.Tensor, es_p1: torch.Tensor, activa: torch.Tensor) -> None:
        """
        Acumula movimientos realizados en un turno.

        Args:
            moved: (N,) float, 1 si el actor actual se movió, 0 en caso contrario.
            es_p1: (N,) bool, True si el actor actual es P1, False si es P2.
            activa: (N,) bool, True para partidas no terminadas antes de este turno.
        """
        mask = activa.float()
        self.p1_movements += (torch.where(es_p1, moved, torch.zeros_like(moved)) * mask).sum().item()
        self.p2_movements += (torch.where(~es_p1, moved, torch.zeros_like(moved)) * mask).sum().item()

    def accumulate_attacks(
        self,
        tipo_actor: torch.Tensor,
        accion_actor: torch.Tensor,
        es_p1: torch.Tensor,
        activa: torch.Tensor,
    ) -> None:
        """
        Acumula el uso de habilidades de ataque por tipo de guerrero.

        Args:
            tipo_actor: (N,) ID del guerrero que realiza la acción (1..WARRIOR_QUANTITY).
            accion_actor: (N,) índice de acción (0-3 = habilidad, 5/6 = movimiento).
            es_p1: (N,) bool, True si el actor es P1.
            activa: (N,) bool, True para partidas no terminadas antes de este turno.
        """
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

    def accumulate_warrior_use(self, warrior1: torch.Tensor, warrior2: torch.Tensor) -> None:
        c1 = torch.bincount(warrior1, minlength=constants.WARRIOR_QUANTITY + 1)[1:]
        c2 = torch.bincount(warrior2, minlength=constants.WARRIOR_QUANTITY + 1)[1:]
        self._p1_warrior_use_tensor += c1.float()
        self._p2_warrior_use_tensor += c2.float()

        # NUEVO: actualizar EMA de uso reciente (proporciones normalizadas por batch)
        decay = constants.WARRIOR_USE_EMA_DECAY
        prop1 = c1.float() / c1.sum().clamp(min=1.0)
        prop2 = c2.float() / c2.sum().clamp(min=1.0)
        self._p1_warrior_use_ema = decay * self._p1_warrior_use_ema + (1 - decay) * prop1
        self._p2_warrior_use_ema = decay * self._p2_warrior_use_ema + (1 - decay) * prop2
        
    def accumulate_rusher_stats(self,winner:torch.tensor,rusher_mask: torch.tensor):
        self.partidas_vs_rusher += torch.where(rusher_mask,1,0).sum()
        self.p1_victories_vs_rusher += (rusher_mask & (winner == 0)).sum().item()
        self.empates_vs_rusher += (rusher_mask & (winner == 2)).sum().item()
        self.p2_victories_vs_rusher = self.partidas_vs_rusher -self.p1_victories_vs_rusher -self.empates_vs_rusher

    # ------------------------------------------------------------
    # Cierre de partidas finalizadas
    # ------------------------------------------------------------

    def close_finished_games(
        self,
        termina_ahora: torch.Tensor,
        winner: torch.Tensor,
        p1_deaths: torch.Tensor,
        p2_deaths: torch.Tensor,
        turn_number: torch.Tensor,
        por_muerte_mask: torch.Tensor,
        por_turnos_mask: torch.Tensor,
    ) -> None:
        """
        Consolida las estadísticas de las partidas que acaban de terminar.

        Args:
            termina_ahora: (N,) bool, True para partidas que finalizan en este turno.
            winner: (N,) int, 0=P1, 1=P2, 2=Empate.
            p1_deaths, p2_deaths: (N,) número de muertes acumuladas.
            turn_number: (N,) turno actual.
            por_muerte_mask: (N,) bool, True si la partida terminó por muerte de equipo.
            por_turnos_mask: (N,) bool, True si la partida terminó por límite de turnos.
        """
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