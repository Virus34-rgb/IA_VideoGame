"""
Módulo de estadísticas para el entorno vectorizado.

Acumula métricas de batalla (daño, cura, bloqueos, movimientos, selección de héroes, etc.)
y genera informes en texto plano.
"""
import torch
from dataclasses import dataclass
from typing import Dict, List, Any

from constants import MAX_POOL_SIZE, WARRIOR_QUANTITY
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
        self.p1_tot_heal: float = 0.0
        self.p2_tot_heal: float = 0.0
        self.p1_total_deaths: int = 0
        self.p2_total_deaths: int = 0
        self.total_turns: int = 0
        self.total_reward_p1: float = 0.0
        self.total_reward_p2: float = 0.0

        self.p1_movements: float = 0.0
        self.p2_movements: float = 0.0

        # Tensores acumuladores por tipo de guerrero y habilidad
        self._p1_attacks_tensor = torch.zeros(WARRIOR_QUANTITY + 1, MAX_POOL_SIZE)
        self._p2_attacks_tensor = torch.zeros(WARRIOR_QUANTITY + 1, MAX_POOL_SIZE)
        self._p1_warrior_use_tensor = torch.zeros(WARRIOR_QUANTITY)
        self._p2_warrior_use_tensor = torch.zeros(WARRIOR_QUANTITY)

        # Buffers por batch (se reinician en start_batch)
        self._p1_damage_batch: torch.Tensor | None = None
        self._p2_damage_batch: torch.Tensor | None = None
        self._p1_blocks_batch: torch.Tensor | None = None
        self._p2_blocks_batch: torch.Tensor | None = None
        self._p1_evaded_batch: torch.Tensor | None = None
        self._p2_evaded_batch: torch.Tensor | None = None
        self._p1_heal_batch: torch.Tensor | None = None
        self._p2_heal_batch: torch.Tensor | None = None

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
            counts = torch.bincount(idx, minlength=(WARRIOR_QUANTITY + 1) * MAX_POOL_SIZE)
            self._p1_attacks_tensor += counts.view(WARRIOR_QUANTITY + 1, MAX_POOL_SIZE).float()

        if mask_p2.any():
            idx = tipo_actor[mask_p2] * constants.MAX_POOL_SIZE + accion_actor[mask_p2]
            counts = torch.bincount(idx, minlength=(WARRIOR_QUANTITY + 1) * MAX_POOL_SIZE)
            self._p2_attacks_tensor += counts.view(WARRIOR_QUANTITY + 1, MAX_POOL_SIZE).float()

    def accumulate_warrior_use(self, warrior1: torch.Tensor, warrior2: torch.Tensor) -> None:
        """
        Acumula la selección de guerreros al inicio de la partida.

        Args:
            warrior1, warrior2: (N,) IDs de los guerreros seleccionados por P1 y P2.
        """
        c1 = torch.bincount(warrior1, minlength=WARRIOR_QUANTITY + 1)[1:]
        c2 = torch.bincount(warrior2, minlength=WARRIOR_QUANTITY + 1)[1:]
        self._p1_warrior_use_tensor += c1.float()
        self._p2_warrior_use_tensor += c2.float()

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

    # ------------------------------------------------------------
    # Generación de informes
    # ------------------------------------------------------------

    def guardar_stats(
        self, 
        path: str, 
        warriors_classes: Dict[int, Any],
        p1_elo: float = 0.0,
        p2_elo: float = 0.0,
        pool_elos: Dict[int, float] | None = None,
        ) -> None:
        """
        Guarda un informe de estadísticas en un archivo de texto.

        Args:
            path: Ruta del archivo de salida.
            warriors_classes: Diccionario {id: WarriorData} con los datos de los héroes.
        """
        summary = self._build_summary()
        p1_warrior_use = self._p1_warrior_use_tensor.tolist()
        p2_warrior_use = self._p2_warrior_use_tensor.tolist()
        p1_attacks = {i: self._p1_attacks_tensor[i].tolist() for i in range(1, WARRIOR_QUANTITY + 1)}
        p2_attacks = {i: self._p2_attacks_tensor[i].tolist() for i in range(1, WARRIOR_QUANTITY + 1)}

        sections = [
            self._section_resultados(summary),
            self._section_elo(p1_elo, p2_elo, pool_elos or {}),
            self._section_recompensa(summary),
            self._section_seleccion(p1_warrior_use, p2_warrior_use),
            self._section_dano(summary),
            self._section_healing(summary),
            self._section_bajas(summary),
            self._section_ataques(p1_attacks, p2_attacks, warriors_classes),
            self._section_movimientos(),
            self._section_bloqueos(summary),
        ]

        header = "=" * 65 + "\n                    ESTADÍSTICAS IA\n" + "=" * 65 + "\n"
        body = "\n\n".join(sections)
        footer = "\n" + "=" * 65 + "\n"

        with open(path, "w", encoding="utf-8") as f:
            f.write(header + "\n" + body + footer)

    def _build_summary(self) -> StatsSummary:
        """Construye el objeto de resumen a partir de los acumuladores."""
        return StatsSummary(
            partidas=self.partidas,
            p1_victories=self.p1_victories,
            p2_victories=self.p2_victories,
            empates=self.empates,
            partidas_por_muerte=self.partidas_por_muerte,
            partidas_por_limite_turnos=self.partidas_por_limite_turnos,
            total_turns=self.total_turns,
            p1_damage=self.p1_damage,
            p2_damage=self.p2_damage,
            p1_succes_blocks=self.p1_succes_blocks,
            p2_succes_blocks=self.p2_succes_blocks,
            p1_tot_damage_evaded=self.p1_tot_damage_evaded,
            p2_tot_damage_evaded=self.p2_tot_damage_evaded,
            p1_tot_heal=self.p1_tot_heal,
            p2_tot_heal=self.p2_tot_heal,
            p1_total_deaths=self.p1_total_deaths,
            p2_total_deaths=self.p2_total_deaths,
            total_reward_p1=self.total_reward_p1,
            total_reward_p2=self.total_reward_p2,
            p1_movements=self.p1_movements,
            p2_movements=self.p2_movements,
        )

    # ------------------------------------------------------------
    # Secciones del informe (métodos privados de formateo)
    # ------------------------------------------------------------

    @staticmethod
    def _section(title: str, lines: List[str]) -> str:
        """Genera una sección formateada con título y líneas."""
        return title + "\n" + "-" * 65 + "\n" + "\n".join(lines)

    def _section_resultados(self, s: StatsSummary) -> str:
        d = s.to_dict()
        return self._section("RESULTADOS", [
            f"Partidas:                  {s.partidas}",
            f"Victorias P1:              {s.p1_victories} ({d['p1_winrate']:.2f}%)",
            f"Victorias P2:              {s.p2_victories} ({d['p2_winrate']:.2f}%)",
            f"Empates:                   {s.empates} ({d['drawrate']:.2f}%)",
            f"Win ratio P1 (sin empates):{d['p1_win_ratio_excl_draws']:.2f}%",
            f"Win ratio P2 (sin empates):{d['p2_win_ratio_excl_draws']:.2f}%",
            f"Terminadas por muerte:     {s.partidas_por_muerte} ({d['partidas_por_muerte_pct']:.2f}%)",
            f"Terminadas por límite:     {s.partidas_por_limite_turnos} ({d['partidas_por_limite_turnos_pct']:.2f}%)",
            f"Turnos totales:            {s.total_turns}",
            f"Turnos medios por partida: {d['avg_turns']:.2f}",
        ])

    def _section_recompensa(self, s: StatsSummary) -> str:
        d = s.to_dict()
        return self._section("RECOMPENSA ACUMULADA", [
            f"Reward total P1:           {s.total_reward_p1:.2f}",
            f"Reward total P2:           {s.total_reward_p2:.2f}",
            f"Reward media P1:           {d['p1_reward_avg']:.2f}/partida",
            f"Reward media P2:           {d['p2_reward_avg']:.2f}/partida",
        ])

    def _section_seleccion(self, p1_use: List[int], p2_use: List[int]) -> str:
        lines = [
            f"Selecciones totales P1:    {sum(p1_use)}",
            f"Selecciones totales P2:    {sum(p2_use)}",
            "",
            "P1:",
            *self._warrior_selection_lines(p1_use),
            "",
            "P2:",
            *self._warrior_selection_lines(p2_use),
        ]
        return self._section("SELECCIÓN DE GUERREROS", lines)

    def _section_dano(self, s: StatsSummary) -> str:
        d = s.to_dict()
        return self._section("DAÑO", [
            f"Daño total P1:             {s.p1_damage:.2f}",
            f"Daño total P2:             {s.p2_damage:.2f}",
            f"Daño medio P1:             {d['p1_damage_avg']:.2f}",
            f"Daño medio P2:             {d['p2_damage_avg']:.2f}",
        ])

    def _section_healing(self, s: StatsSummary) -> str:
        d = s.to_dict()
        return self._section("Healing", [
            f"Heal total P1:             {s.p1_tot_heal:.2f}",
            f"Heal total P2:             {s.p2_tot_heal:.2f}",
            f"Heal medio P1:             {d['p1_tot_heal_avg']:.2f}",
            f"Heal medio P2:             {d['p2_tot_heal_avg']:.2f}",
        ])

    def _section_bajas(self, s: StatsSummary) -> str:
        d = s.to_dict()
        return self._section("BAJAS (PROPIAS)", [
            f"Bajas totales P1:          {s.p1_total_deaths}",
            f"Bajas totales P2:          {s.p2_total_deaths}",
            f"Bajas medias P1:           {d['p1_deaths_avg']:.2f}",
            f"Bajas medias P2:           {d['p2_deaths_avg']:.2f}",
        ])

    def _section_ataques(
        self,
        p1_attacks: Dict[int, List[int]],
        p2_attacks: Dict[int, List[int]],
        warriors_classes: Dict[int, Any],
    ) -> str:
        total_p1 = sum(sum(a) for a in p1_attacks.values())
        total_p2 = sum(sum(a) for a in p2_attacks.values())
        lines = [
            f"Ataques totales P1:        {total_p1}",
            f"Ataques totales P2:        {total_p2}",
            "",
            "P1 - USO DE HABILIDADES POR GUERRERO:",
            *self._ability_usage_lines(p1_attacks, warriors_classes),
            "",
            "P2 - USO DE HABILIDADES POR GUERRERO:",
            *self._ability_usage_lines(p2_attacks, warriors_classes),
        ]
        return self._section("ATAQUES (habilidad seleccionada)", lines)

    def _section_movimientos(self) -> str:
        return self._section("MOVIMIENTOS", [
            f"Movimientos P1:            {int(self.p1_movements)}",
            f"Movimientos P2:            {int(self.p2_movements)}",
        ])

    def _section_bloqueos(self, s: StatsSummary) -> str:
        d = s.to_dict()
        return self._section("BLOQUEOS Y DAÑO EVITADO", [
            f"Bloqueos exitosos P1:      {s.p1_succes_blocks:.2f} -> {d['p1_success_blocks_avg']:.2f}/partida",
            f"Bloqueos exitosos P2:      {s.p2_succes_blocks:.2f} -> {d['p2_success_blocks_avg']:.2f}/partida",
            f"Daño evitado P1:           {s.p1_tot_damage_evaded:.2f} -> {d['p1_damage_evaded_avg']:.2f}/partida",
            f"Daño evitado P2:           {s.p2_tot_damage_evaded:.2f} -> {d['p2_damage_evaded_avg']:.2f}/partida",
        ])
        
    def _section_elo(self, p1_elo: float, p2_elo: float, pool_elos: Dict[int, float]) -> str:
        """Genera la sección de ratings Elo (P1, P2 y snapshots de la pool)."""
        lines = [
            f"Elo P1:                    {p1_elo:.1f}",
            f"Elo P2:                    {p2_elo:.1f}",
            "",
        ]
        if pool_elos:
            lines.append(f"Snapshots en la pool:      {len(pool_elos)}")
            lines.append("")
            # Ordenados de mayor a menor Elo para lectura rápida
            for cp_id, elo in sorted(pool_elos.items(), key=lambda kv: kv[1], reverse=True):
                lines.append(f"  Checkpoint {cp_id:>4d}:        {elo:.1f}")
        else:
            lines.append("Pool vacía (sin snapshots aún).")
        return self._section("ELO (MATCHMAKING)", lines)

    # ------------------------------------------------------------
    # Utilidades de formateo (estáticas)
    # ------------------------------------------------------------

    @staticmethod
    def _warrior_selection_lines(warrior_use: List[int]) -> List[str]:
        """Genera líneas de texto para la frecuencia de selección de guerreros."""
        total = sum(warrior_use)
        names = {1: "Knight", 2: "Archer", 3: "Rogue", 4: "Wizard", 5: "Cleric"}
        lines = []
        for i, uses in enumerate(warrior_use):
            pct = uses / total * 100 if total > 0 else 0.0
            lines.append(f"{names[i+1]}:          {int(uses):4d} ({pct:6.2f}%)")
        return lines

    @staticmethod
    def _ability_usage_lines(
        attacks: Dict[int, List[int]],
        warriors_classes: Dict[int, Any],
    ) -> List[str]:
        """Genera líneas de texto para el uso de habilidades por guerrero.
        Muestra el porcentaje de uso dentro del guerrero y el porcentaje global.
        """
        total_global = sum(sum(a) for a in attacks.values())
        lines = []
        for warrior_id, counts in attacks.items():
            warrior = warriors_classes[warrior_id]
            warrior_total = sum(counts)
            # Línea de encabezado del guerrero con total y porcentaje global
            pct_global_warrior = (warrior_total / total_global * 100) if total_global > 0 else 0.0
            lines.append(f"  Guerrero {warrior_id} (total: {warrior_total}, {pct_global_warrior:6.2f}% global):")
            for ability_idx, count in enumerate(counts):
                ability_name = warrior.ability_pool[ability_idx].name
                pct_warrior = (count / warrior_total * 100) if warrior_total > 0 else 0.0
                pct_global = (count / total_global * 100) if total_global > 0 else 0.0
                lines.append(f"    {ability_name:15s} {int(count):4d} ({pct_warrior:6.2f}% del guerrero, {pct_global:6.2f}% global)")
        return lines