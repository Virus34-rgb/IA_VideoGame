"""
Formateo de un StatsAccumulator a informe de texto. Extraído de StatsV.
No posee ningún estado propio de acumulación -- recibe el StatsAccumulator ya
poblado como parámetro en cada llamada, nunca como atributo de constructor
(así no hay riesgo de que quede "desactualizado" respecto al acumulador real).
"""
from typing import Dict, List, Any
from AI.Environment.statsV import StatsSummary
from AI.Logging.stats_accumulator import StatsAccumulator
import constants  # o donde quede definido StatsSummary


class StatsReportWriter:
    @staticmethod
    def write(path: str, stats: "StatsAccumulator", warriors_classes: Dict[int, Any],
               p1_elo: float = 0.0, p2_elo: float = 0.0, pool_elos: Dict[int, float] | None = None) -> None:
        summary = StatsReportWriter._build_summary(stats)
        p1_warrior_use = stats._p1_warrior_use_tensor.tolist()
        p2_warrior_use = stats._p2_warrior_use_tensor.tolist()
        p1_attacks = {i: stats._p1_attacks_tensor[i].tolist() for i in range(1, __import__("constants").WARRIOR_QUANTITY + 1)}
        p2_attacks = {i: stats._p2_attacks_tensor[i].tolist() for i in range(1, __import__("constants").WARRIOR_QUANTITY + 1)}

        sections = [
            StatsReportWriter._section_resultados(stats, summary),
            StatsReportWriter._section_rusher(stats),
            StatsReportWriter._section_elo(p1_elo, p2_elo, pool_elos or {}),
            StatsReportWriter._section_recompensa(stats, summary),
            StatsReportWriter._section_dano(stats, summary),
            StatsReportWriter._section_healing(stats, summary),
            StatsReportWriter._section_bajas(stats, summary),
            StatsReportWriter._section_movimientos(stats),
            StatsReportWriter._section_bloqueos(stats, summary),
            StatsReportWriter._section_seleccion(p1_warrior_use, p2_warrior_use),
            StatsReportWriter._section_ataques(p1_attacks, p2_attacks, warriors_classes),
        ]
        header = "=" * 65 + "\n                    ESTADÍSTICAS IA\n" + "=" * 65 + "\n"
        body = "\n\n".join(s for s in sections if s)
        footer = "\n" + "=" * 65 + "\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + "\n" + body + footer)
            
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
        p1_attacks = {i: self._p1_attacks_tensor[i].tolist() for i in range(1, constants.WARRIOR_QUANTITY + 1)}
        p2_attacks = {i: self._p2_attacks_tensor[i].tolist() for i in range(1, constants.WARRIOR_QUANTITY + 1)}

        sections = [
            self._section_resultados(summary),
            self._section_rusher() ,
            self._section_elo(p1_elo, p2_elo, pool_elos or {}),
            self._section_recompensa(summary),
            
            self._section_dano(summary),
            self._section_healing(summary),
            self._section_bajas(summary),
            self._section_movimientos(),
            self._section_bloqueos(summary),
            self._section_seleccion(p1_warrior_use, p2_warrior_use),
            self._section_ataques(p1_attacks, p2_attacks, warriors_classes),
        ]

        header = "=" * 65 + "\n                    ESTADÍSTICAS IA\n" + "=" * 65 + "\n"
        body = "\n\n".join(s for s in sections if s)
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
    
    @staticmethod
    def _section_resultados(,stats, s: StatsSummary) -> str:
        d = s.to_dict()
        return stats._section("RESULTADOS", [
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

    def _section_recompensa(stats, s: StatsSummary) -> str:
        d = s.to_dict()
        return stats._section("RECOMPENSA ACUMULADA", [
            f"Reward total P1:           {s.total_reward_p1:.2f}",
            f"Reward total P2:           {s.total_reward_p2:.2f}",
            f"Reward media P1:           {d['p1_reward_avg']:.2f}/partida",
            f"Reward media P2:           {d['p2_reward_avg']:.2f}/partida",
        ])

    def _section_seleccion(self,stats, p1_use: List[int], p2_use: List[int]) -> str:
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

    def _section_dano(stats, s: StatsSummary) -> str:
        d = s.to_dict()
        return stats._section("DAÑO", [
            f"Daño total P1:             {s.p1_damage:.2f}",
            f"Daño total P2:             {s.p2_damage:.2f}",
            f"Daño medio P1:             {d['p1_damage_avg']:.2f}",
            f"Daño medio P2:             {d['p2_damage_avg']:.2f}",
            f"Daño por sobrekill P1:     {stats.p1_overkill_damage:.2f}",
            f"Daño por sobrekill P2:     {stats.p2_overkill_damage:.2f}",
            f"Daño por sobrekill medio P1:{stats.p1_overkill_damage / stats.partidas:.2f}",
            f"Daño por sobrekill medio P2:{stats.p2_overkill_damage / stats.partidas:.2f}",
            f"Kill confirmed P1:         {stats.p1_kill_confirmed:.2f}",
            f"Kill confirmed P2:         {stats.p2_kill_confirmed:.2f}",
            f"Kill confirmed medio P1:   {stats.p1_kill_confirmed / stats.partidas:.2f}",
            f"Kill confirmed medio P2:   {stats.p2_kill_confirmed / stats.partidas:.2f}",
        ])

    def _section_healing(stats, s: StatsSummary) -> str:
        d = s.to_dict()
        return stats._section("Healing", [
            f"Heal total P1:             {s.p1_tot_heal:.2f}",
            f"Heal total P2:             {s.p2_tot_heal:.2f}",
            f"Heal medio P1:             {d['p1_tot_heal_avg']:.2f}",
            f"Heal medio P2:             {d['p2_tot_heal_avg']:.2f}",
            f"Cura desperdiciada media P1:  {stats.wasted_heal_p1 / stats.partidas:.2f}",
            f"Cura desperdiciada media P2:  {stats.wasted_heal_p2 / stats.partidas:.2f}",
        ])

    def _section_bajas(stats, s: StatsSummary) -> str:
        d = s.to_dict()
        return stats._section("BAJAS (PROPIAS)", [
            f"Bajas totales P1:          {s.p1_total_deaths}",
            f"Bajas totales P2:          {s.p2_total_deaths}",
            f"Bajas medias P1:           {d['p1_deaths_avg']:.2f}",
            f"Bajas medias P2:           {d['p2_deaths_avg']:.2f}",
        ])

    def _section_ataques(
        self,
        stats,
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
        return stats._section("ATAQUES (habilidad seleccionada)", lines)

    def _section_movimientos(stats) -> str:
        return stats._section("MOVIMIENTOS", [
            f"Movimientos P1:            {int(stats.p1_movements)}",
            f"Movimientos P2:            {int(stats.p2_movements)}",
            f"Movimientos estratégicos P1:{int(stats.p1_strategic_movement)}",
            f"Movimientos estratégicos P2:{int(stats.p2_strategic_movement)}",
            f"Movimientos estratégicos P1 (%):{stats.p1_strategic_movement / max(stats.p1_movements, 1) * 100:.2f}%",
            f"Movimientos estratégicos P2 (%):{stats.p2_strategic_movement / max(stats.p2_movements, 1) * 100:.2f}%",
        ])

    def _section_bloqueos(stats, s: StatsSummary) -> str:
        d = s.to_dict()
        return stats._section("BLOQUEOS Y DAÑO EVITADO", [
            f"Daño evitado P1:           {s.p1_tot_damage_evaded:.2f} -> {d['p1_damage_evaded_avg']:.2f}/partida",
            f"Daño evitado P2:           {s.p2_tot_damage_evaded:.2f} -> {d['p2_damage_evaded_avg']:.2f}/partida",
            f"Defensas desperdiciadas P1:{stats.wasted_defense_p1 / stats.partidas:.2f}",
            f"Defensas desperdiciadas P2:{stats.wasted_defense_p2 / stats.partidas:.2f}",
        ])
        
    def _section_elo(stats, p1_elo: float, p2_elo: float, pool_elos: Dict[int, float]) -> str:
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
        return stats._section("ELO (MATCHMAKING)", lines)
    
    def _section_rusher(stats,stats):
        if(stats.partidas_vs_rusher > 0):
                    return stats._section("RESULTADOS VS RUSHER", [
            f"PARTIDAS VS RUSHER:           {stats.partidas_vs_rusher}",
            f"Victorias IA:           {stats.p1_victories_vs_rusher} -> {stats.p1_victories_vs_rusher / stats.partidas_vs_rusher}",
            f"Victorias Rusher:           {stats.p2_victories_vs_rusher} -> {stats.p2_victories_vs_rusher / stats.partidas_vs_rusher}",
            f"Empates:           {stats.empates_vs_rusher} -> {stats.empates_vs_rusher / stats.partidas_vs_rusher}",
        ])
        return ""


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
        names = {1: "Knight", 2: "Archer", 3: "Rogue", 4: "Wizard", 5: "Cleric"}
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