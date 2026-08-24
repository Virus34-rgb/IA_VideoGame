from constants import WARRIOR_QUANTITY


class Stats:
    def __init__(self):
        self.p1_total_deaths = 0
        self.p2_total_deaths = 0
        self.partidas = 0
        self.partidas_por_muerte = 0
        self.partidas_por_limite_turnos = 0
        self.p1_victories = 0
        self.p2_victories = 0
        self.empates = 0
        self.p1_damage = 0
        self.p2_damage = 0
        self.p1_movements = 0
        self.p2_movements = 0
        self.p1_attacks = {i: [0, 0, 0, 0] for i in range(1, WARRIOR_QUANTITY + 1)}
        self.p2_attacks = {i: [0, 0, 0, 0] for i in range(1, WARRIOR_QUANTITY + 1)}
        self.p1_succes_blocks = 0
        self.p2_succes_blocks = 0
        self.p1_tot_damage_evaded = 0
        self.p2_tot_damage_evaded = 0
        self.p1_tot_heal = 0
        self.p2_tot_heal = 0
        self.p1_warrior_use = [0] * WARRIOR_QUANTITY
        self.p2_warrior_use = [0] * WARRIOR_QUANTITY
        self.total_turns = 0
        self.total_reward_p1 = 0
        self.total_reward_p2 = 0

    def reset(self):
        self.__init__()

    # ------------------------------------------------------------
    # Cálculos derivados (no se guardan, se calculan al exportar)
    # ------------------------------------------------------------
    def _summary(self):
        partidas = max(self.partidas, 1)
        decisive_games = self.p1_victories + self.p2_victories

        return {
            "partidas": self.partidas,
            "p1_winrate": self.p1_victories / partidas * 100,
            "p2_winrate": self.p2_victories / partidas * 100,
            "drawrate": self.empates / partidas * 100,
            "p1_win_ratio_excl_draws": (self.p1_victories / decisive_games * 100) if decisive_games else 0,
            "p2_win_ratio_excl_draws": (self.p2_victories / decisive_games * 100) if decisive_games else 0,
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
            "p1_tot_heal_avg":self.p1_tot_heal / partidas,
            "p2_tot_heal_avg":self.p2_tot_heal / partidas
        }

    def _warrior_selection_lines(self, warrior_use):
        total = sum(warrior_use)
        lines = []
        enemigos = {1:"Knight",2:"Archer",3:"Rogue",4:"Wizard",5:"Cleric"}
        for i, uses in enumerate(warrior_use):
            pct = uses / total * 100 if total > 0 else 0
            avg = uses / max(self.partidas, 1)
            lines.append(f"{enemigos[i+1]}:          {uses:4d} ({pct:6.2f}%) -> {avg:.2f}/partida")
        return lines

    def _ability_usage_lines(self, attacks, warriors_classes):
        total = sum(sum(a) for a in attacks.values())
        lines = []
        for warrior_id, counts in attacks.items():
            warrior = warriors_classes[warrior_id]
            lines.append(f"  Guerrero {warrior_id}:")
            for ability_idx, count in enumerate(counts):
                ability_name = warrior.abilities[ability_idx].name
                pct = count / total * 100 if total > 0 else 0
                avg = count / max(self.partidas, 1)
                lines.append(f"    {ability_name:15s} {count:4d} ({pct:6.2f}%) -> {avg:.2f}/partida")
        return lines

    # ------------------------------------------------------------
    # Exportación
    # ------------------------------------------------------------
    def guardar_stats(self, path, warriors_classes):
        s = self._summary()

        sections = []

        sections.append(self._section("RESULTADOS", [
            f"Partidas:                  {s['partidas']}",
            f"Victorias P1:              {self.p1_victories} ({s['p1_winrate']:.2f}%)",
            f"Victorias P2:              {self.p2_victories} ({s['p2_winrate']:.2f}%)",
            f"Empates:                   {self.empates} ({s['drawrate']:.2f}%)",
            f"Win ratio P1 (sin empates):{s['p1_win_ratio_excl_draws']:.2f}%",
            f"Win ratio P2 (sin empates):{s['p2_win_ratio_excl_draws']:.2f}%",
            f"Terminadas por muerte:     {self.partidas_por_muerte} ({s['partidas_por_muerte_pct']:.2f}%)",
            f"Terminadas por límite:     {self.partidas_por_limite_turnos} ({s['partidas_por_limite_turnos_pct']:.2f}%)",
            f"Turnos totales:            {self.total_turns}",
            f"Turnos medios por partida: {s['avg_turns']:.2f}",
        ]))

        sections.append(self._section("RECOMPENSA ACUMULADA", [
            f"Reward total P1:           {self.total_reward_p1:.2f}",
            f"Reward total P2:           {self.total_reward_p2:.2f}",
            f"Reward media P1:           {s['p1_reward_avg']:.2f}/partida",
            f"Reward media P2:           {s['p2_reward_avg']:.2f}/partida",
        ]))

        sections.append(self._section("SELECCIÓN DE GUERREROS", [
            f"Selecciones totales P1:    {sum(self.p1_warrior_use)}",
            f"Selecciones totales P2:    {sum(self.p2_warrior_use)}",
            "",
            "P1:",
            *self._warrior_selection_lines(self.p1_warrior_use),
            "",
            "P2:",
            *self._warrior_selection_lines(self.p2_warrior_use),
        ]))

        sections.append(self._section("DAÑO", [
            f"Daño total P1:             {self.p1_damage}",
            f"Daño total P2:             {self.p2_damage}",
            f"Daño medio P1:             {s['p1_damage_avg']:.2f}",
            f"Daño medio P2:             {s['p2_damage_avg']:.2f}",
        ]))
        
        sections.append(self._section("Healing", [
            f"Daño total P1:             {self.p1_tot_heal}",
            f"Daño total P2:             {self.p2_tot_heal}",
            f"Daño medio P1:             {s['p1_tot_heal_avg']:.2f}",
            f"Daño medio P2:             {s['p2_tot_heal_avg']:.2f}",
        ]))

        sections.append(self._section("BAJAS (PROPIAS)", [
            f"Bajas totales P1:          {self.p1_total_deaths}",
            f"Bajas totales P2:          {self.p2_total_deaths}",
            f"Bajas medias P1:           {s['p1_deaths_avg']:.2f}",
            f"Bajas medias P2:           {s['p2_deaths_avg']:.2f}",
        ]))

        sections.append(self._section("ATAQUES (habilidad seleccionada)", [
            f"Ataques totales P1:        {sum(sum(a) for a in self.p1_attacks.values())}",
            f"Ataques totales P2:        {sum(sum(a) for a in self.p2_attacks.values())}",
            "",
            "P1 - USO DE HABILIDADES POR GUERRERO:",
            *self._ability_usage_lines(self.p1_attacks, warriors_classes),
            "",
            "P2 - USO DE HABILIDADES POR GUERRERO:",
            *self._ability_usage_lines(self.p2_attacks, warriors_classes),
        ]))

        sections.append(self._section("MOVIMIENTOS", [
            f"Movimientos P1:            {self.p1_movements}",
            f"Movimientos P2:            {self.p2_movements}",
        ]))

        sections.append(self._section("BLOQUEOS Y DAÑO EVITADO", [
            f"Bloqueos exitosos P1:      {self.p1_succes_blocks} -> {s['p1_success_blocks_avg']:.2f}/partida",
            f"Bloqueos exitosos P2:      {self.p2_succes_blocks} -> {s['p2_success_blocks_avg']:.2f}/partida",
            f"Daño evitado P1:           {self.p1_tot_damage_evaded} -> {s['p1_damage_evaded_avg']:.2f}/partida",
            f"Daño evitado P2:           {self.p2_tot_damage_evaded} -> {s['p2_damage_evaded_avg']:.2f}/partida",
        ]))

        header = "=" * 65 + "\n                    ESTADÍSTICAS IA\n" + "=" * 65 + "\n"
        body = "\n\n".join(sections)
        footer = "\n" + "=" * 65 + "\n"

        with open(path, "w", encoding="utf-8") as file:
            file.write(header + "\n" + body + footer)

    @staticmethod
    def _section(title, lines):
        return title + "\n" + "-" * 65 + "\n" + "\n".join(lines)