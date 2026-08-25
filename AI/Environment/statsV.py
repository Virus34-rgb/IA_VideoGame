import torch
from constants import WARRIOR_QUANTITY


class StatsV:
    def __init__(self):
        self.reset()

    def reset(self):
        self.partidas = 0
        self.p1_victories = 0
        self.p2_victories = 0
        self.empates = 0
        self.partidas_por_muerte = 0
        self.partidas_por_limite_turnos = 0

        self.p1_damage = 0.0
        self.p2_damage = 0.0
        self.p1_succes_blocks = 0.0
        self.p2_succes_blocks = 0.0
        self.p1_tot_damage_evaded = 0.0
        self.p2_tot_damage_evaded = 0.0
        self.p1_tot_heal = 0.0
        self.p2_tot_heal = 0.0
        self.p1_total_deaths = 0
        self.p2_total_deaths = 0
        self.total_turns = 0
        self.total_reward_p1 = 0.0
        self.total_reward_p2 = 0.0

        # NUEVO: recuperados del Stats original. Se acumulan DIRECTAMENTE
        # cada turno (no requieren esperar al cierre de partida como
        # damage/heal/blocks) porque son totales simples — el enmascarado
        # por "activa" (partidas no terminadas antes de este turno) ya evita
        # el doble conteo, así que no hace falta el patrón batch+close.
        self.p1_movements = 0.0
        self.p2_movements = 0.0
        # Tensores acumuladores (num_types+1, 4) — fila 0 sin usar (id=0 = vacío)
        self._p1_attacks_tensor = torch.zeros(WARRIOR_QUANTITY + 1, 4)
        self._p2_attacks_tensor = torch.zeros(WARRIOR_QUANTITY + 1, 4)
        self._p1_warrior_use_tensor = torch.zeros(WARRIOR_QUANTITY)
        self._p2_warrior_use_tensor = torch.zeros(WARRIOR_QUANTITY)

        self._p1_damage_batch = None
        self._p2_damage_batch = None
        self._p1_blocks_batch = None
        self._p2_blocks_batch = None
        self._p1_evaded_batch = None
        self._p2_evaded_batch = None
        self._p1_heal_batch = None
        self._p2_heal_batch = None

    def start_batch(self, N):
        self._p1_damage_batch = torch.zeros(N)
        self._p2_damage_batch = torch.zeros(N)
        self._p1_blocks_batch = torch.zeros(N)
        self._p2_blocks_batch = torch.zeros(N)
        self._p1_evaded_batch = torch.zeros(N)
        self._p2_evaded_batch = torch.zeros(N)
        self._p1_heal_batch = torch.zeros(N)
        self._p2_heal_batch = torch.zeros(N)

    def accumulate_turn(self, damage_p1, damage_p2, blocks_p1, blocks_p2,
                         avoided_p1, avoided_p2, heal_p1, heal_p2, ya_terminadas_antes):
        activa = ~ya_terminadas_antes
        self._p1_damage_batch += torch.where(activa, damage_p1, torch.zeros_like(damage_p1))
        self._p2_damage_batch += torch.where(activa, damage_p2, torch.zeros_like(damage_p2))
        self._p1_blocks_batch += torch.where(activa, blocks_p1, torch.zeros_like(blocks_p1))
        self._p2_blocks_batch += torch.where(activa, blocks_p2, torch.zeros_like(blocks_p2))
        self._p1_evaded_batch += torch.where(activa, avoided_p1, torch.zeros_like(avoided_p1))
        self._p2_evaded_batch += torch.where(activa, avoided_p2, torch.zeros_like(avoided_p2))
        self._p1_heal_batch += torch.where(activa, heal_p1, torch.zeros_like(heal_p1))
        self._p2_heal_batch += torch.where(activa, heal_p2, torch.zeros_like(heal_p2))

    def accumulate_movements(self, moved, es_p1, activa):
        """
        NUEVO. moved: (N,) float 0/1 para el actor de esta posición del bucle
        de turno. es_p1: (N,) bool, True donde el actor de esta posición es P1.
        activa: (N,) bool, partidas que no habían terminado antes de este turno.
        """
        mask = activa.float()
        self.p1_movements += (torch.where(es_p1, moved, torch.zeros_like(moved)) * mask).sum().item()
        self.p2_movements += (torch.where(es_p1, torch.zeros_like(moved), moved) * mask).sum().item()

    def accumulate_attacks(self, tipo_actor, accion_actor, es_p1, activa):
        """
        NUEVO. Replica `if action in (1,2,3,4): attack_stats[id][action-1] += 1`
        del original — aquí accion_actor ya está en base 0 (0-3 = habilidad,
        5/6 = movimiento, quedan excluidos automáticamente del rango 0-3).
        Vectorizado con bincount, sin bucle Python por evento.
        """
        es_habilidad = (accion_actor >= 0) & (accion_actor <= 3)
        mask = es_habilidad & activa
        mask_p1 = mask & es_p1
        mask_p2 = mask & ~es_p1

        if mask_p1.any():
            idx = tipo_actor[mask_p1] * 4 + accion_actor[mask_p1]
            counts = torch.bincount(idx, minlength=(WARRIOR_QUANTITY + 1) * 4)
            self._p1_attacks_tensor += counts.view(WARRIOR_QUANTITY + 1, 4).float()
        if mask_p2.any():
            idx = tipo_actor[mask_p2] * 4 + accion_actor[mask_p2]
            counts = torch.bincount(idx, minlength=(WARRIOR_QUANTITY + 1) * 4)
            self._p2_attacks_tensor += counts.view(WARRIOR_QUANTITY + 1, 4).float()

    def accumulate_warrior_use(self, warrior1, warrior2):
        """NUEVO. Llamar desde Environment.warrior_selected. warrior1/warrior2: (N,) long."""
        c1 = torch.bincount(warrior1, minlength=WARRIOR_QUANTITY + 1)[1:]
        c2 = torch.bincount(warrior2, minlength=WARRIOR_QUANTITY + 1)[1:]
        self._p1_warrior_use_tensor += c1.float()
        self._p2_warrior_use_tensor += c2.float()

    def close_finished_games(self, termina_ahora, winner, p1_deaths, p2_deaths, turn_number,
                              por_muerte_mask, por_turnos_mask):
        n_cerradas = termina_ahora.sum().item()
        if n_cerradas == 0:
            return

        self.partidas += n_cerradas
        self.p1_victories += (termina_ahora & (winner == 0)).sum().item()
        self.p2_victories += (termina_ahora & (winner == 1)).sum().item()
        self.empates += (termina_ahora & (winner == 2)).sum().item()
        self.partidas_por_muerte += (termina_ahora & por_muerte_mask).sum().item()
        self.partidas_por_limite_turnos += (termina_ahora & por_turnos_mask).sum().item()

        idx = termina_ahora.nonzero(as_tuple=True)[0]
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
    # Cálculos derivados y exportación — RESTAURADO del Stats original
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
            "p1_tot_heal_avg": self.p1_tot_heal / partidas,
            "p2_tot_heal_avg": self.p2_tot_heal / partidas,
        }

    def _warrior_selection_lines(self, warrior_use):
        total = sum(warrior_use)
        lines = []
        enemigos = {1: "Knight", 2: "Archer", 3: "Rogue", 4: "Wizard", 5: "Cleric"}
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

    def guardar_stats(self, path, warriors_classes):
        s = self._summary()
        p1_warrior_use = [int(x) for x in self._p1_warrior_use_tensor.tolist()]
        p2_warrior_use = [int(x) for x in self._p2_warrior_use_tensor.tolist()]
        p1_attacks = {i: [int(x) for x in self._p1_attacks_tensor[i].tolist()] for i in range(1, WARRIOR_QUANTITY + 1)}
        p2_attacks = {i: [int(x) for x in self._p2_attacks_tensor[i].tolist()] for i in range(1, WARRIOR_QUANTITY + 1)}

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
            f"Selecciones totales P1:    {sum(p1_warrior_use)}",
            f"Selecciones totales P2:    {sum(p2_warrior_use)}",
            "",
            "P1:",
            *self._warrior_selection_lines(p1_warrior_use),
            "",
            "P2:",
            *self._warrior_selection_lines(p2_warrior_use),
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
            f"Ataques totales P1:        {sum(sum(a) for a in p1_attacks.values())}",
            f"Ataques totales P2:        {sum(sum(a) for a in p2_attacks.values())}",
            "",
            "P1 - USO DE HABILIDADES POR GUERRERO:",
            *self._ability_usage_lines(p1_attacks, warriors_classes),
            "",
            "P2 - USO DE HABILIDADES POR GUERRERO:",
            *self._ability_usage_lines(p2_attacks, warriors_classes),
        ]))

        sections.append(self._section("MOVIMIENTOS", [
            f"Movimientos P1:            {int(self.p1_movements)}",
            f"Movimientos P2:            {int(self.p2_movements)}",
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