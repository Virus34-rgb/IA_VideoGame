class StatsV:
    """
    Reemplaza a Stats. Los contadores siguen siendo escalares acumulados
    (igual que el original) — lo que cambia es CÓMO se alimentan: en vez de
    += 1 por evento individual, se suman tensores (N,) enmascarados por
    "recien_terminadas" para no contar dos veces una partida ya cerrada.
    """
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

        # NUEVO respecto al original: acumuladores POR PARTIDA dentro del
        # lote actual, que se van sumando turno a turno y solo se "cierran"
        # (vuelcan a los contadores agregados de arriba) cuando la partida
        # termina. Se inicializan a tamaño 0 y se dimensionan en start_batch.
        self._p1_damage_batch = None
        self._p2_damage_batch = None
        self._p1_blocks_batch = None
        self._p2_blocks_batch = None
        self._p1_evaded_batch = None
        self._p2_evaded_batch = None
        self._p1_heal_batch = None
        self._p2_heal_batch = None

    def start_batch(self, N):
        """NUEVO. Llamar al inicio de cada lote (en Environment.reset())."""
        import torch
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
        """
        NUEVO. Llamar una vez por turno desde Environment.turn(), ANTES de
        anular reward de partidas ya cerradas — aquí se aplica la misma
        máscara: una partida ya cerrada no debe seguir acumulando stats.
        """
        import torch
        activa = ~ya_terminadas_antes
        self._p1_damage_batch += torch.where(activa, damage_p1, torch.zeros_like(damage_p1))
        self._p2_damage_batch += torch.where(activa, damage_p2, torch.zeros_like(damage_p2))
        self._p1_blocks_batch += torch.where(activa, blocks_p1, torch.zeros_like(blocks_p1))
        self._p2_blocks_batch += torch.where(activa, blocks_p2, torch.zeros_like(blocks_p2))
        self._p1_evaded_batch += torch.where(activa, avoided_p1, torch.zeros_like(avoided_p1))
        self._p2_evaded_batch += torch.where(activa, avoided_p2, torch.zeros_like(avoided_p2))
        self._p1_heal_batch += torch.where(activa, heal_p1, torch.zeros_like(heal_p1))
        self._p2_heal_batch += torch.where(activa, heal_p2, torch.zeros_like(heal_p2))

    def close_finished_games(self, termina_ahora, winner, p1_deaths, p2_deaths, turn_number,
                            por_muerte_mask, por_turnos_mask):
        """
        CAMBIO: recibe las máscaras exactas de causa de fin calculadas en
        _check_end_conditions, en vez de re-derivarlas de forma aproximada.
        """
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

    def guardar_stats(self, path, warriors_classes):
        """Idéntico en espíritu al original — vuelca los contadores agregados a .txt."""
        partidas = max(self.partidas, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Partidas totales: {self.partidas}\n")
            f.write(f"P1 victorias: {self.p1_victories} ({self.p1_victories/partidas*100:.1f}%)\n")
            f.write(f"P2 victorias: {self.p2_victories} ({self.p2_victories/partidas*100:.1f}%)\n")
            f.write(f"Empates: {self.empates} ({self.empates/partidas*100:.1f}%)\n")
            f.write(f"  - por muerte: {self.partidas_por_muerte}\n")
            f.write(f"  - por límite de turnos: {self.partidas_por_limite_turnos}\n")
            f.write(f"Daño medio P1: {self.p1_damage/partidas:.2f}\n")
            f.write(f"Daño medio P2: {self.p2_damage/partidas:.2f}\n")
            f.write(f"Turnos medios: {self.total_turns/partidas:.2f}\n")
            f.write(f"Reward medio P1: {self.total_reward_p1/partidas:.2f}\n")
            f.write(f"Reward medio P2: {self.total_reward_p2/partidas:.2f}\n")