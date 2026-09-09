
import torch

from AI.Environment.action_mask import compute_action_mask
import constants
from AI.Environment.abilityData import EffectType


import torch

import constants
from AI.Environment.abilityData import EffectType


class PlayerRusherV:
    def __init__(self, N: int, environment, aggression: float = 1.0) -> None:
        self.N = N
        self.environment = environment
        self.name = "PlayerRusherV"
        self.elo = constants.ELO_INITIAL
        self.aggression = self._as_tensor(aggression)

        self._warrior_damage_score = self._build_warrior_damage_score()
        self._warrior_defensive_score = self._build_warrior_defensive_score()

    def _as_tensor(self, value) -> torch.Tensor:
        """Normaliza aggression a tensor (N,) en [0,1], acepte float o tensor."""
        if isinstance(value, torch.Tensor):
            t = value.float().clamp(0.0, 1.0)
            if t.dim() == 0:
                t = t.expand(self.N).clone()
            return t
        return torch.full((self.N,), float(value), dtype=torch.float).clamp(0.0, 1.0)

    def set_aggression(self, aggression) -> None:
        """Permite cambiar la agresividad entre partidas/batches sin recrear
        la instancia (útil para samplear un valor distinto cada batch desde
        TrainerV, ver guía punto 4)."""
        self.aggression = self._as_tensor(aggression)

    def _build_warrior_damage_score(self) -> torch.Tensor:
        """(WARRIOR_QUANTITY+1,) float — índice 0 vacío (score 0), índices 1..5
        con la suma de daño*nº_objetivos de sus habilidades de ATAQUE."""
        num_types = self.environment.damage_por_tipo_habilidad.shape[0]
        score = torch.zeros(num_types, dtype=torch.float)
        for warrior_id, warrior_data in self.environment.warriors_classes.items():
            total = 0.0
            for ability in warrior_data.ability_pool:
                if ability.effect_type == EffectType.ATTACK:
                    n_objetivos = max(len(ability.target_positions), 1)
                    total += ability.damage * n_objetivos
            score[warrior_id] = total
        return score

    def _build_warrior_defensive_score(self) -> torch.Tensor:
        """(WARRIOR_QUANTITY+1,) float — análogo a _build_warrior_damage_score
        pero sumando solo habilidades DEFEND_FULL/DEFEND_HALF/SELF_HEAL/TEAM_HEAL.
        Para defensas (daño=0) cuenta la habilidad como 1 punto en vez de 0,
        para que un guerrero con muchas defensas no puntúe 0 y quede
        indistinguible de uno sin ninguna."""
        num_types = self.environment.damage_por_tipo_habilidad.shape[0]
        score = torch.zeros(num_types, dtype=torch.float)
        defensive_types = (
            EffectType.DEFEND_FULL, EffectType.DEFEND_HALF,
            EffectType.SELF_HEAL, EffectType.TEAM_HEAL,
        )
        for warrior_id, warrior_data in self.environment.warriors_classes.items():
            total = 0.0
            for ability in warrior_data.ability_pool:
                if ability.effect_type in defensive_types:
                    total += max(ability.damage, 1)
            score[warrior_id] = total
        return score

    # ------------------------------------------------------------
    # Draft (selección de equipo, modo castillo)
    # ------------------------------------------------------------
    def selection(self, batch_encoded_states, disposition, opp_initial_warrior,
                  castle_alive=None, already_used=None, castle_types=None):
        if not constants.USE_META_GAME:
            raise NotImplementedError("PlayerRusherV solo soporta el modo castillo (USE_META_GAME=True).")

        item_index, position = self._decidir_seleccion(
            disposition, opp_initial_warrior, castle_alive, already_used, castle_types,
        )
        action = item_index * 3 + position
        return item_index, position, action

    def _decidir_seleccion(self, disposition, opp_initial_warrior, castle_alive, already_used, castle_types):
        """
        CAMBIADO: antes elegía siempre el slot de mayor _warrior_damage_score.
        Ahora el score combina daño y defensa ponderado por self.aggression:
            score = aggression * score_daño + (1 - aggression) * score_defensivo
        Con aggression=1.0 es exactamente el comportamiento original (rusher
        puro). Con aggression=0.0 prioriza guerreros con más herramientas
        defensivas/curación en vez de daño puro.
        """
        N = disposition.shape[0]

        item_disponible_base = castle_alive & ~already_used

        tipo_usado = torch.zeros(N, constants.WARRIOR_QUANTITY, dtype=torch.bool)
        for slot in range(3):
            tipo = disposition[:, slot]
            mask = tipo > 0
            idx = (tipo - 1).clamp(min=0)
            tipo_usado[mask, idx[mask]] = True

        tipos_slot = castle_alive * castle_types
        idx_tipo = (tipos_slot - 1).clamp(min=0)
        tipo_disponible = ~tipo_usado.gather(1, idx_tipo)

        item_disponible = item_disponible_base & tipo_disponible

        # NUEVO: score combinado en vez de solo score de daño
        score_dano_slot = self._warrior_damage_score[tipos_slot]
        score_def_slot = self._warrior_defensive_score[tipos_slot]
        agg = self.aggression.view(N, 1)   # (N,1) para broadcastear contra (N, MAX_CASTLE_SIZE)
        score_por_slot = agg * score_dano_slot + (1.0 - agg) * score_def_slot
        score_por_slot = score_por_slot.masked_fill(~item_disponible, float("-inf"))

        item_index = torch.argmax(score_por_slot, dim=1)

        pos_libre = self._posiciones_libres(disposition)
        position = torch.argmax(pos_libre.int(), dim=1)

        return item_index, position

    # ------------------------------------------------------------
    # Turno de combate
    # ------------------------------------------------------------
    def turn(self, batch_encoded_obs, own_disposition, own_cooldowns, own_alive,
             enemy_disposition, own_instance_abilities):
        action_mask = compute_action_mask(
            own_disposition, own_cooldowns, own_alive, enemy_disposition, own_instance_abilities
            ,self.environment.target_mask_por_tipo_habilidad
        )
        actions = self._decidir_turno(
            action_mask, own_disposition, own_cooldowns, own_alive,
            enemy_disposition, own_instance_abilities,
        )
        return actions

    def _decidir_turno(self, action_mask, own_disposition, own_cooldowns, own_alive,
                        enemy_disposition, own_instance_abilities):
        """
        CAMBIADO respecto a la versión original: antes siempre elegía el
        botón de máximo daño real entre las 4 habilidades. Ahora, para cada
        slot vivo, se decide PROBABILÍSTICAMENTE por partida (no por slot: la
        misma tirada de moneda aplica a los 3 slots de una partida, para que
        el estilo sea consistente durante el turno) entre:
          - con prob. self.aggression: comportamiento rusher puro (máximo
            daño real, igual que antes)
          - con prob. (1 - self.aggression): comportamiento "turtle": si hay
            alguna habilidad de defensa/curación válida, se prioriza esa
            (defensa si el slot puede ser atacado — reutilizando
            _check_if_targeted del entorno sería ideal pero para no acoplar
            el rusher a resolveAction, aquí se prioriza simplemente
            defensa > curación > ataque cuando todas están disponibles);
            si no hay ninguna válida, cae al mismo criterio de máximo daño.

        El resto de la lógica (fallback a movimiento, decodificación de
        acción) es idéntica a la versión original.
        """
        N = own_disposition.shape[0]

        pool_idx = own_instance_abilities
        tipo_expand = own_disposition.unsqueeze(-1).expand(-1, -1, 4)

        damage_base = self.environment.damage_por_tipo_habilidad[tipo_expand, pool_idx]
        effect_type = self.environment.effect_type_por_tipo_habilidad[tipo_expand, pool_idx]
        target_mask = self.environment.target_mask_por_tipo_habilidad[tipo_expand, pool_idx]

        es_ataque = effect_type == EffectType.ATTACK
        es_defensa = (effect_type == EffectType.DEFEND_FULL) | (effect_type == EffectType.DEFEND_HALF)
        es_cura = (effect_type == EffectType.SELF_HEAL) | (effect_type == EffectType.TEAM_HEAL)

        enemy_alive_exp = (enemy_disposition > 0).view(N, 1, 1, 3)
        n_objetivos_alcanzados = (target_mask & enemy_alive_exp).sum(dim=-1).float()

        damage_total = damage_base * n_objetivos_alcanzados * es_ataque.float()

        mask_habilidades = action_mask[:, :, :4]

        # --- Score "rusher puro" (comportamiento original) ---
        score_rusher = damage_total.masked_fill(~mask_habilidades, float("-inf"))

        # --- Score "turtle": prioridad defensa > cura > ataque ---
        # Usamos una escala grande y separada por tipo para que defensa
        # siempre gane a cura, y cura siempre gane a ataque, cuando ambas
        # están disponibles simultáneamente — evita tener que normalizar
        # daño real de ataque contra "utilidad" de defensa/cura, que no son
        # comparables directamente.
        PRIORIDAD_DEFENSA = 2.0
        PRIORIDAD_CURA = 1.0
        score_turtle = torch.zeros_like(damage_total)
        score_turtle.masked_fill_(es_defensa, PRIORIDAD_DEFENSA)
        score_turtle.masked_fill_(es_cura & (score_turtle == 0), PRIORIDAD_CURA)
        # si ninguna opción es defensa/cura, cae al mismo criterio de daño
        # que el rusher puro (evita quedarse sin hacer nada útil)
        hay_defensa_o_cura = (es_defensa | es_cura) & mask_habilidades
        usar_dano_como_fallback = ~hay_defensa_o_cura.any(dim=-1, keepdim=True)
        score_turtle = torch.where(usar_dano_como_fallback.expand_as(score_turtle), damage_total, score_turtle)
        score_turtle = score_turtle.masked_fill(~mask_habilidades, float("-inf"))

        # --- Elección probabilística por partida (misma tirada para los 3 slots) ---
        usa_rusher = (torch.rand(N, device=own_disposition.device) < self.aggression).view(N, 1).expand(N, 3)
        
        usa_rusher = usa_rusher.unsqueeze(-1)  # (N, 3, 1)
        score_final = torch.where(usa_rusher, score_rusher, score_turtle)
        
        mejor_boton = torch.argmax(score_final, dim=-1)
        hay_habilidad_valida = mask_habilidades.any(dim=-1)

        mov_pos_valido = action_mask[:, :, 4]
        mov_neg_valido = action_mask[:, :, 5]

        accion_idx_0_5 = torch.where(
            hay_habilidad_valida, mejor_boton,
            torch.where(mov_pos_valido, torch.full_like(mejor_boton, 4),
                torch.where(mov_neg_valido, torch.full_like(mejor_boton, 5), mejor_boton)),
        )

        codigo_entorno = self._decode_ability_index(accion_idx_0_5)

        hay_alguna_accion = hay_habilidad_valida | mov_pos_valido | mov_neg_valido
        actions = torch.where(hay_alguna_accion & own_alive, codigo_entorno, torch.full_like(codigo_entorno, -1))

        return actions

    # ------------------------------------------------------------
    # Auxiliares de draft (para usar dentro de _decidir_seleccion)
    # ------------------------------------------------------------
    @staticmethod
    def _slots_disponibles(castle_alive, already_used):
        """(N, MAX_CASTLE_SIZE) bool — slots vivos y aún no usados en este draft."""
        return castle_alive & ~already_used

    @staticmethod
    def _posiciones_libres(disposition):
        """(N, 3) bool — posiciones de combate aún sin ocupar."""
        return disposition == 0

    @staticmethod
    def _tipo_por_slot(castle_types, castle_alive):
        """(N, MAX_CASTLE_SIZE) long — tipo de guerrero en cada slot, 0 si muerto."""
        return castle_types * castle_alive.long()

    # ------------------------------------------------------------
    # Auxiliares de turno (para usar dentro de _decidir_turno)
    # ------------------------------------------------------------
    @staticmethod
    def _decode_ability_index(idx_0_5):
        """Convierte índice de máscara (0-5) a código de acción de entorno.
        0-3 se mantienen igual (habilidad), 4->5 (mov+), 5->6 (mov-)."""
        return torch.where(
            idx_0_5 == 4, torch.full_like(idx_0_5, 5),
            torch.where(idx_0_5 == 5, torch.full_like(idx_0_5, 6), idx_0_5),
        )

    def _damage_table_for(self, own_disposition):
        """(N,3) tipo de guerrero -> usar junto con environment.damage_por_tipo_habilidad
        para consultar el daño de cada botón de habilidad (0-3) por slot:
            own_instance_abilities (N,3,4) da el pool_idx real de cada botón
            self.environment.damage_por_tipo_habilidad[tipo, pool_idx] da el daño
        """
        return own_disposition

    # ------------------------------------------------------------
    # No aplica: sin red neuronal, sin ruido, sin checkpoints
    # ------------------------------------------------------------
    def reset_noise(self):
        pass

    def update_epsilon(self, n_games: int = 1) -> None:
        pass