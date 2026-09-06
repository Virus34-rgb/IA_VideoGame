
import torch

import constants
from AI.Environment.abilityData import EffectType


class PlayerRusherV:
    def __init__(self, N: int, environment) -> None:
        self.N = N
        self.environment = environment
        self.name = "PlayerRusherV"
        # Inofensivo: igual que en PlayerNoAIV, evita AttributeError si algún
        # código consulta el Elo de forma incondicional (ej. logging, pool).
        self.elo = constants.ELO_INITIAL

        # Tabla estática (no depende de la partida): daño potencial total por
        # tipo de guerrero, sumando SOLO habilidades de ataque y ponderando
        # por cuántos objetivos alcanza cada una (target_positions). Ej: una
        # habilidad de daño 5 que pega a 3 objetivos suma 15, no 5.
        # Se usa como criterio de draft ("mayor número de ataques" = mayor
        # daño total potencial del tipo).
        self._warrior_damage_score = self._build_warrior_damage_score()

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

    # ------------------------------------------------------------
    # Draft (selección de equipo, modo castillo)
    # ------------------------------------------------------------
    def selection(self, batch_encoded_states, disposition, opp_initial_warrior,
                  castle_alive=None, already_used=None, castle_types=None):
        """
        Debe devolver (item_index, position, action) con la misma forma que
        PlayerAIV.selection():
            item_index: (N,) long — slot de castillo elegido (modo meta-juego)
            position:   (N,) long — posición de combate 0..2
            action:     (N,) long — item_index*3 + position (para remember, aunque
                        el rusher no aprende, TrainerV puede seguir pidiéndolo)

        Auxiliares ya disponibles para tu lógica:
            - self._slots_disponibles(castle_alive, already_used) -> (N, MAX_CASTLE_SIZE) bool
            - self._posiciones_libres(disposition) -> (N, 3) bool
            - self._tipo_por_slot(castle_types, slot) para consultar qué tipo hay en un slot
        """
        if not constants.USE_META_GAME:
            raise NotImplementedError("PlayerRusherV solo soporta el modo castillo (USE_META_GAME=True).")

        item_index, position = self._decidir_seleccion(
            disposition, opp_initial_warrior, castle_alive, already_used, castle_types,
        )
        action = item_index * 3 + position
        return item_index, position, action

    def _decidir_seleccion(self, disposition, opp_initial_warrior, castle_alive, already_used, castle_types):
        """
        Heurística: entre los slots de castillo disponibles (vivos, no usados
        aún en este draft, y cuyo TIPO no esté ya repetido en la disposición
        actual — misma restricción anti-repetición que PlayerAIV._mask_selection),
        elige el de mayor daño potencial total (self._warrior_damage_score).

        La posición de combate se elige simplemente como la primera libre
        (0,1,2 en orden) ya que el rusher no tiene preferencia posicional.
        """
        N = disposition.shape[0]
        num_items = constants.MAX_CASTLE_SIZE

        # --- Máscara de disponibilidad de slots (idéntica a _mask_selection) ---
        item_disponible_base = castle_alive & ~already_used   # (N, MAX_CASTLE_SIZE)

        tipo_usado = torch.zeros(N, constants.WARRIOR_QUANTITY, dtype=torch.bool)
        for slot in range(3):
            tipo = disposition[:, slot]
            mask = tipo > 0
            idx = (tipo - 1).clamp(min=0)
            tipo_usado[mask, idx[mask]] = True

        tipos_slot = castle_alive * castle_types                  # (N, MAX_CASTLE_SIZE), 0 en muertos
        idx_tipo = (tipos_slot - 1).clamp(min=0)                  # (N, MAX_CASTLE_SIZE)
        tipo_disponible = ~tipo_usado.gather(1, idx_tipo)         # (N, MAX_CASTLE_SIZE)

        item_disponible = item_disponible_base & tipo_disponible  # (N, MAX_CASTLE_SIZE)

        # --- Score de daño potencial por slot, según el tipo que contiene ---
        score_por_slot = self._warrior_damage_score[tipos_slot]   # (N, MAX_CASTLE_SIZE)
        score_por_slot = score_por_slot.masked_fill(~item_disponible, float("-inf"))

        item_index = torch.argmax(score_por_slot, dim=1)          # (N,) slot elegido

        # --- Posición de combate: la primera libre en orden 0,1,2 ---
        pos_libre = self._posiciones_libres(disposition)           # (N,3) bool
        # argmax sobre bool da el primer índice True (o 0 si no hay ninguno,
        # pero eso no debería ocurrir salvo error externo: siempre queda hueco
        # cuando se llama a selection() dentro del draft de 3 posiciones).
        position = torch.argmax(pos_libre.int(), dim=1)            # (N,)

        return item_index, position

    # ------------------------------------------------------------
    # Turno de combate
    # ------------------------------------------------------------
    def turn(self, batch_encoded_obs, own_disposition, own_cooldowns, own_alive,
             enemy_disposition, own_instance_abilities):
        """
        Debe devolver actions: (N, 3) long, con valores en {-1, 0,1,2,3 (habilidad), 5,6 (movimiento)}
        igual que PlayerAIV.turn() / PlayerNoAIV.turn().

        Auxiliar ya disponible:
            - self.compute_action_mask(...) -> (N,3,6) bool, misma lógica exacta
              que PlayerAIV/PlayerNoAIV (slot 0-3 = habilidades, 4 = mov+, 5 = mov-
              en el espacio de máscara; recuerda el mapeo con _decode_ability_index
              si trabajas en ese espacio de 0-5 en vez de códigos de entorno).
            - own_health / enemy_health: identifica el lado con
              `own_disposition is self.environment.p1_disposition` igual que hace
              PlayerNoAIV, para saber si mirar p1_healths o p2_healths.
        """
        action_mask = self.compute_action_mask(
            own_disposition, own_cooldowns, own_alive, enemy_disposition, own_instance_abilities,
        )
        soy_p1 = own_disposition is self.environment.p1_disposition
        own_health = self.environment.p1_healths if soy_p1 else self.environment.p2_healths
        enemy_health = self.environment.p2_healths if soy_p1 else self.environment.p1_healths

        actions = self._decidir_turno(
            action_mask, own_disposition, own_cooldowns, own_alive, own_health,
            enemy_disposition, enemy_health, own_instance_abilities,
        )
        return actions

    def _decidir_turno(self, action_mask, own_disposition, own_cooldowns, own_alive,
                        own_health, enemy_disposition, enemy_health, own_instance_abilities):
        """
        Para cada slot vivo, calcula el daño total real que haría cada uno de
        los 4 botones de habilidad (columnas 0-3 de action_mask) SI se pudiera
        usar ahora mismo (respetando la máscara), contando cuántos objetivos
        enemigos vivos alcanzaría realmente esa habilidad — no solo el daño
        base, sino daño_base * nº_objetivos_enemigos_vivos_alcanzados. Elige
        siempre el botón de mayor daño total. Si ningún botón de habilidad es
        válido, cae a movimiento (columnas 4/5) o -1 si tampoco hay eso.

        Nunca elige defender/curar salvo que sea la única opción disponible
        (en cuyo caso, dentro de las 4 columnas de habilidad, su daño real
        calculado será 0 y perderá frente a cualquier ataque válido; si TODAS
        las habilidades disponibles son no-ofensivas, se elige la de mayor
        índice de score, que en empate a 0 simplemente es la primera).
        """
        N = own_disposition.shape[0]
        device = own_disposition.device

        # pool_idx real de cada botón (0-3) por slot: (N,3,4)
        pool_idx = own_instance_abilities

        # tipo de guerrero por slot, expandido a los 4 botones: (N,3,4)
        tipo_expand = own_disposition.unsqueeze(-1).expand(-1, -1, 4)

        damage_base = self.environment.damage_por_tipo_habilidad[tipo_expand, pool_idx]        # (N,3,4)
        effect_type = self.environment.effect_type_por_tipo_habilidad[tipo_expand, pool_idx]    # (N,3,4)
        target_mask = self.environment.target_mask_por_tipo_habilidad[tipo_expand, pool_idx]    # (N,3,4,3)

        es_ataque = effect_type == EffectType.ATTACK   # (N,3,4)

        # Nº de objetivos enemigos vivos que realmente alcanzaría cada botón:
        # target_mask (N,3,4,3) & enemy_alive expandido (N,1,1,3)
        enemy_alive_exp = (enemy_disposition > 0).view(N, 1, 1, 3)
        n_objetivos_alcanzados = (target_mask & enemy_alive_exp).sum(dim=-1).float()   # (N,3,4)

        # Daño total real = daño_base * nº_objetivos, solo si es ataque; 0 en caso contrario
        damage_total = torch.where(es_ataque, damage_base * n_objetivos_alcanzados, torch.zeros_like(damage_base))

        # Aplicar la máscara de validez (columnas 0-3 de action_mask)
        mask_habilidades = action_mask[:, :, :4]   # (N,3,4) bool
        damage_total = damage_total.masked_fill(~mask_habilidades, float("-inf"))

        mejor_boton = torch.argmax(damage_total, dim=-1)                    # (N,3) en {0,1,2,3}
        hay_habilidad_valida = mask_habilidades.any(dim=-1)                  # (N,3) bool

        # Fallback a movimiento si no hay ninguna habilidad válida: se prioriza
        # moverse (columna 4 o 5, lo que esté disponible) antes que quedarse
        # sin acción, ya que el rusher siempre intenta seguir siendo relevante.
        mov_pos_valido = action_mask[:, :, 4]   # (N,3) bool
        mov_neg_valido = action_mask[:, :, 5]   # (N,3) bool

        accion_idx_0_5 = torch.where(
            hay_habilidad_valida, mejor_boton,
            torch.where(mov_pos_valido, torch.full_like(mejor_boton, 4),
                torch.where(mov_neg_valido, torch.full_like(mejor_boton, 5), mejor_boton)),
        )

        codigo_entorno = self._decode_ability_index(accion_idx_0_5)   # (N,3)

        hay_alguna_accion = hay_habilidad_valida | mov_pos_valido | mov_neg_valido
        actions = torch.where(hay_alguna_accion & own_alive, codigo_entorno, torch.full_like(codigo_entorno, -1))

        return actions

    # ------------------------------------------------------------
    # Máscara de acciones válidas — idéntica a PlayerAIV.compute_action_mask
    # y PlayerNoAIV.compute_action_mask. Duplicada aquí por el mismo motivo:
    # no depende de ninguna red neuronal, solo de disposición/cooldowns/vida/
    # habilidades, así que cualquier jugador (humano, IA, scripted) necesita
    # exactamente la misma información para saber qué puede hacer.
    # ------------------------------------------------------------
    def compute_action_mask(self, own_disposition, own_cooldowns, own_alive,
                             enemy_disposition, own_instance_abilities):
        N = own_disposition.shape[0]
        mask = own_alive.unsqueeze(-1).expand(N, 3, 6).clone()

        mask[:, :, :4] &= (own_cooldowns == 0)

        table = self.environment.target_mask_por_tipo_habilidad
        target_mask_pool = table[own_disposition]
        idx = own_instance_abilities.unsqueeze(-1).expand(-1, -1, -1, 3)
        target_mask_full = target_mask_pool.gather(2, idx)

        enemy_ocupado = (enemy_disposition > 0).unsqueeze(1).unsqueeze(1)
        hay_target_valido = (target_mask_full & enemy_ocupado).any(dim=-1)
        sin_target = ~hay_target_valido & target_mask_full.any(dim=-1)

        mask[:, :, :4] &= ~sin_target

        mask[:, 0, 5] = False
        mask[:, 2, 4] = False

        return mask

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