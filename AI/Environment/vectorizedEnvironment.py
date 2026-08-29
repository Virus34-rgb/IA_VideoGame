"""
Entorno vectorizado para Castle Game.

Ejecuta N partidas en paralelo usando tensores de PyTorch.
Todas las operaciones están vectorizadas para maximizar el rendimiento.
"""
import torch
from typing import Tuple, Dict, Any, Optional

from AI.Environment.gameState import GameState
from AI.Environment.statsV import StatsV
from AI.Environment.warriorFactory import get_warriors_classes
from constants import (
    DISCOUNT_FACTOR,
    REWARD_WEIGHTS,
    TURN_PENALTY_BASE,
    TURN_PENALTY_RAMP_START,
    TURN_PENALTY_RAMP_TURNS,
    TURN_PENALTY_MAX,
    DRAW_PENALTY,
    WIN_REWARD,
    MAX_TURNS,
    MAX_DEATHS_PER_TEAM,
)


class VectorizedEnvironment:
    """
    Entorno de batalla vectorizado.

    Gestiona N partidas en paralelo, cada una con dos jugadores (P1 y P2).
    Cada jugador tiene 3 guerreros en 3 posiciones (slots).

    Atributos principales (todos tensores de forma (N, ...)):
        p1_disposition, p2_disposition: (N, 3) IDs de guerreros en cada slot.
        p1_healths, p2_healths: (N, 3) vida actual de cada guerrero.
        p1_cooldowns, p2_cooldowns: (N, 3, 4) cooldowns de habilidades.
        p1_alive, p2_alive: (N, 3) máscara de guerreros vivos.
        p1_deaths, p2_deaths: (N,) número de muertes acumuladas.
        ended: (N,) bool, True si la partida ha terminado.
        winner: (N,) 0=P1, 1=P2, 2=Empate, -1=sin decidir.
        turn_number: (N,) turno actual.
    """

    def __init__(self, N: int) -> None:
        """
        Args:
            N: Número de partidas paralelas.
        """
        self.N: int = N
        self.indices: torch.Tensor = torch.arange(N)

        # Datos estáticos de los guerreros
        self.warriors_classes: Dict[int, Any] = get_warriors_classes()

        # Tablas estáticas (se construyen en _build_static_tables)
        self.max_health_por_tipo: torch.Tensor
        self.speed_por_tipo: torch.Tensor
        self.damage_por_tipo_habilidad: torch.Tensor
        self.can_repeat_por_tipo_habilidad: torch.Tensor
        self.target_mask_por_tipo_habilidad: torch.Tensor
        self._build_static_tables()

        # Estado de las partidas
        self.p1_disposition: torch.Tensor
        self.p2_disposition: torch.Tensor
        self.p1_healths: torch.Tensor
        self.p2_healths: torch.Tensor
        self.p1_cooldowns: torch.Tensor
        self.p2_cooldowns: torch.Tensor
        self.p1_alive: torch.Tensor
        self.p2_alive: torch.Tensor
        self.p1_initialWarrior: torch.Tensor
        self.p2_initialWarrior: torch.Tensor
        self.p1_initialPosition: torch.Tensor
        self.p2_initialPosition: torch.Tensor
        self.p1_deaths: torch.Tensor
        self.p2_deaths: torch.Tensor
        self.ended: torch.Tensor
        self.winner: torch.Tensor
        self.turn_number: torch.Tensor

        # Estadísticas
        self.stats: StatsV = StatsV()
        
        # Inicializar estado (reset implícito)
        self.reset()

    def reset(self) -> GameState:
        """
        Reinicia todas las partidas a su estado inicial.

        Returns:
            GameState: Estado inicial del entorno.
        """
        self.p1_disposition = torch.zeros((self.N, 3), dtype=torch.long)
        self.p2_disposition = torch.zeros((self.N, 3), dtype=torch.long)
        self.p1_healths = torch.zeros((self.N, 3), dtype=torch.float)
        self.p2_healths = torch.zeros((self.N, 3), dtype=torch.float)
        self.p1_cooldowns = torch.zeros((self.N, 3, 4), dtype=torch.bool)
        self.p2_cooldowns = torch.zeros((self.N, 3, 4), dtype=torch.bool)
        self.p1_alive = torch.zeros((self.N, 3), dtype=torch.bool)
        self.p2_alive = torch.zeros((self.N, 3), dtype=torch.bool)
        self.p1_initialWarrior = torch.zeros(self.N, dtype=torch.long)
        self.p2_initialWarrior = torch.zeros(self.N, dtype=torch.long)
        self.p1_initialPosition = torch.zeros(self.N, dtype=torch.long)
        self.p2_initialPosition = torch.zeros(self.N, dtype=torch.long)
        self.p1_deaths = torch.zeros(self.N, dtype=torch.long)
        self.p2_deaths = torch.zeros(self.N, dtype=torch.long)
        self.ended = torch.zeros(self.N, dtype=torch.bool)
        self.winner = torch.full((self.N,), -1, dtype=torch.long)
        self.turn_number = torch.zeros(self.N, dtype=torch.int)

        self.stats.start_batch(self.N)
        return self.get_state()

    def get_state(self) -> GameState:
        """Devuelve el estado actual como un objeto GameState."""
        return GameState(
            self.p1_disposition,
            self.p2_disposition,
            self.p1_deaths,
            self.p2_deaths,
            self.turn_number,
        )

    def team_selection(
        self,
        warrior_p1: torch.Tensor,
        pos1: torch.Tensor,
        warrior_p2: torch.Tensor,
        pos2: torch.Tensor,
        selected: int,
        health1: torch.Tensor,
        health2: torch.Tensor,
    ) -> GameState:
        """
        Selecciona un guerrero para cada jugador en la fase de draft.

        Args:
            warrior_p1, warrior_p2: (N,) IDs de los guerreros seleccionados.
            pos1, pos2: (N,) posiciones (0-2) donde se colocan.
            selected: 0, 1 o 2 (paso de selección).
            health1, health2: (N,) vida inicial de los guerreros.
        """
        self._warrior_selected(warrior_p1, pos1, warrior_p2, pos2, selected, health1, health2)
        return self.get_state()

    def _warrior_selected(
        self,
        warrior1: torch.Tensor,
        pos1: torch.Tensor,
        warrior2: torch.Tensor,
        pos2: torch.Tensor,
        selected: int,
        health1: torch.Tensor,
        health2: torch.Tensor,
    ) -> None:
        """
        Implementación interna de la selección de guerreros.
        """
        # Guardar el primer guerrero seleccionado (para referencia)
        if selected == 0:
            self.p1_initialWarrior[self.indices] = warrior1
            self.p2_initialWarrior[self.indices] = warrior2
            self.p1_initialPosition[self.indices] = pos1
            self.p2_initialPosition[self.indices] = pos2

        # Colocar los guerreros en sus posiciones
        self.p1_disposition[self.indices, pos1] = warrior1
        self.p2_disposition[self.indices, pos2] = warrior2
        self.p1_healths[self.indices, pos1] = health1
        self.p2_healths[self.indices, pos2] = health2
        self.p1_alive[self.indices, pos1] = True
        self.p2_alive[self.indices, pos2] = True

        # Acumular estadísticas de selección
        self.stats.accumulate_warrior_use(warrior1, warrior2)

    def turn(
        self,
        actionsp1: torch.Tensor,
        actionsp2: torch.Tensor,
    ) -> Tuple[GameState, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ejecuta un turno completo para todas las partidas.

        Args:
            actionsp1, actionsp2: (N, 3) acciones elegidas por cada jugador
                                   (índices 0-3 habilidades, 5=movPos, 6=movNeg).

        Returns:
            Tuple con:
                - GameState: nuevo estado del entorno.
                - rewardP1, rewardP2: (N,) recompensas para cada jugador.
                - ended: (N,) bool, True para partidas que han terminado.
        """
        self.turn_number += 1

        # Reiniciar cooldowns de guerreros vivos (duran 1 turno)
        # NOTA: esto es correcto porque los cooldowns se activan en _update_own_cooldowns
        # y se reinician al inicio del siguiente turno.
        self.p1_cooldowns = self.p1_cooldowns & ~self.p1_alive.unsqueeze(-1)
        self.p2_cooldowns = self.p2_cooldowns & ~self.p2_alive.unsqueeze(-1)

        ya_terminadas_antes = self.ended.clone()

        # Determinar el orden de actuación por velocidad
        order, actor_alive_inicio = self._get_turn_order()
        p1_alive_inicio = actor_alive_inicio[:, :3]
        p2_alive_inicio = actor_alive_inicio[:, 3:]

        # Inicializar acumuladores de métricas por turno
        damage_p1 = torch.zeros(self.N)
        damage_p2 = torch.zeros(self.N)
        damage_avoided_p1 = torch.zeros(self.N)
        damage_avoided_p2 = torch.zeros(self.N)
        blocks_p1 = torch.zeros(self.N)
        blocks_p2 = torch.zeros(self.N)
        heal_p1 = torch.zeros(self.N)
        heal_p2 = torch.zeros(self.N)

        # Salud normalizada antes del turno (para shaping de recompensa)
        p1_health_before = self._normalized_team_health(self.p1_healths, self.p1_disposition)
        p2_health_before = self._normalized_team_health(self.p2_healths, self.p2_disposition)

        # Procesar las 6 acciones en orden de velocidad
        for position in range(6):
            actor_idx = order[:, position]
            player = actor_idx // 3
            pos = actor_idx % 3
            es_p1 = (player == 0)

            # Preparar máscaras para seleccionar los datos del actor actual
            player_mask = es_p1.unsqueeze(1)
            player_mask_3 = player_mask.unsqueeze(-1)

            own_disp = torch.where(player_mask, self.p1_disposition, self.p2_disposition)
            enemy_disp = torch.where(player_mask, self.p2_disposition, self.p1_disposition)
            own_health = torch.where(player_mask, self.p1_healths, self.p2_healths)
            enemy_health = torch.where(player_mask, self.p2_healths, self.p1_healths)
            own_actions = torch.where(player_mask, actionsp1, actionsp2)
            actor_action = own_actions.gather(1, pos.unsqueeze(1)).squeeze(1)
            enemy_actions = torch.where(player_mask, actionsp2, actionsp1)
            own_alive = torch.where(player_mask, self.p1_alive, self.p2_alive)
            enemy_alive = torch.where(player_mask, self.p2_alive, self.p1_alive)
            own_cooldowns = torch.where(player_mask_3, self.p1_cooldowns, self.p2_cooldowns)
            actor_type = own_disp.gather(1, pos.unsqueeze(1)).squeeze(1)

            # Resolver la acción del actor actual
            (
                dmg,
                avoided,
                blocked,
                moved,
                healed,
                new_own_disp,
                new_enemy_disp,
                new_own_health,
                new_enemy_health,
                new_own_cd,
                new_own_alive,
                new_enemy_alive,
            ) = self._resolve_action(
                pos,
                actor_type,
                own_disp,
                enemy_disp,
                own_health,
                enemy_health,
                own_cooldowns,
                own_alive,
                enemy_alive,
                actor_action,
                enemy_actions,
            )

            # Actualizar el estado global del entorno para todos los jugadores
            self.p1_disposition = torch.where(player_mask, new_own_disp, new_enemy_disp)
            self.p2_disposition = torch.where(player_mask, new_enemy_disp, new_own_disp)
            self.p1_healths = torch.where(player_mask, new_own_health, new_enemy_health)
            self.p2_healths = torch.where(player_mask, new_enemy_health, new_own_health)
            self.p1_alive = torch.where(player_mask, new_own_alive, new_enemy_alive)
            self.p2_alive = torch.where(player_mask, new_enemy_alive, new_own_alive)
            self.p1_cooldowns = torch.where(player_mask_3, new_own_cd, self.p1_cooldowns)
            self.p2_cooldowns = torch.where(~player_mask_3, new_own_cd, self.p2_cooldowns)

            # Acumular métricas
            damage_p1 += torch.where(es_p1, dmg, torch.zeros_like(dmg))
            damage_p2 += torch.where(es_p1, torch.zeros_like(dmg), dmg)
            damage_avoided_p1 += torch.where(~es_p1, avoided, torch.zeros_like(avoided))
            damage_avoided_p2 += torch.where(~es_p1, torch.zeros_like(avoided), avoided)
            blocks_p1 += torch.where(~es_p1, blocked, torch.zeros_like(blocked))
            blocks_p2 += torch.where(~es_p1, torch.zeros_like(blocked), blocked)
            heal_p1 += torch.where(es_p1, healed, torch.zeros_like(healed))
            heal_p2 += torch.where(~es_p1, healed, torch.zeros_like(healed))

            # Acumular estadísticas de movimientos y ataques
            self.stats.accumulate_movements(moved, es_p1, ~ya_terminadas_antes)
            self.stats.accumulate_attacks(actor_type, actor_action, es_p1, ~ya_terminadas_antes)

        # Calcular el shaping de recompensa (diferencia de salud normalizada)
        p1_health_after = self._normalized_team_health(self.p1_healths, self.p1_disposition)
        p2_health_after = self._normalized_team_health(self.p2_healths, self.p2_disposition)
        health_diff_before = p1_health_before / 3 - p2_health_before / 3
        health_diff_after = p1_health_after / 3 - p2_health_after / 3

        # Contar nuevas muertes
        p1_new_deaths = (p1_alive_inicio & ~self.p1_alive).sum(dim=1).to(self.p1_deaths.dtype)
        p2_new_deaths = (p2_alive_inicio & ~self.p2_alive).sum(dim=1).to(self.p2_deaths.dtype)

        # Acumular estadísticas del turno
        self.stats.accumulate_turn(
            damage_p1,
            damage_p2,
            blocks_p1,
            blocks_p2,
            damage_avoided_p1,
            damage_avoided_p2,
            heal_p1,
            heal_p2,
            ya_terminadas_antes,
        )

        # Calcular recompensas
        rewardP1, rewardP2 = self._calculate_rewards(
            damage_p1,
            damage_p2,
            damage_avoided_p1,
            damage_avoided_p2,
            heal_p1,
            heal_p2,
            health_diff_before,
            health_diff_after,
            p1_new_deaths,
            p2_new_deaths,
        )

        # Las partidas que ya habían terminado no reciben recompensa
        rewardP1 = torch.where(ya_terminadas_antes, torch.zeros_like(rewardP1), rewardP1)
        rewardP2 = torch.where(ya_terminadas_antes, torch.zeros_like(rewardP2), rewardP2)

        return self.get_state(), rewardP1, rewardP2, self.ended

    def _resolve_action(
        self,
        pos: torch.Tensor,
        actors: torch.Tensor,
        own_disposition: torch.Tensor,
        enemy_disposition: torch.Tensor,
        own_health: torch.Tensor,
        enemy_health: torch.Tensor,
        own_cooldowns: torch.Tensor,
        own_alive: torch.Tensor,
        enemy_alive: torch.Tensor,
        actions_actor: torch.Tensor,
        enemy_actions: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Resuelve la acción de un único actor (un slot de un jugador).

        Returns:
            Tupla con:
                - daño infligido al enemigo
                - daño evitado por el enemigo
                - bloqueos del enemigo
                - movió? (0/1)
                - curación realizada
                - nueva disposición propia
                - nueva disposición enemiga (sin cambios)
                - nueva salud propia
                - nueva salud enemiga
                - nuevos cooldowns propios
                - nuevo estado de vida propio
                - nuevo estado de vida enemigo
        """
        # Máscaras de tipos de acción
        mask_movPos = (actions_actor == 5) & (pos != 2)
        mask_movNeg = (actions_actor == 6) & (pos != 0)
        mask_self_heal = (actors == 2) & (actions_actor == 1)
        mask_team_heal = (actors == 5) & (actions_actor == 1)
        # DESPUÉS
        # El actor puede haber muerto en una acción anterior de este mismo turno
        # (el orden se fija al inicio, pero own_alive se actualiza tras cada acción).
        actor_alive_now = own_alive.gather(1, pos.unsqueeze(1)).squeeze(1)

        mask_defend = (
            ((actors == 5) & (actions_actor == 2)) | #El índice de la accion no va de 1-4 (como en warriorFactory)
            ((actors == 3) & (actions_actor == 1)) | #Sino de 0-3
            ((actors == 1) & (actions_actor == 1))
        ) & actor_alive_now
        mask_movPos = mask_movPos & actor_alive_now
        mask_movNeg = mask_movNeg & actor_alive_now
        mask_self_heal = mask_self_heal & actor_alive_now
        mask_team_heal = mask_team_heal & actor_alive_now
        mask_ataque = ~(mask_movPos | mask_movNeg | mask_self_heal | mask_team_heal | mask_defend) & actor_alive_now
        
        # Resolver movimiento (si procede)
        moved, new_disp_mov, new_health_mov, new_cd_mov = self._resolve_action_movement(
            actors, own_disposition, own_health, own_cooldowns, actions_actor, pos
        )

        # Aplicar movimiento
        own_new_disp = own_disposition.clone()
        own_new_disp = torch.where(mask_movPos.unsqueeze(1), new_disp_mov, own_new_disp)
        own_new_disp = torch.where(mask_movNeg.unsqueeze(1), new_disp_mov, own_new_disp)

        # Resolver ataque (si procede)
        damage_raw, blocked_raw, enemy_health_after_attack, enemy_alive_after_attack = (
            self._resolve_action_attack(
                actors,
                actions_actor,
                enemy_disposition,
                enemy_health,
                enemy_alive,
                enemy_actions,
            )
        )
        damage = torch.where(mask_ataque, damage_raw, torch.zeros_like(damage_raw))
        blocked = torch.where(mask_ataque, blocked_raw, torch.zeros_like(blocked_raw))

        # Resolver curación (self o team)
        healed_self, own_health_self = self._resolve_action_self_heal(actors, actions_actor, pos, own_health)
        healed_team, own_health_team = self._resolve_action_team_heal(actors, actions_actor, own_disposition, own_health, own_alive)

        # Aplicar curas y movimiento a la salud
        own_new_health = own_health.clone()
        own_new_health = torch.where(mask_movPos.unsqueeze(1), new_health_mov, own_new_health)
        own_new_health = torch.where(mask_movNeg.unsqueeze(1), new_health_mov, own_new_health)
        own_new_health = torch.where(mask_self_heal.unsqueeze(1), own_health_self, own_new_health)
        own_new_health = torch.where(mask_team_heal.unsqueeze(1), own_health_team, own_new_health)
        enemy_new_health = torch.where(mask_ataque.unsqueeze(1), enemy_health_after_attack, enemy_health)

        # Resolver cooldowns
        mask_usa_habilidad = (mask_ataque | mask_self_heal | mask_team_heal | mask_defend)
        own_cd_new = self._update_own_cooldowns(actors, actions_actor, pos, own_cooldowns, mask_usa_habilidad)

        # Aplicar cooldowns del movimiento
        mask_movPos_4 = mask_movPos.view(-1, 1, 1)
        mask_movNeg_4 = mask_movNeg.view(-1, 1, 1)
        own_cd_new = torch.where(mask_movNeg_4, new_cd_mov, own_cd_new)
        own_cd_new = torch.where(mask_movPos_4, new_cd_mov, own_cd_new)

        # Estado final de vida del enemigo
        enemy_alive_final = torch.where(mask_ataque.unsqueeze(1), enemy_alive_after_attack, enemy_alive)

        # Curación total realizada
        heal = torch.zeros_like(own_health[:, 0])
        heal = torch.where(mask_self_heal, healed_self, heal)
        heal = torch.where(mask_team_heal, healed_team, heal)

        # Daño evitado (por bloqueos/defensa)
        damage_avoided = torch.where(mask_ataque, blocked_raw, torch.zeros_like(blocked_raw))

        return (
            damage,
            damage_avoided,
            blocked,
            moved,
            heal,
            own_new_disp,
            enemy_disposition,  # la disposición enemiga no cambia
            own_new_health,
            enemy_new_health,
            own_cd_new,
            own_alive,          # la vida propia no cambia (salvo que muera después, pero se maneja fuera)
            enemy_alive_final,
        )

    # ------------------------------------------------------------
    # Acciones específicas: movimiento, ataque, cura, cooldowns
    # ------------------------------------------------------------

    def _resolve_action_movement(
        self,
        actors: torch.Tensor,
        own_disposition: torch.Tensor,
        own_health: torch.Tensor,
        own_cooldowns: torch.Tensor,
        actions_actor: torch.Tensor,
        pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Resuelve el movimiento de un guerrero (intercambio de posición).

        Returns:
            Tupla: (movió?, nueva_disposición, nueva_salud, nuevos_cooldowns)
        """
        mask_movPos = (actions_actor == 5) & (pos != 2)
        mask_movNeg = (actions_actor == 6) & (pos != 0)
        moved = (mask_movPos | mask_movNeg).float()

        pos_destino_pos = (pos + 1).clamp(max=2)
        pos_destino_neg = (pos - 1).clamp(min=0)

        # Intercambiar disposición (movimiento positivo)
        own_new_disp = own_disposition.clone()
        origen = own_disposition.gather(1, pos.unsqueeze(1)).squeeze(1)
        destino = own_disposition.gather(1, pos_destino_pos.unsqueeze(1)).squeeze(1)
        own_new_disp.scatter_(1, pos.unsqueeze(1), torch.where(mask_movPos, destino, origen).unsqueeze(1))
        own_new_disp.scatter_(1, pos_destino_pos.unsqueeze(1), torch.where(mask_movPos, origen, destino).unsqueeze(1))

        # Intercambiar disposición (movimiento negativo)
        origen2 = own_new_disp.gather(1, pos.unsqueeze(1)).squeeze(1)
        destino2 = own_new_disp.gather(1, pos_destino_neg.unsqueeze(1)).squeeze(1)
        own_new_disp_final = own_new_disp.clone()
        own_new_disp_final.scatter_(1, pos.unsqueeze(1), torch.where(mask_movNeg, destino2, origen2).unsqueeze(1))
        own_new_disp_final.scatter_(1, pos_destino_neg.unsqueeze(1), torch.where(mask_movNeg, origen2, destino2).unsqueeze(1))

        # Intercambiar salud (movimiento positivo)
        own_new_health = own_health.clone()
        origen_h = own_health.gather(1, pos.unsqueeze(1)).squeeze(1)
        destino_h = own_health.gather(1, pos_destino_pos.unsqueeze(1)).squeeze(1)
        own_new_health.scatter_(1, pos.unsqueeze(1), torch.where(mask_movPos, destino_h, origen_h).unsqueeze(1))
        own_new_health.scatter_(1, pos_destino_pos.unsqueeze(1), torch.where(mask_movPos, origen_h, destino_h).unsqueeze(1))

        # Intercambiar salud (movimiento negativo)
        origen_h2 = own_new_health.gather(1, pos.unsqueeze(1)).squeeze(1)
        destino_h2 = own_new_health.gather(1, pos_destino_neg.unsqueeze(1)).squeeze(1)
        own_new_health_final = own_new_health.clone()
        own_new_health_final.scatter_(1, pos.unsqueeze(1), torch.where(mask_movNeg, destino_h2, origen_h2).unsqueeze(1))
        own_new_health_final.scatter_(1, pos_destino_neg.unsqueeze(1), torch.where(mask_movNeg, origen_h2, destino_h2).unsqueeze(1))

        # Intercambiar cooldowns (movimiento positivo)
        pos_e = pos.view(-1, 1, 1).expand(-1, 1, 4)
        pos_destino_pos_e = pos_destino_pos.view(-1, 1, 1).expand(-1, 1, 4)
        pos_destino_neg_e = pos_destino_neg.view(-1, 1, 1).expand(-1, 1, 4)
        mask_movPos_4 = mask_movPos.view(-1, 1, 1).expand(-1, 1, 4)
        mask_movNeg_4 = mask_movNeg.view(-1, 1, 1).expand(-1, 1, 4)

        own_new_cd = own_cooldowns.clone()
        origen_cd = own_cooldowns.gather(1, pos_e)
        destino_cd = own_cooldowns.gather(1, pos_destino_pos_e)
        own_new_cd.scatter_(1, pos_e, torch.where(mask_movPos_4, destino_cd, origen_cd))
        own_new_cd.scatter_(1, pos_destino_pos_e, torch.where(mask_movPos_4, origen_cd, destino_cd))

        origen_cd2 = own_new_cd.gather(1, pos_e)
        destino_cd2 = own_new_cd.gather(1, pos_destino_neg_e)
        own_new_cd_final = own_new_cd.clone()
        own_new_cd_final.scatter_(1, pos_e, torch.where(mask_movNeg_4, destino_cd2, origen_cd2))
        own_new_cd_final.scatter_(1, pos_destino_neg_e, torch.where(mask_movNeg_4, origen_cd2, destino_cd2))

        return moved, own_new_disp_final, own_new_health_final, own_new_cd_final

    def _resolve_action_attack(
        self,
        actors: torch.Tensor,
        accion_actor: torch.Tensor,
        enemy_disposition: torch.Tensor,
        enemy_health: torch.Tensor,
        enemy_alive: torch.Tensor,
        enemy_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Resuelve un ataque: calcula daño, bloqueos y evasiones.

        Returns:
            Tupla: (daño_total, bloqueos_total, nueva_salud_enemiga, nuevo_estado_vida_enemigo)
        """
        ability_idx = accion_actor.clamp(0, 3)
        would_be_damage = self.damage_por_tipo_habilidad[actors, ability_idx]
        target_mask = self.target_mask_por_tipo_habilidad[actors, ability_idx]

        enemy_new_health = enemy_health.clone()
        damage_total = torch.zeros_like(would_be_damage)
        avoided_total = torch.zeros_like(would_be_damage)
        blocks_total = torch.zeros_like(would_be_damage)

        for slot in range(3):
            es_target = target_mask[:, slot] & enemy_alive[:, slot]
            enemy_id_slot = enemy_disposition[:, slot]
            enemy_action_slot = enemy_actions[:, slot]

            # Bloqueo completo (Knight con Guard Up [idx1] o Rogue con Hide [idx1])
            full_block = ((enemy_id_slot == 1) | (enemy_id_slot == 3)) & (enemy_action_slot == 1)
            # Bloqueo medio (Cleric con Defend [idx2])
            half_block = (enemy_id_slot == 5) & (enemy_action_slot == 2)

            hit_damage = torch.where(
                full_block,
                torch.zeros_like(would_be_damage),
                torch.where(half_block, would_be_damage / 2, would_be_damage),
            )
            avoided = torch.where(
                full_block,
                would_be_damage,
                torch.where(half_block, would_be_damage / 2, torch.zeros_like(would_be_damage)),
            )
            blocked_flag = (full_block | half_block).float()

            # Aplicar solo a los objetivos válidos
            hit_damage = torch.where(es_target, hit_damage, torch.zeros_like(hit_damage))
            avoided = torch.where(es_target, avoided, torch.zeros_like(avoided))
            blocked_flag = torch.where(es_target, blocked_flag, torch.zeros_like(blocked_flag))

            # Reducir salud
            health_slot_actual = enemy_new_health[:, slot]
            enemy_new_health[:, slot] = torch.where(
                es_target,
                health_slot_actual - hit_damage,
                health_slot_actual,
            )

            damage_total += hit_damage
            avoided_total += avoided
            blocks_total += blocked_flag

        enemy_new_alive = enemy_alive & (enemy_new_health > 0)
        return damage_total, blocks_total, enemy_new_health, enemy_new_alive

    def _resolve_action_self_heal(
        self,
        actors: torch.Tensor,
        accion_actor: torch.Tensor,
        pos: torch.Tensor,
        own_health: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Resuelve una curación personal (Archer Heal).

        Returns:
            Tupla: (cantidad_curar, nueva_salud)
        """
        ability_idx = accion_actor.clamp(0, 3)
        heal_amount = self.damage_por_tipo_habilidad[actors, ability_idx]
        max_health_actor = self.max_health_por_tipo[actors]

        current = own_health.gather(1, pos.unsqueeze(1)).squeeze(1)
        new_value = torch.min(max_health_actor, current + heal_amount)
        healed = new_value - current
        own_new_health = own_health.scatter(1, pos.unsqueeze(1), new_value.unsqueeze(1))
        return healed, own_new_health

    def _resolve_action_team_heal(
        self,
        actors: torch.Tensor,
        accion_actor: torch.Tensor,
        own_disposition: torch.Tensor,
        own_health: torch.Tensor,
        own_alive: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Resuelve una curación de equipo (Cleric HealAll).

        Returns:
            Tupla: (curación_total, nueva_salud)
        """
        ability_idx = accion_actor.clamp(0, 3)
        heal_amount = self.damage_por_tipo_habilidad[actors, ability_idx].unsqueeze(1)
        max_health_slot = self.max_health_por_tipo[own_disposition]

        new_value = torch.min(max_health_slot, own_health + heal_amount)
        healed_per_slot = torch.where(own_alive, new_value - own_health, torch.zeros_like(own_health))
        own_new_health = own_health + healed_per_slot
        total_healed = healed_per_slot.sum(dim=1)
        return total_healed, own_new_health

    def _update_own_cooldowns(
        self,
        actors: torch.Tensor,
        accion_actor: torch.Tensor,
        pos: torch.Tensor,
        own_cooldowns: torch.Tensor,
        mask_usa_habilidad: torch.Tensor,
    ) -> torch.Tensor:
        """
        Actualiza los cooldowns después de usar una habilidad.

        Las habilidades no repetibles se marcan como en enfriamiento.
        """
        ability_idx = accion_actor.clamp(0, 3)
        can_repeat = self.can_repeat_por_tipo_habilidad[actors, ability_idx]

        slot_expand = pos.view(-1, 1, 1).expand(-1, 1, 4)
        actor_cd = own_cooldowns.gather(1, slot_expand).squeeze(1)
        reset_cd = torch.zeros_like(actor_cd)
        ability_onehot = torch.nn.functional.one_hot(ability_idx, num_classes=4).bool()

        # Si la habilidad no se puede repetir, activar su cooldown
        marked = torch.where((~can_repeat).unsqueeze(1), reset_cd | ability_onehot, reset_cd)
        new_actor_cd = torch.where(mask_usa_habilidad.unsqueeze(1), marked, actor_cd)

        return own_cooldowns.scatter(1, slot_expand, new_actor_cd.unsqueeze(1))

    def _get_turn_order(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula el orden de actuación basado en la velocidad de los guerreros.

        Returns:
            order: (N, 6) índices de actores (0-5) ordenados por velocidad descendente.
            actor_alive: (N, 6) máscara de actores vivos.
        """
        actor_types = torch.cat([self.p1_disposition, self.p2_disposition], dim=1)
        actor_alive = torch.cat([self.p1_alive, self.p2_alive], dim=1)
        speeds = self.speed_por_tipo[actor_types]
        speeds = torch.where(actor_alive, speeds, torch.full_like(speeds, float("-inf")))
        order = torch.argsort(speeds, dim=1, descending=True, stable=True)
        return order, actor_alive

    def _check_end_conditions(self) -> None:
        """
        Verifica si las partidas han terminado y actualiza winner/ended.
        """
        ya_terminadas = self.ended.clone()

        ambos_muertos = (self.p1_deaths >= MAX_DEATHS_PER_TEAM) & (self.p2_deaths >= MAX_DEATHS_PER_TEAM)
        solo_p1_muerto = (self.p1_deaths >= MAX_DEATHS_PER_TEAM) & ~ambos_muertos
        solo_p2_muerto = (self.p2_deaths >= MAX_DEATHS_PER_TEAM) & ~ambos_muertos & ~solo_p1_muerto
        por_turnos = (self.turn_number > MAX_TURNS) & ~(ambos_muertos | solo_p1_muerto | solo_p2_muerto)
        # Desempate por límite de turnos: gana quien tenga menos bajas; si empatan,
        # quien tenga más vida normalizada de equipo; empate real solo si ambos empatan.
        p1_health_norm = self._normalized_team_health(self.p1_healths, self.p1_disposition)
        p2_health_norm = self._normalized_team_health(self.p2_healths, self.p2_disposition)

        p1_menos_bajas = self.p1_deaths < self.p2_deaths
        p2_menos_bajas = self.p2_deaths < self.p1_deaths
        bajas_iguales = ~p1_menos_bajas & ~p2_menos_bajas

        HEALTH_EPS = 1e-4
        p1_mas_vida = bajas_iguales & (p1_health_norm > p2_health_norm + HEALTH_EPS)
        p2_mas_vida = bajas_iguales & (p2_health_norm > p1_health_norm + HEALTH_EPS)

        por_turnos_gana_p1 = por_turnos & (p1_menos_bajas | p1_mas_vida)
        por_turnos_gana_p2 = por_turnos & (p2_menos_bajas | p2_mas_vida)
        por_turnos_empate = por_turnos & ~(por_turnos_gana_p1 | por_turnos_gana_p2)

        new_winner = self.winner.clone()
        new_winner = torch.where(ambos_muertos, torch.full_like(new_winner, 2), new_winner)
        new_winner = torch.where(solo_p1_muerto, torch.full_like(new_winner, 1), new_winner)
        new_winner = torch.where(solo_p2_muerto, torch.full_like(new_winner, 0), new_winner)
        new_winner = torch.where(por_turnos_gana_p1, torch.full_like(new_winner, 0), new_winner)
        new_winner = torch.where(por_turnos_gana_p2, torch.full_like(new_winner, 1), new_winner)
        new_winner = torch.where(por_turnos_empate, torch.full_like(new_winner, 2), new_winner)

        termina_ahora = (ambos_muertos | solo_p1_muerto | solo_p2_muerto | por_turnos) & ~ya_terminadas

        self.winner = torch.where(termina_ahora, new_winner, self.winner)
        self.ended = self.ended | termina_ahora

        # Acumular estadísticas de cierre
        self.stats.close_finished_games(
            termina_ahora,
            self.winner,
            self.p1_deaths,
            self.p2_deaths,
            self.turn_number,
            por_muerte_mask=(ambos_muertos | solo_p1_muerto | solo_p2_muerto),
            por_turnos_mask=por_turnos,
        )

    def _normalized_team_health(self, healths: torch.Tensor, disposition: torch.Tensor) -> torch.Tensor:
        """ Calcula la salud total normalizada de un equipo."""
        return (healths / self.max_health_por_tipo[disposition]).sum(dim=1)

# DESPUÉS
    def _turn_penalty(self) -> torch.Tensor:
        """
        Penalización de turno progresiva: baja en los primeros turnos (no penaliza
        el posicionamiento inicial) y creciente a partir de TURN_PENALTY_RAMP_START,
        hasta un techo TURN_PENALTY_MAX, para presionar a resolver la partida antes.
        """
        turn = self.turn_number.float()
        exceso = (turn - TURN_PENALTY_RAMP_START).clamp(min=0.0)
        progresion = (exceso / TURN_PENALTY_RAMP_TURNS).clamp(max=1.0)
        return TURN_PENALTY_BASE + progresion * (TURN_PENALTY_MAX - TURN_PENALTY_BASE)

    def _reward(self, **components: torch.Tensor) -> torch.Tensor:
        
        """Calcula la recompensa total como combinación lineal de componentes."""
        
        weighted = sum(REWARD_WEIGHTS[name] * value for name, value in components.items())
        return weighted - self._turn_penalty()

    def _calculate_rewards(
        self,
        damage_p1: torch.Tensor,
        damage_p2: torch.Tensor,
        damage_avoided_p1: torch.Tensor,
        damage_avoided_p2: torch.Tensor,
        healed_p1: torch.Tensor,
        healed_p2: torch.Tensor,
        health_diff_before: torch.Tensor,
        health_diff_after: torch.Tensor,
        newDeaths_p1: torch.Tensor,
        newDeaths_p2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula las recompensas para ambos jugadores al final del turno.
        """
        # Actualizar muertes y verificar condiciones de fin
        self.p1_deaths += newDeaths_p1
        self.p2_deaths += newDeaths_p2
        self._check_end_conditions()

        # Recompensa por victoria/derrota/empate
        gano_p1 = self.winner == 0
        gano_p2 = self.winner == 1
        empate = self.winner == 2
        win_p1 = torch.where(
            gano_p1,
            torch.full_like(damage_p1, WIN_REWARD),
            torch.where(
                gano_p2,
                torch.full_like(damage_p1, -WIN_REWARD),
                torch.where(
                    empate,
                    torch.full_like(damage_p1, -DRAW_PENALTY),
                    torch.zeros_like(damage_p1),
                ),
            ),
        )
        win_p2 = torch.where(
            gano_p2,
            torch.full_like(damage_p1, WIN_REWARD),
            torch.where(
                gano_p1,
                torch.full_like(damage_p1, -WIN_REWARD),
                torch.where(
                    empate,
                    torch.full_like(damage_p1, -DRAW_PENALTY),
                    torch.zeros_like(damage_p1),
                ),
            ),
        )

        # Shaping (diferencia de salud descontada)
        shaping_term_p1 = DISCOUNT_FACTOR * health_diff_after - health_diff_before
        shaping_term_p2 = -shaping_term_p1

        # Recompensa para P1
        rewardP1 = self._reward(
            damage=damage_p1 - damage_p2,
            deaths=newDeaths_p2 - newDeaths_p1,
            win=win_p1,
            blocks=damage_avoided_p1,
            heal=healed_p1,
            shaping_weight=shaping_term_p1,
        )

        # Recompensa para P2
        rewardP2 = self._reward(
            damage=damage_p2 - damage_p1,
            deaths=newDeaths_p1 - newDeaths_p2,
            win=win_p2,
            blocks=damage_avoided_p2,
            heal=healed_p2,
            shaping_weight=shaping_term_p2,
        )

        return rewardP1, rewardP2

    def _build_static_tables(self) -> None:
        """
        Construye las tablas estáticas para el cálculo rápido:
        - max_health_por_tipo
        - speed_por_tipo
        - damage_por_tipo_habilidad
        - can_repeat_por_tipo_habilidad
        - target_mask_por_tipo_habilidad
        """
        num_types = max(self.warriors_classes.keys()) + 1
        num_abilities = 4
        num_slots = 3

        max_health_por_tipo = torch.zeros(num_types, dtype=torch.float)
        speed_por_tipo = torch.zeros(num_types, dtype=torch.float)
        damage_por_tipo_habilidad = torch.zeros(num_types, num_abilities, dtype=torch.float)
        can_repeat_por_tipo_habilidad = torch.zeros(num_types, num_abilities, dtype=torch.bool)
        target_mask_por_tipo_habilidad = torch.zeros(num_types, num_abilities, num_slots, dtype=torch.bool)

        for warrior_id, warrior_data in self.warriors_classes.items():
            max_health_por_tipo[warrior_id] = warrior_data.max_health
            speed_por_tipo[warrior_id] = warrior_data.speed
            for ability_idx, ability in enumerate(warrior_data.abilities):
                damage_por_tipo_habilidad[warrior_id, ability_idx] = ability.damage
                can_repeat_por_tipo_habilidad[warrior_id, ability_idx] = ability.can_repeat
                for target_pos in ability.target_positions:
                    target_mask_por_tipo_habilidad[warrior_id, ability_idx, target_pos] = True

        self.max_health_por_tipo = max_health_por_tipo
        self.speed_por_tipo = speed_por_tipo
        self.damage_por_tipo_habilidad = damage_por_tipo_habilidad
        self.can_repeat_por_tipo_habilidad = can_repeat_por_tipo_habilidad
        self.target_mask_por_tipo_habilidad = target_mask_por_tipo_habilidad