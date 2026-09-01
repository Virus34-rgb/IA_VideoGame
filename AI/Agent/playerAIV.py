"""
Agente DQN para Castle Game.
"""
import torch
from torch import nn
from typing import Tuple, Optional, Any, Dict, List

from AI.Agent.turnNetwork import TurnNetwork
from AI.Agent.selectionNetwork import SelectionNetwork
from AI.Agent.replayMemoryAN import ReplayMemoryAN
from AI.Agent.replayMemoryPM import ReplayMemoryPM
import constants


class PlayerAIV:
    def __init__(self, N: int, environment: Any, use_replay: bool = True) -> None:
        self.N: int = N
        self.environment: Any = environment
        self.name: str = "DqnPlayerV"

        self.epsilon_sel: float = constants.EPSILON_SELECTION
        self.epsilon_turn: float = constants.EPSILON_TURN
        self.epsilon_residual: float = constants.EPSILON_RESIDUAL
        
        selection_state_dim = constants.get_selection_state_dim()
        self.selection_network: SelectionNetwork = SelectionNetwork(sigma_init=constants.NOISY_SIGMA_INIT,input_size=selection_state_dim)
        self.target_selection_network: SelectionNetwork = SelectionNetwork(sigma_init=constants.NOISY_SIGMA_INIT,input_size=selection_state_dim)
        self.target_selection_network.load_state_dict(self.selection_network.state_dict())
        self.optimizer_sel = torch.optim.Adam(self.selection_network.parameters(), lr=constants.SELECTION_LEARNING_RATE,foreach=True)

        self.turn_network: TurnNetwork = TurnNetwork(sigma_init=constants.NOISY_SIGMA_INIT)
        self.target_turn_network: TurnNetwork = TurnNetwork(sigma_init=constants.NOISY_SIGMA_INIT)
        self.target_turn_network.load_state_dict(self.turn_network.state_dict())
        self.optimizer_turn = torch.optim.Adam(self.turn_network.parameters(), lr=constants.TURN_LEARNING_RATE,foreach=True)

        if use_replay:
            self.replay_memory_sel = ReplayMemoryPM(
                constants.SELECTION_REPLAY_DATA, state_dim=selection_state_dim,  
            )
            self.replay_memory_turn = ReplayMemoryAN(
                constants.TURN_REPLAY_DATA, state_dim=constants.TURN_STATE_DIM,             
            )
        else:
            self.replay_memory_sel = None
            self.replay_memory_turn = None

        self.replayed_selection: int = 0
        self.replayed_turn: int = 0
        self.elo = constants.ELO_INITIAL
        self._turn_offsets = torch.tensor([0, 6, 12])

    def remember_selection_batch(self, c_state, action, reward, next_c_state, done) -> None:
        self.replay_memory_sel.push_batch(c_state, action, reward, next_c_state, done)

    def remember_turn_batch(
        self, observation, action, reward, next_observation, done,
        alive, types, cooldowns, opp_types, next_types, next_alive, next_cooldowns, next_opp_types,
        instance_abilities, next_instance_abilities,
        action_mask, next_action_mask,
    ) -> None:
        self.replay_memory_turn.push_batch(
            observation, action, reward, next_observation, done,
            alive, types, cooldowns, opp_types, next_types, next_alive, next_cooldowns, next_opp_types,
            instance_abilities, next_instance_abilities,
            action_mask, next_action_mask,
        )

    def replay_selection(self) -> Optional[float]:
        self.selection_network.reset_noise()
        self.target_selection_network.reset_noise()

        if len(self.replay_memory_sel) < constants.BATCH_SIZE:
            return None

        self.replayed_selection += 1
        #batch: Data, tree_indices: np.ndarray de los índices dentro de sum_tree, weights: np.ndarray de pesos dentro de la red 
        batch, tree_indices, weights = self.replay_memory_sel.sample(constants.BATCH_SIZE)
        weights = torch.from_numpy(weights).float() # los pesos (debido a PER) vienen como un array de numpy, los convertimos a tensor de torch

        states = batch.states.float()# convertimos los estados a float32 (antes float16) para que la red los pueda procesar
        actions = batch.actions
        rewards = batch.rewards
        next_states = batch.next_states.float() # convertimos los estados a float32 (antes float16) para que la red los pueda procesar
        dones = batch.dones

        qvalues = self.selection_network(states) # obtemenos los qvalues de la red para los estados actuales
        #q_selected tiene la forma (N(batch_size),num_actions) mientras que actions tiene la forma (N(batch_size),)
        #Debido a esto, le añadimos una dimensión extra a acions para que coincida en forma a q_selected(N,1) y luego eliminamos esa dimensión extra con squeeze(1)
        q_selected = qvalues.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad(): # no necesitamos calcular gradientes para el target, ya que no se hace backward sobre él
            #calculas la siguiente accion teniendo en cuenta next_states y obtenemos los qbalues para la accion optima con la misma lógica
            next_actions = self.selection_network(next_states).argmax(dim=1)
            next_qvalues = self.target_selection_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            #se calcula el target de la red con la formula de Bellman, teniendo en cuenta si el estado es terminal o no
            target = rewards + constants.DISCOUNT_FACTOR * next_qvalues * (~dones)
            #calculamos los errores de TD (diferencia entre q_selected y target) y los normalizamos para evitar NaN o inf
            td_errors = torch.abs(q_selected - target)
            td_errors = torch.nan_to_num(td_errors, nan=1.0, posinf=10.0, neginf=10.0)

        #Calculamos la perdida mediante la funcion de perdida y optimizamos la red con el optimizador correspondiente
        loss = self._loss_function(q_selected, target, weights)
        self._optimize_step(loss, self.optimizer_sel, self.selection_network, self.target_selection_network, "replayed_selection")
        #actualizamos las prioridades de los indices de la memoria de replay con los errores de TD calculados
        self.replay_memory_sel.update_priorities(tree_indices, td_errors.detach().cpu().numpy())
        return loss.item()

    def replay_turn(self) -> Optional[float]:
        self.turn_network.reset_noise()
        self.target_turn_network.reset_noise()

        if len(self.replay_memory_turn) < constants.BATCH_SIZE:
            return None

        self.replayed_turn += 1
        batch, tree_indices, weights = self.replay_memory_turn.sample(constants.BATCH_SIZE)
        weights = torch.from_numpy(weights).float()

        states =  batch.states.float()
        warrior_mask = batch.alive
        next_states = batch.next_states.float()
        rewards = batch.rewards
        dones = batch.dones
        
        #convertimos las acciones de entorno (0-6 esquivando 4(soy imbecil)) a acciones de red (0-5) 
        # para poder indexar los qvalues de la red
        #actiosn B tiene la forma (N,3) por lo que no requiere de un squeeze(1) como en replay_selection
        actions_b = self._environment_action_to_network(batch.actions)

        #cargamos la mascara de acciones validas para el batch de estados actual, que nos indica que acciones son validas y cuales no
        current_action_mask = batch.action_mask

        qvalues = self.turn_network(states, action_mask=current_action_mask)

        offsets = self._turn_offsets
        actions_global = actions_b + offsets
        q_selected = qvalues.gather(1, actions_global)
        q_selected = q_selected * warrior_mask.float() # si esta muerto se deshabilitan todas las acciones
        q_selected = q_selected.sum(dim=1)

        target = self._multi_agent_double_dqn_target(batch, next_states, rewards, dones)

        with torch.inference_mode(): # no necesitamos calcular gradientes para el target, ya que no se hace backward sobre él
            td_errors = torch.abs(q_selected - target)
            td_errors = torch.nan_to_num(td_errors, nan=1.0, posinf=10.0, neginf=10.0)

        loss = self._loss_function(q_selected, target, weights)
        self._optimize_step(loss, self.optimizer_turn, self.turn_network, self.target_turn_network, "replayed_turn")
        self.replay_memory_turn.update_priorities(tree_indices, td_errors.detach().cpu().numpy())
        return loss.item()

    @staticmethod
    def _loss_function(input, target, weights):
        loss = nn.SmoothL1Loss(reduction="none")(input, target)
        return (loss * weights).mean()

    def selection(self, batch_encoded_states, disposition, opp_initial_warrior, castle_alive=None, already_used=None,castle_types = None):
        """
        Modo catálogo (USE_META_GAME=False): elige (tipo 1..5, posición 0-2) libremente.
        Modo castillo (USE_META_GAME=True): elige (slot de castillo 0..MAX_CASTLE_SIZE-1, posición 0-2)
            libremente entre las instancias vivas y no usadas aún en este draft.
        """
        if constants.RESET_IN_DECISIONS:
            self.selection_network.reset_noise()

        states = batch_encoded_states.float() # conviertes los estados a float32 (antes float16) para que la red los pueda procesar
        #Siempre que no vayamos a llamar a backward, es mejor usar torch.inference_mode() para ahorrar memoria y tiempo de computo, 
        # ya que no se guardan los gradientes ni se hace tracking de operaciones
        with torch.inference_mode():
            logits = self.selection_network(states) # recibes los qvalues de la red para los estados actuales
        masked_logits = self._mask_selection(logits, disposition, castle_alive, already_used,castle_types) #enmascaras las acciones invalidas (ya usadas o slots muertos) para que no sean seleccionadas

        greedy = torch.argmax(masked_logits, dim=1) # si no se explora, se elige la accion con mayor qvalue
        random_action = self._random_valid_action(masked_logits) # si se explora se elige una accion aleatoria entre las validas

        epsilon_efectivo = self.epsilon_residual if not self.selection_network.training else 0.0 #aun si se esta evaluando se puede forzar un margen de exploración
        explora = (torch.rand(self.N) < epsilon_efectivo) | (opp_initial_warrior == 0) # si el guerrero inicial del rival es 0, se fuerza a explorar para evitar que en la evaluación se haga lo mismo (los dos seleccionan el mismo inicial, el mismo secundario,...)
        action = torch.where(explora, random_action, greedy) # si se explora se elige la accion aleatoria, si no se explora se elige la accion greedy

        item_index = action // 3 # si se esta en modo castillo, item_index es el slot del castillo (0..MAX_CASTLE_SIZE-1), si se esta en modo catálogo, item_index es el tipo de guerrero (0..WARRIOR_QUANTITY-1)
        position = action % 3 # la posición es la misma para ambos modos (0..2)

        if not constants.USE_META_GAME:
            item_index = item_index + 1   # compatibilidad histórica: tipo 1..WARRIOR_QUANTITY

        return item_index, position, action

    def _mask_selection(self, logits, disposition, castle_alive=None, already_used=None,castle_types = None):
        N = disposition.shape[0]
        ocupado_pos = disposition > 0   # (N,3)

        if constants.USE_META_GAME:
            num_items = constants.MAX_CASTLE_SIZE

            item_disponible_base = castle_alive & ~already_used   # (N, MAX_CASTLE_SIZE)

            tipo_usado = torch.zeros(N, constants.WARRIOR_QUANTITY, dtype=torch.bool, device=disposition.device) # (N, WARRIOR_QUANTITY)  BOOL
            for slot in range(3):
                tipo = disposition[:, slot] # (N,)  0 para muerto, 1..WARRIOR_QUANTITY para vivos
                mask = (tipo > 0)
                idx = (tipo - 1).clamp(min=0) #(N,)  0..WARRIOR_QUANTITY-1 para vivos
                #Tipo usado (N,5)
                #Mask (N,) indica que guerrero esta usado (tipo 1..5) en la disposición actual, para evitar que se repita en el draft
                #idx (N,) indica el indice del guerrero usado (tipo 0..4)
                #Primero con mask te quedas con las filas que son true
                #Luego con idx[mask] te quedas con los indices de los guerreros usados en esas filas
                #Los marcas como true en tipo_usado para indicar que esos guerreros ya estan usados en la disposición actual y no se pueden seleccionar de nuevo
                tipo_usado[mask, idx[mask]] = True 
                
            #Se pone a 0 todos los tipos del castillo que no esten muertos
            #Ambos son (N, MAX_CASTLE_SIZE) solo que castle alive es de boolean y castle_types de int
            tipos_slot = castle_alive * castle_types  # (N, MAX_CASTLE_SIZE)  0 para muertos
            #Conviertes los tipos (1-5) a indices de la red (0,5)
            idx_tipo = (tipos_slot - 1).clamp(min=0)  # (N, MAX_CASTLE_SIZE)
            #Obtenemos un booleano (N, MAX_CASTLE_SIZE) indicando si ese tipo ya está usado
            #tipo_usado (N, WARRIOR_QUANTITY)
            #idx_tipo (N, MAX_CASTLE_SIZE)
            #torch.gather toma el valor de tipo usado, con el indice index de index = idx_tipo[i,j] (gather en dimension 1) y se lo asigna a tipo_disponible
            tipo_disponible = ~tipo_usado.gather(1, idx_tipo)
            #Los slots muertos (tipo 0) quedarán como ~tipo_usado[0], que es True, pero luego se filtran con castle_alive
            #Enmascaras entre los guerreros disponibles del castillo y los guerreros cuyo tipo no esta disponible
            item_disponible = item_disponible_base & tipo_disponible

        else:
            # Modo catálogo (histórico): sin repetición de tipos
            num_items = constants.WARRIOR_QUANTITY
            usados_tipo = torch.zeros(N, constants.WARRIOR_QUANTITY, dtype=torch.bool)
            for slot in range(3):
                tipo_en_slot = disposition[:, slot]
                hay_tipo = tipo_en_slot > 0
                idx = (tipo_en_slot - 1).clamp(min=0)
                if hay_tipo.any():
                    rows = torch.arange(N)[hay_tipo]
                    usados_tipo[rows, idx[hay_tipo]] = True
            item_disponible = ~usados_tipo

        # Luego, la lógica de expansión de posiciones es la misma para ambos modos
        #Antes (N,MAX_CASTLE_SIZE) después (N,MAX_CASTLE_SIZE,3) (es decir triplica la máscara para cada posición)
        item_expand = item_disponible.unsqueeze(-1).expand(N, num_items, 3)
        #Antes (N,3) después (N,MAX_CASTLE_SIZE,3) (es decir amplia la máscara para cada slot de castillo)
        pos_libre = (~ocupado_pos).unsqueeze(1).expand(N, num_items, 3)

        #La máscara final es la intersección de las dos máscaras anteriores, es decir, solo se permiten las acciones que 
        # estén disponibles en el castillo y que estén en posiciones libres
        #(N,MAX_CASTLE_SIZE * 3) bool
        mask = (item_expand & pos_libre).reshape(N, num_items * 3)
        return logits.masked_fill(~mask, float("-inf"))

    def turn(self, batch_encoded_obs, own_disposition, own_cooldowns, own_alive, enemy_disposition, own_instance_abilities):

        if constants.RESET_IN_DECISIONS:
            self.turn_network.reset_noise()

        obs = batch_encoded_obs.float() #conviertes el estado a float32 (antes float16) para que la red lo pueda procesar
        with torch.inference_mode():
            logits = self.turn_network(obs) #obtenemos los qvalues de la red para los estados actuales
        #Enmascaras los q-values para seleccionar unicamente acciones validas
        masked_logits = self.mask_turn(own_disposition, own_cooldowns, own_alive, enemy_disposition, own_instance_abilities, logits)
        #conviertes los logits en una forma (N,3,6) para poder seleccionar la acción por guerrero y posición
        masked_3d = masked_logits.view(self.N, 3, 6)
        #compruebas si hay alguna acción valida para cada guerrero y posición, si no hay ninguna acción valida se marca como False
        hay_valida = (masked_3d != float("-inf")).any(dim=-1)

        greedy = torch.argmax(masked_3d, dim=-1) #si no se explora, se elige la accion con mayor qvalue
        random_action = self._random_valid_action(masked_3d.reshape(self.N * 3, 6)) # si se explora se elige una accion aleatoria entre las validas
        random_action = random_action.view(self.N, 3) # conviertes la acción aleatoria en una forma (N,3) para poder seleccionar la acción por guerrero y posición 
        
        epsilon_efectivo = self.epsilon_residual if not self.turn_network.training else 0.0
        explora = torch.rand(self.N, 3) < epsilon_efectivo #mascara de exploracion para cada guerrero y partida
        elegido = torch.where(explora, random_action, greedy) # si explora se elige la accion aleatoria, si no se explora se elige la accion greedy

        codigo = self._decode_ability_index(elegido) #conviertes la acción elegida en el código de habilidad correspondiente (0-6)
        actions = torch.where(hay_valida, codigo, torch.full_like(codigo, -1))# si no hay ninguna acción valida para un guerrero y posición, se marca como -1 (ninguna acción)
        return actions #N,3

    def compute_action_mask(self, own_disposition, own_cooldowns, own_alive, enemy_disposition, own_instance_abilities):
        """
        Calcula la máscara booleana de acciones válidas (N, 3, 6), sin aplicarla a
        ningún logit. Separado de mask_turn para poder calcularla una única vez en
        el momento de recolección y reutilizarla desde el replay buffer, en vez de
        recalcularla en cada sample de replay_turn/_multi_agent_double_dqn_target.
        """
        N = own_disposition.shape[0] #cantidad de partidas
        mask = own_alive.unsqueeze(-1).expand(N, 3, 6).clone() # (N, 3, 6) bool, inicialmente todas las acciones son válidas para guerreros vivos

        mask[:, :, :4] &= (own_cooldowns == 0) # si los cooldowns son mayores que 0, se deshabilitan las acciones de ataque (0-3)

        table = self.environment.target_mask_por_tipo_habilidad          # (num_types, POOL, 3)
        #Para cada partida, para cada guerrero, obtenemos la máscara de objetivos válidos según el tipo de habilidad del guerrero
        target_mask_pool = table[own_disposition]                        # (N, 3, POOL, 3)
        idx = own_instance_abilities.unsqueeze(-1).expand(-1, -1, -1, 3)  # (N, 3, 4, 3)
        #para cada geurrero de cada paritda, obtemeos de own_instance_abilities el indice de la habilidad que tiene, 
        # y con ese indice obtenemos de target_mask_pool la máscara de objetivos válidos para esa habilidad
        #target_mask_pool [:,:,idx,:]
        target_mask_full = target_mask_pool.gather(2, idx)                # (N, 3, 4, 3) PARTIDAS/POSICION/HABILIDAD/OBJETIVO
        
        #enemy_disposition es (N,3) con los tipos de guerreros enemigos (0 para muertos, 1..5 para vivos)
        #Despues (N,3,1,1) para poder compararlo con target_mask_full
        enemy_ocupado = (enemy_disposition > 0).unsqueeze(1).unsqueeze(1)
        #Si alguno es true, significa que hay al menos un objetivo válido para esa habilidad y guerrero
        hay_target_valido = (target_mask_full & enemy_ocupado).any(dim=-1)
        #no hay target válido si no hay ningún objetivo válido para esa habilidad y guerrero
        sin_target = ~hay_target_valido & target_mask_full.any(dim=-1)

        mask[:, :, :4] &= ~sin_target

        mask[:, 0, 5] = False # Slot 0 (front) → no puede moverse a la derecha (acción 5)
        mask[:, 2, 4] = False  # Slot 2 (back)  → no puede moverse a la izquierda (acción 4)

        return mask   # (N, 3, 6) bool

    def mask_turn(self, own_disposition, own_cooldowns, own_alive, enemy_disposition, own_instance_abilities, logits):
        """Wrapper de compatibilidad: calcula la máscara y la aplica a logits."""
        N = own_disposition.shape[0]
        mask = self.compute_action_mask(own_disposition, own_cooldowns, own_alive, enemy_disposition, own_instance_abilities)
        mask_flat = mask.reshape(N, 18) #Transforma de (N,3,6) a (N,18) para poder aplicarla a los logits de la red
        return logits.masked_fill(~mask_flat, float("-inf"))

    @staticmethod
    def _random_valid_action(masked_logits):
        valid = (masked_logits != float("-inf")).float()
        valid = torch.where(valid.sum(dim=1, keepdim=True) == 0, torch.ones_like(valid), valid)
        noise = torch.rand_like(valid)
        scored = noise * valid
        return torch.argmax(scored, dim=1)

    @staticmethod
    def _decode_ability_index(idx_0_5):
        return torch.where(idx_0_5 == 4, torch.full_like(idx_0_5, 5), torch.where(idx_0_5 == 5, torch.full_like(idx_0_5, 6), idx_0_5))

    @staticmethod
    def _environment_action_to_network(action):
        out = action.clone()
        out = torch.where(action == -1, torch.zeros_like(out), out)
        out = torch.where(action == 5, torch.full_like(out, 4), out)
        out = torch.where(action == 6, torch.full_like(out, 5), out)
        return out

    @staticmethod
    def _network_action_to_environment(action: int) -> int:
        if action == 4:
            return 5
        if action == 5:
            return 6
        return action

    def update_epsilon(self, n_games: int = 1) -> None:
        decay_sel = constants.EPSILON_SEL_DECAY ** n_games
        decay_turn = constants.EPSILON_TURN_DECAY ** n_games
        self.epsilon_sel = max(constants.EPSILON_SEL_MIN, self.epsilon_sel * decay_sel)
        self.epsilon_turn = max(constants.EPSILON_TURN_MIN, self.epsilon_turn * decay_turn)

    def reset_noise(self):
        self.selection_network.reset_noise()
        self.turn_network.reset_noise()
        self.target_selection_network.reset_noise()
        self.target_turn_network.reset_noise()

    def update_beta(self) -> None:
        self.replay_memory_sel.update_beta(self.replayed_selection)
        self.replay_memory_turn.update_beta(self.replayed_turn)

    def _optimize_step(self, loss, optimizer, network, target_network, replayed_counter_attr):
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), constants.GRAD_CLIP_MAX_NORM)
        optimizer.step()
        replayed = getattr(self, replayed_counter_attr)
        if replayed % constants.COPY_DQN == 0:
            target_network.load_state_dict(network.state_dict())

    def _multi_agent_double_dqn_target(self, batch, next_states, rewards, dones):
        with torch.no_grad():
            next_action_mask = batch.next_action_mask
            next_masks_flat = next_action_mask.reshape(-1, 18)

            next_qvalues_main = self.turn_network(next_states, action_mask=next_action_mask)
            next_qvalues_main = next_qvalues_main.masked_fill(~next_masks_flat, float("-inf"))

            next_q1 = next_qvalues_main[:, 0:6]
            next_q2 = next_qvalues_main[:, 6:12]
            next_q3 = next_qvalues_main[:, 12:18]
            next_a1 = next_q1.argmax(dim=1)
            next_a2 = next_q2.argmax(dim=1)
            next_a3 = next_q3.argmax(dim=1)
            next_actions = torch.stack([next_a1, next_a2 + 6, next_a3 + 12], dim=1)

            target_qvalues = self.target_turn_network(next_states, action_mask=next_action_mask)
            next_qvalues = target_qvalues.gather(1, next_actions)

            next_warrior_mask = batch.next_alive
            next_qvalues = (next_qvalues * next_warrior_mask.float()).sum(dim=1)

            return rewards + (constants.DISCOUNT_FACTOR ** constants.N_STEP) * next_qvalues * (~dones)

    def _network_specs(self):
        return [
            (self.selection_network, self.target_selection_network, self.optimizer_sel, self.replay_memory_sel, "epsilon_sel", "replayed_selection"),
            (self.turn_network, self.target_turn_network, self.optimizer_turn, self.replay_memory_turn, "epsilon_turn", "replayed_turn"),
        ]

    def save_model(self, path1: str, path2: str) -> None:
        for path, (net, target_net, opt, replay_memory, eps_attr, replayed_attr) in zip((path1, path2), self._network_specs()):
            torch.save({
                "dqn": net.state_dict(), "targetdqn": target_net.state_dict(), "optimizer": opt.state_dict(),
                "epsilon": getattr(self, eps_attr), "replayed": getattr(self, replayed_attr),
                "replay_memory": replay_memory.state_dict(), "elo": self.elo,
            }, path)

    def load_model(self, path1: str, path2: str) -> None:
        for path, (net, target_net, opt, replay_memory, eps_attr, replayed_attr) in zip((path1, path2), self._network_specs()):
            checkpoint = torch.load(path, weights_only=False)
            net.load_state_dict(checkpoint["dqn"])
            target_net.load_state_dict(checkpoint["targetdqn"])
            opt.load_state_dict(checkpoint["optimizer"])
            setattr(self, eps_attr, checkpoint["epsilon"])
            setattr(self, replayed_attr, checkpoint["replayed"])
            replay_memory.load_state_dict(checkpoint["replay_memory"])
            self.elo = float(checkpoint.get("elo", constants.ELO_INITIAL))

    def load_model_inference_only(self, path1: str, path2: str) -> None:
        for path, (net, target_net, opt, replay_memory, eps_attr, replayed_attr) in zip((path1, path2), self._network_specs()):
            checkpoint = torch.load(path, weights_only=False)
            net.load_state_dict(checkpoint["dqn"])
            setattr(self, eps_attr, checkpoint["epsilon"])
            self.elo = float(checkpoint.get("elo", constants.ELO_INITIAL))

    def save_model_inference_only(self, path1: str, path2: str) -> None:
        for path, (net, target_net, opt, replay_memory, eps_attr, replayed_attr) in zip((path1, path2), self._network_specs()):
            torch.save({"dqn": net.state_dict(), "epsilon": getattr(self, eps_attr), "elo": self.elo}, path)