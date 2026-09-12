"""
Agente DQN para Castle Game -- fachada que compone AIPolicy + AgentTrainer +
CheckpointManager. Mantiene la MISMA interfaz pública que antes (selection(),
turn(), replay_turn(), save_model(), etc.) para que TrainerV/OpponentPoolV/etc.
no necesiten cambiar ni una línea.
"""
from typing import Optional, Any
import torch

import constants
from AI.Agent.turnNetwork import TurnNetwork
from AI.Agent.selectionNetwork import SelectionNetwork
from AI.Agent.replayMemoryAN import ReplayMemoryAN
from AI.Agent.replayMemoryPM import ReplayMemoryPM
from AI.Agent.training.agent_trainer import AgentTrainer
from AI.Agent.training.checkpoint_manager import CheckpointManager
from AI.Agent.training.ai_policy import AIPolicy


class PlayerAIV:
    def __init__(self, N: int, environment: Any, use_replay: bool = True) -> None:
        self.N: int = N
        self.environment: Any = environment
        self.name: str = "DqnPlayerV"

        selection_state_dim = constants.get_selection_state_dim()

        # Las redes se construyen UNA vez aquí -- las mismas instancias se
        # pasan a los 3 colaboradores. Nadie más crea una TurnNetwork/SelectionNetwork nueva.
        self.selection_network = SelectionNetwork(sigma_init=constants.NOISY_SIGMA_INIT, input_size=selection_state_dim)
        self.target_selection_network = SelectionNetwork(sigma_init=constants.NOISY_SIGMA_INIT, input_size=selection_state_dim)
        self.target_selection_network.load_state_dict(self.selection_network.state_dict())
        self.optimizer_sel = torch.optim.Adam(self.selection_network.parameters(), lr=constants.SELECTION_LEARNING_RATE, foreach=True)

        self.turn_network = TurnNetwork(sigma_init=constants.NOISY_SIGMA_INIT)
        self.target_turn_network = TurnNetwork(sigma_init=constants.NOISY_SIGMA_INIT)
        self.target_turn_network.load_state_dict(self.turn_network.state_dict())
        self.optimizer_turn = torch.optim.Adam(self.turn_network.parameters(), lr=constants.TURN_LEARNING_RATE, foreach=True)

        if constants.USE_TORCH_COMPILE:
            self.turn_network = torch.compile(self.turn_network, mode="reduce-overhead", fullgraph=False)
            self.selection_network = torch.compile(self.selection_network, mode="reduce-overhead", fullgraph=False)

        if use_replay:
            self.replay_memory_sel = ReplayMemoryPM(constants.SELECTION_REPLAY_DATA, state_dim=selection_state_dim)
            self.replay_memory_turn = ReplayMemoryAN(constants.TURN_REPLAY_DATA, state_dim=constants.TURN_STATE_DIM)
        else:
            self.replay_memory_sel = None
            self.replay_memory_turn = None

        # Colaboradores: reciben las MISMAS instancias construidas arriba.
        self._policy = AIPolicy(N, environment, self.selection_network, self.turn_network)
        self._trainer = AgentTrainer(
            self.selection_network, self.target_selection_network, self.optimizer_sel, self.replay_memory_sel,
            self.turn_network, self.target_turn_network, self.optimizer_turn, self.replay_memory_turn,
        )
        self._checkpoint = CheckpointManager(
            self.selection_network, self.target_selection_network, self.optimizer_sel, self.replay_memory_sel,
            self.turn_network, self.target_turn_network, self.optimizer_turn, self.replay_memory_turn,
        )

        self.elo = constants.ELO_INITIAL  # Categoría B: propiedad del agente, no de ningún colaborador.

    # ---- Propiedades de paso (para que el resto del proyecto siga leyendo player1.epsilon_sel, etc. sin cambios) ----
    @property
    def epsilon_sel(self): return self._policy.epsilon_sel
    @epsilon_sel.setter
    def epsilon_sel(self, v): self._policy.epsilon_sel = v

    @property
    def epsilon_turn(self): return self._policy.epsilon_turn
    @epsilon_turn.setter
    def epsilon_turn(self, v): self._policy.epsilon_turn = v

    @property
    def replayed_selection(self): return self._trainer.replayed_selection
    @property
    def replayed_turn(self): return self._trainer.replayed_turn

    # ---- Delegación de la interfaz pública, sin cambios de firma ----
    def remember_selection_batch(self, c_state, action, reward, next_c_state, done) -> None:
        self.replay_memory_sel.push_batch(c_state, action, reward, next_c_state, done)

    def remember_turn_batch(self, observation, action, reward, next_observation, done,
                             alive, types, cooldowns, opp_types, next_types, next_alive, next_cooldowns, next_opp_types,
                             instance_abilities, next_instance_abilities, action_mask, next_action_mask) -> None:
        self.replay_memory_turn.push_batch(
            observation, action, reward, next_observation, done,
            alive, types, cooldowns, opp_types, next_types, next_alive, next_cooldowns, next_opp_types,
            instance_abilities, next_instance_abilities, action_mask, next_action_mask,
        )

    def replay_selection(self) -> Optional[float]:
        return self._trainer.replay_selection()

    def replay_turn(self) -> Optional[float]:
        return self._trainer.replay_turn()

    def selection(self, *args, **kwargs):
        return self._policy.selection(*args, **kwargs)

    def turn(self, *args, **kwargs):
        return self._policy.turn(*args, **kwargs)

    def turn_with_mask(self, *args, **kwargs):
        return self._policy.turn_with_mask(*args, **kwargs)

    def update_epsilon(self, n_games: int = 1) -> None:
        self._policy.update_epsilon(n_games)

    def reset_noise(self):
        self._policy.reset_noise()
        self._trainer.target_selection_network.reset_noise()
        self._trainer.target_turn_network.reset_noise()

    def mean_sigmas(self) -> dict:
        return self._trainer.mean_sigmas()

    def update_beta(self) -> None:
        self._trainer.update_beta()

    def save_model(self, path1: str, path2: str) -> None:
        self._checkpoint.save_model(
            path1, path2,
            epsilons=(self.epsilon_sel, self.epsilon_turn),
            replayed_counts=(self._trainer.replayed_selection, self._trainer.replayed_turn),
            elo=self.elo,
        )

    def load_model(self, path1: str, path2: str) -> None:
        result = self._checkpoint.load_model(path1, path2)
        self.epsilon_sel, self.epsilon_turn = result["epsilons"]
        self._trainer.replayed_selection, self._trainer.replayed_turn = result["replayed_counts"]
        self.elo = result["elo"]

    def load_model_inference_only(self, path1: str, path2: str) -> None:
        result = self._checkpoint.load_model_inference_only(path1, path2)
        self.epsilon_sel, self.epsilon_turn = result["epsilons"]
        self.elo = result["elo"]

    def save_model_inference_only(self, path1: str, path2: str) -> None:
        self._checkpoint.save_model_inference_only(
            path1, path2, epsilons=(self.epsilon_sel, self.epsilon_turn), elo=self.elo,
        )
        
    def priority_stats(self) -> dict:
        """Devuelve percentiles de prioridad de los buffers de turno y selección."""
        out = {}
        if self.replay_memory_turn is not None:
            out["turn"] = self.replay_memory_turn.priority_stats()
        if self.replay_memory_sel is not None:
            out["sel"] = self.replay_memory_sel.priority_stats()
        return out