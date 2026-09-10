"""
Lógica de entrenamiento (replay + optimización) para un agente PlayerAIV.
Extraído de PlayerAIV: replay_turn, replay_selection, _optimize_step,
_multi_agent_double_dqn_target, _loss_function, update_beta.

Diseño: replayed_selection/replayed_turn SÍ viven aquí como self.X propios,
porque son estado intrínseco de "cuánto ha entrenado este colaborador" -- a
diferencia de epsilon/elo (categoría B en PlayerAIV), estos contadores no
tienen ningún otro consumidor fuera de la lógica de replay/optimización y del
guardado de checkpoints (que los LEE via propiedad, no los posee).
"""
from typing import Optional
import torch
from torch import nn

import constants


class AgentTrainer:
    def __init__(self, selection_network, target_selection_network, optimizer_sel, replay_memory_sel,
                 turn_network, target_turn_network, optimizer_turn, replay_memory_turn):
        self.selection_network = selection_network
        self.target_selection_network = target_selection_network
        self.optimizer_sel = optimizer_sel
        self.replay_memory_sel = replay_memory_sel

        self.turn_network = turn_network
        self.target_turn_network = target_turn_network
        self.optimizer_turn = optimizer_turn
        self.replay_memory_turn = replay_memory_turn

        self.replayed_selection: int = 0
        self.replayed_turn: int = 0
        self._turn_offsets = torch.tensor([0, 6, 12])

    def replay_selection(self) -> Optional[float]:
        if len(self.replay_memory_sel) < constants.BATCH_SIZE:
            return None

        self.selection_network.reset_noise()
        self.target_selection_network.reset_noise()

        self.replayed_selection += 1
        batch, tree_indices, weights = self.replay_memory_sel.sample(constants.BATCH_SIZE)
        weights = torch.from_numpy(weights).float()

        states = batch.states.float()
        actions = batch.actions
        rewards = batch.rewards
        next_states = batch.next_states.float()
        dones = batch.dones

        qvalues = self.selection_network(states)
        q_selected = qvalues.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.selection_network(next_states).argmax(dim=1)
            next_qvalues = self.target_selection_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + constants.DISCOUNT_FACTOR * next_qvalues * (~dones)
            td_errors = torch.abs(q_selected - target)
            td_errors = torch.nan_to_num(td_errors, nan=1.0, posinf=10.0, neginf=10.0)

        loss = self._loss_function(q_selected, target, weights)
        self._optimize_step(loss, self.optimizer_sel, self.selection_network, self.target_selection_network, "replayed_selection")
        self.replay_memory_sel.update_priorities(tree_indices, td_errors.detach().cpu().numpy())
        return loss.item()

    def replay_turn(self) -> Optional[float]:
        if len(self.replay_memory_turn) < constants.BATCH_SIZE:
            return None

        self.turn_network.reset_noise()
        self.target_turn_network.reset_noise()

        self.replayed_turn += 1
        batch, tree_indices, weights = self.replay_memory_turn.sample(constants.BATCH_SIZE)
        weights = torch.from_numpy(weights).float()

        states = batch.states.float()
        warrior_mask = batch.alive
        next_states = batch.next_states.float()
        rewards = batch.rewards
        dones = batch.dones

        actions_b = self._environment_action_to_network(batch.actions)
        current_action_mask = batch.action_mask

        qvalues = self.turn_network(states, action_mask=current_action_mask)
        offsets = self._turn_offsets
        actions_global = actions_b + offsets
        q_selected = qvalues.gather(1, actions_global)
        q_selected = q_selected * warrior_mask.float()
        q_selected = q_selected.sum(dim=1)

        target = self._multi_agent_double_dqn_target(batch, next_states, rewards, dones)

        with torch.inference_mode():
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

    def _optimize_step(self, loss, optimizer, network, target_network, replayed_counter_attr):
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), constants.GRAD_CLIP_MAX_NORM)
        optimizer.step()
        network.clamp_sigma(constants.SIGMA_MIN)
        replayed = getattr(self, replayed_counter_attr)
        if replayed % constants.COPY_DQN == 0:
            if hasattr(network, '_orig_mod'):
                network = network._orig_mod
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

    @staticmethod
    def _environment_action_to_network(action):
        out = action.clone()
        out = torch.where(action == -1, torch.zeros_like(out), out)
        out = torch.where(action == 5, torch.full_like(out, 4), out)
        out = torch.where(action == 6, torch.full_like(out, 5), out)
        return out

    def update_beta(self) -> None:
        self.replay_memory_sel.update_beta(self.replayed_selection)
        self.replay_memory_turn.update_beta(self.replayed_turn)

    def mean_sigmas(self) -> dict:
        return {
            "sel": self.selection_network.mean_abs_sigma(),
            "turn": self.turn_network.mean_abs_sigma(),
        }