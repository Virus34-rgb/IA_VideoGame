from collections import namedtuple
import torch

ReplayBatch = namedtuple("ReplayBatch",
    ["states", "actions", "rewards", "next_states", "dones",
     "alive", "next_types", "next_alive", "next_cooldowns", "next_opp_types"])


class ReplayStorage:
    def __init__(self, capacity, state_dim, action_shape=(), is_turn_storage=False):
        self.is_turn_storage = is_turn_storage

        self.states = torch.zeros(capacity, state_dim)
        self.actions = torch.zeros(capacity, *action_shape, dtype=torch.long)
        self.rewards = torch.zeros(capacity)
        self.next_states = torch.zeros(capacity, state_dim)
        self.dones = torch.zeros(capacity, dtype=torch.bool)

        # Exclusivos de la IA de turno (mask_turn / double dqn multi-agente).
        # Solo se reservan si is_turn_storage=True; en selección se quedan a None.
        if is_turn_storage:
            self.alive = torch.zeros(capacity, 3, dtype=torch.bool)
            self.next_types = torch.zeros(capacity, 3, dtype=torch.long)
            self.next_alive = torch.zeros(capacity, 3, dtype=torch.bool)
            self.next_cooldowns = torch.zeros(capacity, 3, 4, dtype=torch.bool)
            self.next_opp_types = torch.zeros(capacity, 3, dtype=torch.long)
        else:
            self.alive = None
            self.next_types = None
            self.next_alive = None
            self.next_cooldowns = None
            self.next_opp_types = None

    def get_batch(self, values):
        if self.is_turn_storage:
            return ReplayBatch(
                self.states[values], self.actions[values], self.rewards[values],
                self.next_states[values], self.dones[values],
                self.alive[values], self.next_types[values], self.next_alive[values],
                self.next_cooldowns[values], self.next_opp_types[values],
            )
        return ReplayBatch(
            self.states[values], self.actions[values], self.rewards[values],
            self.next_states[values], self.dones[values],
            None, None, None, None, None,
        )

    def state_dict(self):
        state = {
            "states": self.states, "actions": self.actions, "rewards": self.rewards,
            "next_states": self.next_states, "dones": self.dones,
        }
        if self.is_turn_storage:
            state.update({
                "alive": self.alive, "next_types": self.next_types,
                "next_alive": self.next_alive, "next_cooldowns": self.next_cooldowns,
                "next_opp_types": self.next_opp_types,
            })
        return state

    def load_state_dict(self, state):
        self.states.copy_(state["states"])
        self.actions.copy_(state["actions"])
        self.rewards.copy_(state["rewards"])
        self.next_states.copy_(state["next_states"])
        self.dones.copy_(state["dones"])
        if self.is_turn_storage:
            self.alive.copy_(state["alive"])
            self.next_types.copy_(state["next_types"])
            self.next_alive.copy_(state["next_alive"])
            self.next_cooldowns.copy_(state["next_cooldowns"])
            self.next_opp_types.copy_(state["next_opp_types"])