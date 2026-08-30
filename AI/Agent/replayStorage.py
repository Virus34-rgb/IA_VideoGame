"""
Estructura de almacenamiento para el replay buffer (Structure of Arrays).
Permite guardar grandes cantidades de transiciones de forma eficiente.
"""
from collections import namedtuple
import torch

ReplayBatch = namedtuple(
    "ReplayBatch",
    [
        "states", "actions", "rewards", "next_states", "dones",
        "alive", "types", "cooldowns", "opp_types",
        "next_types", "next_alive", "next_cooldowns", "next_opp_types",
        "instance_abilities", "next_instance_abilities",
    ],
)


class ReplayStorage:
    """
    Almacena transiciones en tensores preasignados.
    Para el buffer de turno (is_turn_storage=True) se guardan campos adicionales.
    Para el buffer de selección, esos campos son None.
    """

    def __init__(self, capacity: int, state_dim: int, action_shape: tuple = (),
                 is_turn_storage: bool = False) -> None:
        self.is_turn_storage = is_turn_storage

        # Campos comunes
        self.states = torch.zeros(capacity, state_dim,dtype= torch.float16)
        self.actions = torch.zeros(capacity, *action_shape, dtype=torch.long)
        self.rewards = torch.zeros(capacity)
        self.next_states = torch.zeros(capacity, state_dim, dtype=torch.float16)
        self.dones = torch.zeros(capacity, dtype=torch.bool)

        # Campos específicos de turno
        if is_turn_storage:
            self.alive = torch.zeros(capacity, 3, dtype=torch.bool)
            self.types = torch.zeros(capacity, 3, dtype=torch.long)
            self.cooldowns = torch.zeros(capacity, 3, 4, dtype=torch.long)
            self.opp_types = torch.zeros(capacity, 3, dtype=torch.long)
            self.next_types = torch.zeros(capacity, 3, dtype=torch.long)
            self.next_alive = torch.zeros(capacity, 3, dtype=torch.bool)
            self.next_cooldowns = torch.zeros(capacity, 3, 4, dtype=torch.long)
            self.next_opp_types = torch.zeros(capacity, 3, dtype=torch.long)
            self.instance_abilities = torch.zeros(capacity, 3, 4, dtype=torch.long)
            self.next_instance_abilities = torch.zeros(capacity, 3, 4, dtype=torch.long)
        else:
            self.alive = None
            self.types = None
            self.cooldowns = None
            self.opp_types = None
            self.next_types = None
            self.next_alive = None
            self.next_cooldowns = None
            self.next_opp_types = None
            self.instance_abilities = None  
            self.next_instance_abilities = None  

    def get_batch(self, indices: torch.Tensor) -> ReplayBatch:
        if self.is_turn_storage:
            return ReplayBatch(
                self.states[indices], self.actions[indices], self.rewards[indices],
                self.next_states[indices], self.dones[indices],
                self.alive[indices], self.types[indices], self.cooldowns[indices], self.opp_types[indices],
                self.next_types[indices], self.next_alive[indices], self.next_cooldowns[indices], self.next_opp_types[indices],
                self.instance_abilities[indices], self.next_instance_abilities[indices],  
            )
        else:
            return ReplayBatch(
                self.states[indices], self.actions[indices], self.rewards[indices],
                self.next_states[indices], self.dones[indices],
                None, None, None, None, None, None, None, None, None, None,
            )

    def state_dict(self) -> dict:
        state = {
            "states": self.states, "actions": self.actions, "rewards": self.rewards,
            "next_states": self.next_states, "dones": self.dones,
        }
        if self.is_turn_storage:
            state.update({
                "alive": self.alive, "types": self.types, "cooldowns": self.cooldowns, "opp_types": self.opp_types,
                "next_types": self.next_types, "next_alive": self.next_alive,
                "next_cooldowns": self.next_cooldowns, "next_opp_types": self.next_opp_types,
                "instance_abilities": self.instance_abilities,                
                "next_instance_abilities": self.next_instance_abilities,      
            })
        return state

    def load_state_dict(self, state: dict) -> None:
        self.states.copy_(state["states"])
        self.actions.copy_(state["actions"])
        self.rewards.copy_(state["rewards"])
        self.next_states.copy_(state["next_states"])
        self.dones.copy_(state["dones"])
        if self.is_turn_storage:
            self.alive.copy_(state["alive"])
            self.types.copy_(state["types"])
            self.cooldowns.copy_(state["cooldowns"])
            self.opp_types.copy_(state["opp_types"])
            self.next_types.copy_(state["next_types"])
            self.next_alive.copy_(state["next_alive"])
            self.next_cooldowns.copy_(state["next_cooldowns"])
            self.next_opp_types.copy_(state["next_opp_types"])
            self.instance_abilities.copy_(state["instance_abilities"])             
            self.next_instance_abilities.copy_(state["next_instance_abilities"])    