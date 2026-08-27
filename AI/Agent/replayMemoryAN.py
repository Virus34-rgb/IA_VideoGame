from collections import namedtuple
import random

import numpy

from AI.Agent.replayStorage import ReplayStorage
from AI.Agent.sumTree import SumTree
from constants import ALPHA, BETA_DECAY_RATE, BETA_END, BETA_START, PER_EPSILON


class ReplayMemoryAN(object):

    def __init__(self, capacity,state_dim):
        self.capacity = capacity
        self.memory = SumTree(capacity)
        self.memoryData = ReplayStorage(capacity, state_dim, action_shape=(3,), is_turn_storage=True)
        
        self.alpha = ALPHA
        self.beta = BETA_START
        self.beta_end = BETA_END

        self.epsilon = PER_EPSILON
        self.max_priority = 1.0
    
    def push_batch(self, states, actions, rewards, next_states, dones,
            alive, types, cooldowns, opp_types,                       # NUEVO: types, cooldowns, opp_types
            next_types, next_alive, next_cooldowns, next_opp_types):
        n = len(states)
        priorities = numpy.full(n, self.max_priority, dtype=numpy.float32)
        indices = self.memory.add_batch(priorities)
        self.memoryData.states[indices] = states
        self.memoryData.actions[indices] = actions
        self.memoryData.rewards[indices] = rewards
        self.memoryData.next_states[indices] = next_states
        self.memoryData.dones[indices] = dones
        self.memoryData.alive[indices] = alive
        self.memoryData.types[indices] = types                            
        self.memoryData.cooldowns[indices] = cooldowns                     
        self.memoryData.opp_types[indices] = opp_types                    
        self.memoryData.next_types[indices] = next_types
        self.memoryData.next_alive[indices] = next_alive
        self.memoryData.next_cooldowns[indices] = next_cooldowns
        self.memoryData.next_opp_types[indices] = next_opp_types

    def sample(self, batch_size):
        total = self.memory.total()
        segment = total / batch_size
        
        starts = numpy.arange(batch_size) * segment
        ends = (numpy.arange(batch_size) + 1) * segment
        
        values = numpy.random.uniform(starts, ends)
        values = numpy.minimum(values, total - 1e-6)
        
        tree_indices, priorities = self.memory.get_batch(values)

        probabilities = numpy.maximum(priorities, 1e-8) / total
        weights = (total * probabilities) ** (-self.beta)
        weights /= weights.max()
        
        data_indices = tree_indices - self.capacity + 1   # conversión árbol -> datos
        batch = self.memoryData.get_batch(data_indices)
        
        return batch, tree_indices, weights   # tree_indices se devuelve tal cual, para update_priorities

    def update_priorities(self, indices, priorities):
        priorities = numpy.abs(priorities) + self.epsilon
        priorities = priorities ** self.alpha
        self.memory.update_batch(indices, priorities)
        self.max_priority = max(self.max_priority,numpy.max(priorities))

    def update_beta(self, replayed_count):
        self.beta = BETA_END - (BETA_END - BETA_START) * (BETA_DECAY_RATE ** replayed_count)

    def __len__(self):
        return len(self.memory)

    def state_dict(self):
        return {
            "tree": self.memory.tree.copy(),
            "write": self.memory.write,
            "size": self.memory.size,
            "max_priority": self.max_priority,
            "memoryData": self.memoryData.state_dict(),   # objeto correcto
        }

    def load_state_dict(self, state):
        self.memory.tree = state["tree"].copy()
        self.memory.write = state["write"]
        self.memory.size = state["size"]
        self.max_priority = state["max_priority"]
        self.memoryData.load_state_dict(state["memoryData"]) 