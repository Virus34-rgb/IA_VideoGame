from collections import namedtuple
import random

from AI.Agent.sumTree import SumTree
from constants import ALPHA, BETA_DECAY_RATE, BETA_END, BETA_START, PER_EPSILON


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemoryAN(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = SumTree(capacity)

        self.alpha = ALPHA
        self.beta = BETA_START
        self.beta_end = BETA_END

        self.epsilon = PER_EPSILON
        self.max_priority = 1.0

    def push(self, *args):
        transition = Transition(*args)
        self.memory.add(self.max_priority, transition)

    def sample(self, batch_size):
        batch = []
        indices = []
        weights = []
        total = self.memory.total()
        segment = total / batch_size

        for i in range(batch_size):
            start = segment * i
            end = segment * (i + 1)

            value = random.uniform(start, end)
            value = min(value, total - 1e-6)

            index, priority, data = self.memory.get(value)

            batch.append(data)
            indices.append(index)
            probability = max(priority, 1e-8) / total
            weight = (total * probability) ** (-self.beta)
            weights.append(weight)

        maxw = max(weights)
        for pos, w in enumerate(weights):
            weights[pos] = w / maxw

        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            priority = abs(priority) + self.epsilon
            priority = priority ** self.alpha

            self.memory.update(index, priority)

            self.max_priority = max(self.max_priority, priority)

    def update_beta(self, replayed_count):
        self.beta = BETA_END - (BETA_END - BETA_START) * (BETA_DECAY_RATE ** replayed_count)

    def __len__(self):
        return len(self.memory)

    def state_dict(self):
        return {
            "tree": self.memory.tree.copy(),
            "data": self.memory.data.copy(),
            "write": self.memory.write,
            "max_priority": self.max_priority,
        }

    def load_state_dict(self, state):
        self.memory.tree = state["tree"].copy()
        self.memory.data = state["data"].copy()
        self.memory.write = state["write"]
        self.max_priority = state["max_priority"]