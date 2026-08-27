from collections import deque, namedtuple
import torch


BufferBatch = namedtuple(
    "BufferBatch",
    [
        "states",
        "actions",
        "rewards",
        "next_states",
        "dones",
        "alive",
        "types",
        "cooldowns",
        "opp_types",
        "next_types",
        "next_alive",
        "next_cooldowns",
        "next_opp_types",
    ],
)


class NStepBuffer:
    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma
        self.gamma_powers = torch.tensor([gamma ** i for i in range(n_step)]).unsqueeze(1)  # (n_step, 1)

        self.states = deque()
        self.actions = deque()
        self.rewards = deque()
        self.dones = deque()

        self.alive = deque()
        self.types = deque()
        self.cooldowns = deque()
        self.opp_types = deque()


    def push(self, states, actions, rewards, dones, alive, types, cooldowns, opp_types):
        self.states.append(states)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.dones.append(dones)

        self.alive.append(alive)
        self.types.append(types)
        self.cooldowns.append(cooldowns)
        self.opp_types.append(opp_types)

        if len(self.states) < self.n_step + 1:
            return None

        state_0 = self.states.popleft()
        action_0 = self.actions.popleft()
        alive_0 = self.alive.popleft()
        types_0 = self.types.popleft()
        cooldowns_0 = self.cooldowns.popleft()
        opp_types_0 = self.opp_types.popleft()
        
        rewards_window = [self.rewards[i] for i in range(self.n_step)]

        done_flag = self.dones[self.n_step - 1]

        self.rewards.popleft()
        self.dones.popleft()

        rewards_stack = torch.stack(rewards_window, dim=0)  # (n_step, N)
        n_step_reward = (rewards_stack * self.gamma_powers).sum(dim=0)  # (N,)

        next_state = self.states[-1]
        next_types = self.types[-1]
        next_alive = self.alive[-1]
        next_cooldowns = self.cooldowns[-1]
        next_opp_types = self.opp_types[-1]

        return BufferBatch(
            states=state_0, actions=action_0, rewards=n_step_reward,
            next_states=next_state, dones=done_flag,
            alive=alive_0, types=types_0, cooldowns=cooldowns_0, opp_types=opp_types_0,
            next_types=next_types, next_alive=next_alive,
            next_cooldowns=next_cooldowns, next_opp_types=next_opp_types,
        )

    def flush(self):
        while self.states:
            state_0 = self.states.popleft()
            action_0 = self.actions.popleft()
            alive_0 = self.alive.popleft()
            types_0 = self.types.popleft()
            cooldowns_0 = self.cooldowns.popleft()
            opp_types_0 = self.opp_types.popleft()

            # Igual que en push(): la ventana se lee ANTES del pop, tomando
            # TODO lo que quede en self.rewards (entre 1 y n_step elementos,
            # según cuántos turnos faltaban para madurar solo).
            rewards_window = list(self.rewards)
            num_steps = len(rewards_window)

            self.rewards.popleft()
            self.dones.popleft()

            rewards_stack = torch.stack(rewards_window, dim=0)
            discounts = self.gamma_powers[:num_steps]
            n_step_reward = (rewards_stack * discounts).sum(dim=0)

            if self.states:
                next_state = self.states[-1]
                next_types = self.types[-1]
                next_alive = self.alive[-1]
                next_cooldowns = self.cooldowns[-1]
                next_opp_types = self.opp_types[-1]
            else:
                next_state = state_0
                next_types = types_0
                next_alive = alive_0
                next_cooldowns = cooldowns_0
                next_opp_types = opp_types_0

            done = torch.ones_like(alive_0[:, 0], dtype=torch.bool)

            yield BufferBatch(
                states=state_0, actions=action_0, rewards=n_step_reward,
                next_states=next_state, dones=done,
                alive=alive_0, types=types_0, cooldowns=cooldowns_0, opp_types=opp_types_0,
                next_types=next_types, next_alive=next_alive,
                next_cooldowns=next_cooldowns, next_opp_types=next_opp_types,
            )

    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()

        self.alive.clear()
        self.types.clear()
        self.cooldowns.clear()
        self.opp_types.clear()

    def __len__(self):
        return len(self.states)