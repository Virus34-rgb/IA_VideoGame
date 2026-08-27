from torch import nn

from constants import USE_DUELING_DQN

class TurnNetwork(nn.Module):
    def __init__(self, input_size=58, output_size=18, num_slots=3, actions_per_slot=6):
        super().__init__()
        assert output_size == num_slots * actions_per_slot, (
            f"output_size ({output_size}) debe ser num_slots*actions_per_slot "
            f"({num_slots}*{actions_per_slot})"
        )
        self.num_slots = num_slots
        self.actions_per_slot = actions_per_slot
        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        if USE_DUELING_DQN:
            self.value = nn.Linear(32, num_slots)
            self.advantage = nn.Linear(32, output_size)
        else:
            self.value = nn.Linear(32, output_size)

    def forward(self, x, action_mask=None):
        trunk = self.shared(x)
        if USE_DUELING_DQN:
            val = self.value(trunk).view(-1, self.num_slots, 1)                          # (B, 3, 1)
            adv = self.advantage(trunk).view(-1, self.num_slots, self.actions_per_slot)  # (B, 3, 6)

            if action_mask is not None:
                mask = action_mask.float()  # (B, 3, 6), 1.0 = válida, 0.0 = inválida
                valid_count = mask.sum(dim=2, keepdim=True)          # (B, 3, 1)
                valid_count = valid_count.clamp(min=1.0)
                adv_masked_sum = (adv * mask).sum(dim=2, keepdim=True)  # (B, 3, 1)
                adv_mean = adv_masked_sum / valid_count                 # (B, 3, 1) media SOLO de válidas
            else:
                adv_mean = adv.mean(dim=2, keepdim=True)                # comportamiento original

            logits = val + (adv - adv_mean)
            logits = logits.view(-1, self.num_slots * self.actions_per_slot)  # (B, 18)
        else:
            logits = self.value(trunk)

        return logits