from constants import MAX_TURNS, WARRIOR_QUANTITY


class ObservationV:

    def __init__(self, pl_types, pl_alive, pl_speed_norm, pl_health_norm,
                 pl_cooldowns, opp_life, opp_disposition, turn):
        self.pl_types = pl_types
        self.pl_alive = pl_alive
        self.pl_speed_norm = pl_speed_norm
        self.pl_health_norm = pl_health_norm
        self.pl_cooldowns = pl_cooldowns
        self.opp_life = opp_life
        self.opp_disposition = opp_disposition
        self.turn = turn

    def normalize(self):
        observation = []
        for pos in range(3):
            if not self.pl_alive[pos]:
                observation.extend(self.id_to_one_hot(0, WARRIOR_QUANTITY))
                observation.append(0)
                observation.append(0)
                observation.extend([0, 0, 0, 0, 0, 0])
            else:
                observation.extend(self.id_to_one_hot(self.pl_types[pos], WARRIOR_QUANTITY))
                observation.append(self.pl_speed_norm[pos])
                observation.append(self.pl_health_norm[pos])
                observation.extend(self.normalize_abilities(self.pl_cooldowns[pos]))
        observation.extend(self.opp_life)
        for warrior_id in self.opp_disposition:
            observation.extend(self.id_to_one_hot(warrior_id, WARRIOR_QUANTITY))
        observation.append(min(self.turn / MAX_TURNS, 1.0))
        return observation

    @staticmethod
    def id_to_one_hot(warrior_id, warrior_quantity):
        result = [0] * warrior_quantity
        if warrior_id:
            result[warrior_id - 1] = 1
        return result

    @staticmethod
    def normalize_abilities(cooldowns_4):
        encoded = [0, 0, 0, 0, 0, 0]
        for position, en_cooldown in enumerate(cooldowns_4):
            encoded[position] = 0 if en_cooldown else 1
        encoded[4] = 0
        encoded[5] = 0
        return encoded

    @staticmethod
    def normalize_batch(pl_types, pl_alive, pl_speed_norm, pl_health_norm,
                         pl_cooldowns, opp_life, opp_disposition, turn_norm):
        import torch
        N = pl_types.shape[0]

        # one-hot de tipo por slot propio: (N, 3, WARRIOR_QUANTITY)
        idx = (pl_types - 1).clamp(min=0)
        one_hot_own = torch.nn.functional.one_hot(idx, num_classes=WARRIOR_QUANTITY).float()
        one_hot_own = one_hot_own * pl_alive.unsqueeze(-1).float()  # slots vacíos -> todo 0

        speed = torch.where(pl_alive, pl_speed_norm, torch.zeros_like(pl_speed_norm)).unsqueeze(-1)
        health = torch.where(pl_alive, pl_health_norm, torch.zeros_like(pl_health_norm)).unsqueeze(-1)

        cd_usable = (~pl_cooldowns).float()  # (N, 3, 4), 1 = usable
        cd_usable = torch.where(pl_alive.unsqueeze(-1), cd_usable, torch.zeros_like(cd_usable))
        extra_zeros = torch.zeros(N, 3, 2)

        propio = torch.cat([one_hot_own, speed, health, cd_usable, extra_zeros], dim=-1)  # (N, 3, WQ+2+6)
        propio = propio.flatten(start_dim=1)  # (N, 3*(WQ+8))

        idx_opp = (opp_disposition - 1).clamp(min=0)
        one_hot_opp = torch.nn.functional.one_hot(idx_opp, num_classes=WARRIOR_QUANTITY).float()
        one_hot_opp = one_hot_opp * (opp_disposition > 0).unsqueeze(-1).float()
        one_hot_opp = one_hot_opp.flatten(start_dim=1)  # (N, 3*WQ)

        return torch.cat([propio, opp_life, one_hot_opp, turn_norm.unsqueeze(-1)], dim=-1)