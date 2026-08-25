from AI.Agent.observationV import ObservationV
from constants import WARRIOR_QUANTITY


class Choose_stateV:

    def __init__(self, pl_disposition_ids, pl_warriors, opp_initial_warrior, opp_initial_position):
        self.pl_disposition = pl_disposition_ids  # lista de 3 ints
        self.pl_warriors = pl_warriors             # ids disponibles, sin cambios
        self.opp_initial_warrior = opp_initial_warrior
        self.opp_initialPosition = opp_initial_position

    def encode_choose_state(self):
        observation = []
        for warrior_id in self.pl_disposition:
            observation.extend(ObservationV.id_to_one_hot(warrior_id, WARRIOR_QUANTITY))
        for id in self.pl_warriors:
            observation.extend(ObservationV.id_to_one_hot(id, WARRIOR_QUANTITY))
        observation.extend(ObservationV.id_to_one_hot(self.opp_initial_warrior, WARRIOR_QUANTITY))
        observation.append(self.opp_initialPosition / 3)
        return observation

    @staticmethod
    def encode_choose_state_batch(pl_disposition, pl_warriors_ids, opp_initial_warrior, opp_initial_position):
        import torch
        idx = (pl_disposition - 1).clamp(min=0)
        one_hot_disp = torch.nn.functional.one_hot(idx, num_classes=WARRIOR_QUANTITY).float()
        one_hot_disp = one_hot_disp * (pl_disposition > 0).unsqueeze(-1).float()
        one_hot_disp = one_hot_disp.flatten(start_dim=1)

        idx_w = (pl_warriors_ids - 1).clamp(min=0)
        one_hot_w = torch.nn.functional.one_hot(idx_w, num_classes=WARRIOR_QUANTITY).float()
        one_hot_w = one_hot_w * (pl_warriors_ids > 0).unsqueeze(-1).float()
        one_hot_w = one_hot_w.flatten(start_dim=1)

        idx_opp = (opp_initial_warrior - 1).clamp(min=0)
        one_hot_opp = torch.nn.functional.one_hot(idx_opp, num_classes=WARRIOR_QUANTITY).float()
        one_hot_opp = one_hot_opp * (opp_initial_warrior > 0).unsqueeze(-1).float()

        return torch.cat([one_hot_disp, one_hot_w, one_hot_opp,
                           (opp_initial_position / 3).unsqueeze(-1)], dim=-1)