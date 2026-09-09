
import torch


class RewardCalculator:
    def __init__(self,REWARD_WEIGHTS,WIN_REWARD,DRAW_PENALTY,TURN_PENALTY_BASE,TURN_PENALTY_MAX,
                 TURN_PENALTY_RAMP_START,TURN_PENALTY_RAMP_TURNS,REWARD_SCALE,DISCOUNT_FACTOR):
        self.REWARD_WEIGHTS = REWARD_WEIGHTS
        self.WIN_REWARD = WIN_REWARD
        self.DRAW_PENALTY = DRAW_PENALTY
        self.TURN_PENALTY_BASE = TURN_PENALTY_BASE
        self.TURN_PENALTY_MAX = TURN_PENALTY_MAX
        self.TURN_PENALTY_RAMP_START = TURN_PENALTY_RAMP_START
        self.TURN_PENALTY_RAMP_TURNS = TURN_PENALTY_RAMP_TURNS
        self.REWARD_SCALE = REWARD_SCALE
        self.DISCOUNT_FACTOR = DISCOUNT_FACTOR
        
    def _turn_penalty(self,turn_number) -> torch.Tensor:
        turn = turn_number.float()
        exceso = (turn - self.TURN_PENALTY_RAMP_START).clamp(min=0.0)
        progresion = (exceso / self.TURN_PENALTY_RAMP_TURNS).clamp(max=1.0)
        return self.TURN_PENALTY_BASE + progresion * (self.TURN_PENALTY_MAX - self.TURN_PENALTY_BASE)

    def _reward(self,turn_number,**components: torch.Tensor) -> torch.Tensor:
        weighted = sum(self.REWARD_WEIGHTS[name] * value for name, value in components.items())
        return (weighted - self._turn_penalty(turn_number)) / self.REWARD_SCALE

    def calculate_rewards(
        self, damage_p1, damage_p2, damage_avoided_p1, damage_avoided_p2,
        healed_p1, healed_p2, health_diff_before, health_diff_after, newDeaths_p1, newDeaths_p2,
        wasted_heal_p1,wasted_heal_p2,wasted_defense_p1,wasted_defense_p2,strategic_movement_p1,strategic_movement_p2,
        overkill_damage_p1, overkill_damage_p2,kill_confirmed_p1, kill_confirmed_p2,winner,turn_number,
    ):
        

        gano_p1 = winner == 0
        gano_p2 = winner == 1
        empate = winner == 2
        win_p1 = torch.where(
            gano_p1, torch.full_like(damage_p1, self.WIN_REWARD),
            torch.where(gano_p2, torch.full_like(damage_p1, -self.WIN_REWARD),
                torch.where(empate, torch.full_like(damage_p1, -self.DRAW_PENALTY), torch.zeros_like(damage_p1))),
        )
        win_p2 = torch.where(
            gano_p2, torch.full_like(damage_p1, self.WIN_REWARD),
            torch.where(gano_p1, torch.full_like(damage_p1, -self.WIN_REWARD),
                torch.where(empate, torch.full_like(damage_p1, -self.DRAW_PENALTY), torch.zeros_like(damage_p1))),
        )

        shaping_term_p1 = self.DISCOUNT_FACTOR * health_diff_after - health_diff_before
        shaping_term_p2 = -shaping_term_p1

        rewardP1 = self._reward(turn_number=turn_number,
            damage=damage_p1 - damage_p2, deaths=newDeaths_p2 - newDeaths_p1, win=win_p1,
            blocks=damage_avoided_p1, heal=healed_p1, shaping_weight=shaping_term_p1,
            wasted_heal = wasted_heal_p1,wasted_defense = wasted_defense_p1,
            strategic_movement = strategic_movement_p1,kill_confirmed = kill_confirmed_p1, overkill_damage = overkill_damage_p1
        )
        rewardP2 = self._reward(turn_number=turn_number,
            damage=damage_p2 - damage_p1, deaths=newDeaths_p1 - newDeaths_p2, win=win_p2,
            blocks=damage_avoided_p2, heal=healed_p2, shaping_weight=shaping_term_p2,
            wasted_heal = wasted_heal_p2,wasted_defense = wasted_defense_p2,
            strategic_movement = strategic_movement_p2,kill_confirmed = kill_confirmed_p2, overkill_damage = overkill_damage_p2
        )
        return rewardP1, rewardP2