import torch

import constants


class RewardCalculator:
    def __init__(self, REWARD_WEIGHTS, WIN_REWARD, DRAW_PENALTY,
                 TURN_PENALTY_BASE, TURN_PENALTY_MAX,
                 TURN_PENALTY_RAMP_START, TURN_PENALTY_RAMP_TURNS,
                 REWARD_SCALE, DISCOUNT_FACTOR,
                 WASTED_DEFENSE_WEIGHT_MAX_AGGRO,       # ← nuevo
                 STRATEGIC_MOVEMENT_WEIGHT_MAX_AGGRO):  # ← nuevo
        self.REWARD_WEIGHTS = REWARD_WEIGHTS
        self.WIN_REWARD = WIN_REWARD
        self.DRAW_PENALTY = DRAW_PENALTY
        self.TURN_PENALTY_BASE = TURN_PENALTY_BASE
        self.TURN_PENALTY_MAX = TURN_PENALTY_MAX
        self.TURN_PENALTY_RAMP_START = TURN_PENALTY_RAMP_START
        self.TURN_PENALTY_RAMP_TURNS = TURN_PENALTY_RAMP_TURNS
        self.REWARD_SCALE = REWARD_SCALE
        self.DISCOUNT_FACTOR = DISCOUNT_FACTOR
        self.WASTED_DEFENSE_WEIGHT_MAX_AGGRO = WASTED_DEFENSE_WEIGHT_MAX_AGGRO
        self.STRATEGIC_MOVEMENT_WEIGHT_MAX_AGGRO = STRATEGIC_MOVEMENT_WEIGHT_MAX_AGGRO

    def _turn_penalty(self, turn_number) -> torch.Tensor:
        turn = turn_number.float()
        exceso = (turn - self.TURN_PENALTY_RAMP_START).clamp(min=0.0)
        progresion = (exceso / self.TURN_PENALTY_RAMP_TURNS).clamp(max=1.0)
        return self.TURN_PENALTY_BASE + progresion * (self.TURN_PENALTY_MAX - self.TURN_PENALTY_BASE)

    def _interpolate_weight(self, base_weight: float, max_weight: float,
                            aggression: torch.Tensor) -> torch.Tensor:
        """Interpolación lineal por fila:
            aggression=0 → base_weight
            aggression=1 → max_weight
        `aggression` debe venir ya clamped a [0,1] (lo garantiza OpponentProfileTracker).
        """
        return base_weight + (max_weight - base_weight) * aggression

    def _reward(self, turn_number, extra_weighted=None, **components: torch.Tensor) -> torch.Tensor:
        """Suma ponderada de componentes + término extra pre-ponderado (ya por fila),
        menos turn penalty, dividido por REWARD_SCALE.
        """
        weighted = sum(self.REWARD_WEIGHTS[name] * value for name, value in components.items())
        if extra_weighted is not None:
            weighted = weighted + extra_weighted
        return (weighted - self._turn_penalty(turn_number)) / self.REWARD_SCALE

    def calculate_rewards(
        self, damage_p1, damage_p2, damage_avoided_p1, damage_avoided_p2,
        healed_p1, healed_p2, health_diff_before, health_diff_after, newDeaths_p1, newDeaths_p2,
        wasted_heal_p1, wasted_heal_p2, wasted_defense_p1, wasted_defense_p2,
        strategic_movement_p1, strategic_movement_p2,
        overkill_damage_p1, overkill_damage_p2, kill_confirmed_p1, kill_confirmed_p2,
        winner, turn_number,
        opp_aggression_p1=None, opp_aggression_p2=None,   # ← nuevos
    ):
        # --- Defaults neutros si algún caller no los pasa ---
        # Nota: 0.5 en la interpolación NO es idéntico al comportamiento actual
        # (−3.25 vs −5 para wasted_defense, +7 vs +5 para strategic_movement).
        # Para el test de regresión usar constantes iguales al extremo (ver guía).
        if opp_aggression_p1 is None:
            opp_aggression_p1 = torch.full_like(damage_p1, 0.5)
        if opp_aggression_p2 is None:
            opp_aggression_p2 = torch.full_like(damage_p1, 0.5)

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

        if constants.USE_PROFILE_CONDITIONED_REWARD:
            wd_w_p1 = self._interpolate_weight(
                self.REWARD_WEIGHTS["wasted_defense"],
                self.WASTED_DEFENSE_WEIGHT_MAX_AGGRO, opp_aggression_p1,
            )
            sm_w_p1 = self._interpolate_weight(
                self.REWARD_WEIGHTS["strategic_movement"],
                self.STRATEGIC_MOVEMENT_WEIGHT_MAX_AGGRO, opp_aggression_p1,
            )
            wd_w_p2 = self._interpolate_weight(
                self.REWARD_WEIGHTS["wasted_defense"],
                self.WASTED_DEFENSE_WEIGHT_MAX_AGGRO, opp_aggression_p2,
            )
            sm_w_p2 = self._interpolate_weight(
                self.REWARD_WEIGHTS["strategic_movement"],
                self.STRATEGIC_MOVEMENT_WEIGHT_MAX_AGGRO, opp_aggression_p2,
            )
        else:
            wd_w_p1 = torch.full_like(damage_p1, self.REWARD_WEIGHTS["wasted_defense"])
            sm_w_p1 = torch.full_like(damage_p1, self.REWARD_WEIGHTS["strategic_movement"])
            wd_w_p2 = torch.full_like(damage_p1, self.REWARD_WEIGHTS["wasted_defense"])
            sm_w_p2 = torch.full_like(damage_p1, self.REWARD_WEIGHTS["strategic_movement"])

        extra_p1 = wd_w_p1 * wasted_defense_p1 + sm_w_p1 * strategic_movement_p1
        extra_p2 = wd_w_p2 * wasted_defense_p2 + sm_w_p2 * strategic_movement_p2

        rewardP1 = self._reward(
            turn_number=turn_number,
            extra_weighted=extra_p1,
            damage=damage_p1 - damage_p2,
            deaths=newDeaths_p2 - newDeaths_p1,
            win=win_p1,
            blocks=damage_avoided_p1,
            heal=healed_p1,
            shaping_weight=shaping_term_p1,
            wasted_heal=wasted_heal_p1,
            kill_confirmed=kill_confirmed_p1,
            overkill_damage=overkill_damage_p1,
        )
        rewardP2 = self._reward(
            turn_number=turn_number,
            extra_weighted=extra_p2,
            damage=damage_p2 - damage_p1,
            deaths=newDeaths_p1 - newDeaths_p2,
            win=win_p2,
            blocks=damage_avoided_p2,
            heal=healed_p2,
            shaping_weight=shaping_term_p2,
            wasted_heal=wasted_heal_p2,
            kill_confirmed=kill_confirmed_p2,
            overkill_damage=overkill_damage_p2,
        )
        return rewardP1, rewardP2