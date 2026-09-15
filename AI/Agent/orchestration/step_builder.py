"""
Construcción de la lista de TrainingStep según las flags de constants.py.
Extraído de mainV.py -- fábrica pura, sin estado propio.
"""
from typing import List

from AI.Agent.playerAIV import PlayerAIV
from AI.Agent.playerNoIAV import PlayerNoAIV
from AI.Agent.player_rusher import PlayerRusherV
from config import RunConfig
from training_step import TrainingStep
import constants


def build_steps(config: RunConfig) -> List[TrainingStep]:
    """Construye la lista de TrainingStep a ejecutar, según config y las
    flags RUN_*/HUMAN_*/PLAY_* de constants.py."""
    steps = []

    if constants.RUN_SELF_PLAY:
        steps.append(TrainingStep(
            name="Entrenamiento self-play",
            action="train",
            episodes=config.train_episodes,
            opponent_factory=PlayerAIV,
        ))

        if constants.RUN_RUSHER_FINETUNE:
            finetune_stats = (
                config.stats_rusher_finetune_low,
                config.stats_rusher_finetune_medium,
                config.stats_rusher_finetune_hight,
            )
            # Cada fase cubre una banda de agresividad del rusher distinta
            # (RUSHER_FINETUNE_PHASES = [(fraccion, agg_min, agg_max), ...]);
            # "fraction" reparte el presupuesto total de episodios entre fases.
            for phase_idx, (fraction, agg_min, agg_max) in enumerate(constants.RUSHER_FINETUNE_PHASES):
                steps.append(TrainingStep(
                    name=f"Fine-tuning vs Rusher (fase {phase_idx + 1}, agg {agg_min}-{agg_max})",
                    action="evaluate",
                    stats_path=finetune_stats[phase_idx],
                    episodes=max(1, int(constants.RUSHER_FINETUNE_EPISODES * fraction)),
                    opponent_factory=PlayerRusherV,
                    learn_p1=True,
                    learn_p2=False,
                    rusher_aggression_min=agg_min,
                    rusher_aggression_max=agg_max,
                ))

    if constants.RUN_EVALUATION:
        steps.append(TrainingStep(
            name="Evaluación final",
            action="evaluate",
            stats_path=config.stats2_path,
            episodes=config.eval_episodes,
            opponent_factory=PlayerAIV,
            load_opponent_checkpoint=(config.path_p2_sel, config.path_p2_turn),
        ))

    if constants.RUN_RUSHER_TESTS:
        steps.append(TrainingStep(
            name="Evaluación vs Rusher (aggression=0.0)",
            action="evaluate",
            stats_path=config.stats_rusher_aggr_low_path,
            episodes=constants.RUSHER_TEST_EPISODES,
            opponent_factory=PlayerRusherV,
            rusher_aggression=0.0,
        ))
        steps.append(TrainingStep(
            name="Evaluación vs Rusher (aggression=0.5)",
            action="evaluate",
            stats_path=config.stats_rusher_aggr_mid_path,
            episodes=constants.RUSHER_TEST_EPISODES,
            opponent_factory=PlayerRusherV,
            rusher_aggression=0.5,
        ))
        steps.append(TrainingStep(
            name="Evaluación vs Rusher (aggression=1.0)",
            action="evaluate",
            stats_path=config.stats_rusher_aggr_high_path,
            episodes=constants.RUSHER_TEST_EPISODES,
            opponent_factory=PlayerRusherV,
            rusher_aggression=1.0,
        ))

    if constants.HUMAN_OPPONENT != "none":
        player1_checkpoint = None
        if constants.HUMAN_OPPONENT == "ia2":
            player1_checkpoint = (config.path_p2_sel, config.path_p2_turn)
        elif constants.HUMAN_OPPONENT != "ia1":
            raise ValueError(
                f"HUMAN_OPPONENT debe ser 'none', 'ia1' o 'ia2', no {constants.HUMAN_OPPONENT!r}"
            )

        steps.append(TrainingStep(
            name=f"Fine-tuning contra humano ({constants.HUMAN_OPPONENT.upper()})",
            action="train",
            stats_path=config.stats_human,
            episodes=constants.HUMAN_EPISODES,
            opponent_factory=PlayerNoAIV,
            player1_checkpoint=player1_checkpoint,
            learn_p1=True,
            learn_p2=False,
            epsilon_turn=constants.HUMAN_EPSILON,
        ))

    if constants.PLAY_AGAINST_AI:
        steps.append(TrainingStep(
            name="Jugar contra IA",
            action="evaluate",
            episodes=constants.PLAY_EPISODES,
            opponent_factory=PlayerNoAIV,
            player1_checkpoint=(config.path_p1_sel, config.path_p1_turn),
            learn_p1=False,
            learn_p2=False,
            epsilon_turn=constants.PLAY_EPSILON,
        ))

    return steps