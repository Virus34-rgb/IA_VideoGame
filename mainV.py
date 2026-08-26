# mainV.py
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from AI.Agent.opponent_poolV import OpponentPoolV
from AI.Agent.playerAIV import PlayerAIV
from AI.Agent.playerNoIAV import PlayerNoAIV
from AI.Environment.vectorizedEnvironment import VectorizedEnvironment
from AI.Agent.trainerV import TrainerV
from AI.Logging.metrics_logger import MetricsLogger
from config import RunConfig
from training_step import TrainingStep
import constants


class MainV:
    def __init__(self, config: RunConfig, steps: list, N: int, player_class: Optional[Callable] = None):
        self.config = config
        self.steps = steps
        self.N = N
        # NUEVO: permite inyectar una clase de PlayerAIV distinta (para
        # comparar arquitecturas, no solo hiperparámetros) sin tocar el
        # resto de la clase.
        self.player_class = player_class or PlayerAIV
        self.player1 = None
        self.environment = None
        self.logger = None
        self.opponent_pool = OpponentPoolV(self.config.path_opp_pool)

    def setup(self):
        os.makedirs(self.config.p1_path, exist_ok=True)
        os.makedirs(self.config.p2_path, exist_ok=True)
        self.environment = VectorizedEnvironment(self.N)
        self.player1 = self.player_class(self.N, self.environment)
        self.logger = MetricsLogger(
            output_dir=self.config.base_path,
            run_name=f"v{self.config.version}",
        )
        self.logger.dump_config(constants, extra={
            "N": self.N,
            "steps": [
                {"name": s.name, "action": s.action, "episodes": s.episodes}
                for s in self.steps
            ]
        })
        self._print_configuration()

    def run(self):
        self.setup()
        for step in self.steps:
            self._run_step(step)
        self._print_summary()
        self.logger.plot_progress(show=False)

    def _run_step(self, step: TrainingStep):
        print(f"\n{'=' * 65}\nSTEP: {step.name} ({step.action}, {step.episodes} lotes de {self.N})\n{'-' * 65}")

        if step.opponent_factory is PlayerNoAIV and self.N != 1:
            raise ValueError(
                f"El step '{step.name}' usa PlayerNoAIV (jugador humano), que solo "
                f"admite N=1. MainV está configurado con N={self.N}."
            )

        if step.player1_checkpoint:
            active_player1 = self.player_class(self.N, self.environment)
            sel_path, turn_path = step.player1_checkpoint
            if os.path.exists(sel_path) and os.path.exists(turn_path):
                active_player1.load_model(sel_path, turn_path)
        else:
            active_player1 = self.player1

        opponent = self._build_opponent(step)
        if step.load_opponent_checkpoint and hasattr(opponent, "load_model"):
            sel_path, turn_path = step.load_opponent_checkpoint
            if os.path.exists(sel_path) and os.path.exists(turn_path):
                opponent.load_model(sel_path, turn_path)

        trainer = TrainerV(
            active_player1, opponent, self.environment, self.opponent_pool,
            train_batches=step.episodes if step.action == "train" else 0,
            eval_batches=step.episodes if step.action == "evaluate" else 0,
            pathp1_1=self.config.path_p1_sel, pathp1_2=self.config.path_p1_turn,
            pathp2_1=self.config.path_p2_sel, pathp2_2=self.config.path_p2_turn,
            path_stats=self.config.stats_path, path_stats2=self.config.stats2_path,
            logger=self.logger,
        )

        start = time.time()

        if step.learn_p1 is None and step.learn_p2 is None:
            if step.action == "train":
                trainer.train()
            else:
                trainer.evaluate()
        else:
            default_learn = step.action == "train"
            learn_p1 = step.learn_p1 if step.learn_p1 is not None else default_learn
            learn_p2 = step.learn_p2 if step.learn_p2 is not None else default_learn
            epsilon_turn = step.epsilon_turn if step.epsilon_turn is not None else (0.5 if default_learn else 0.02)
            stats_path = self.config.stats_path if step.action == "train" else self.config.stats2_path

            if step.action == "evaluate":
                self.environment.stats.reset()

            trainer._load_if_exists()
            trainer._run(
                batches=step.episodes,
                epsilon_turn=epsilon_turn,
                epsilon_sel=step.epsilon_sel,
                learn_p1=learn_p1,
                learn_p2=learn_p2,
                stats_path=stats_path,
                restore_epsilon=True,
            )

            if learn_p1:
                p1_paths = step.player1_checkpoint if step.player1_checkpoint else (self.config.path_p1_sel, self.config.path_p1_turn)
                trainer._save_if_supported(active_player1, *p1_paths)
            if learn_p2:
                p2_paths = step.load_opponent_checkpoint if step.load_opponent_checkpoint else (self.config.path_p2_sel, self.config.path_p2_turn)
                trainer._save_if_supported(opponent, *p2_paths)

        elapsed = time.time() - start
        print(f"{step.name} terminado en {self._format_time(elapsed)}")

    def _build_opponent(self, step: TrainingStep):
        if step.opponent_factory is PlayerNoAIV:
            return PlayerNoAIV()
        return self.player_class(self.N, self.environment)

    def _print_configuration(self):
        print("=" * 65)
        print("                    CASTLE GAME (VECTORIZADO)")
        print("=" * 65)
        print(f"Versión:  IA V{self.config.version}   |   N (partidas por lote): {self.N}")
        for step in self.steps:
            print(f"  - {step.name}: {step.action}, {step.episodes} lotes")
        print(f"Logs en:  {self.config.base_path}")
        print("=" * 65)

    def _print_summary(self):
        print("=" * 65)
        print("                         FINALIZADO")
        print("=" * 65)
        print(f"Modelos guardados en: {self.config.base_path}")
        print(f"Logs (loss/progreso/config): {self.config.base_path}")

    @staticmethod
    def _format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        if hours > 0:
            return f"{hours}h {minutes}m {seconds:.2f}s"
        if minutes > 0:
            return f"{minutes}m {seconds:.2f}s"
        return f"{seconds:.2f}s"


# ==================================================================
# SISTEMA DE COMPARACIÓN DE RUNS
# ==================================================================
@dataclass
class RunSpec:
    """
    Especifica una configuración completa de run: nombre (para identificar
    sus CSV/PNG), tamaño de lote, número de lotes de entrenamiento/evaluación,
    overrides puntuales de constants.py, y opcionalmente una clase de
    PlayerAIV distinta (para comparar arquitecturas, no solo hiperparámetros).
    """
    run_name: str
    N: int
    train_batches: int
    eval_batches: int
    constants_overrides: dict = field(default_factory=dict)
    player_class: Optional[Callable] = None


def run_single(config: RunConfig, run_spec: RunSpec):

    originales = {}
    for key, value in run_spec.constants_overrides.items():
        originales[key] = getattr(constants, key)
        setattr(constants, key, value)

    try:
        run_config = RunConfig(
            version=f"{config.version}_{run_spec.run_name}",
            train_episodes=run_spec.train_batches,
            eval_episodes=run_spec.eval_batches,
        )
        steps = build_steps(run_config)
        main = MainV(run_config, steps, N=run_spec.N, player_class=run_spec.player_class)
        main.run()
        return run_config.base_path, f"v{run_config.version}"
    finally:
        for key, value in originales.items():
            setattr(constants, key, value)


def run_comparison(config: RunConfig, run_specs: list):
    """
    Ejecuta todos los runs y al terminar genera una comparación
    conjunta aunque cada run tenga su propio directorio.
    """

    runs = []

    for spec in run_specs:
        print(f"\n{'#' * 65}\n"f" RUN: {spec.run_name}\n"f"{'#' * 65}")

        base_path, versioned_name = run_single(config, spec)

        runs.append((base_path, versioned_name))

    print(
        f"\n{'=' * 65}\n"
        f" COMPARACIÓN FINAL\n"
        f"{'=' * 65}"
    )

    comparison_path = MetricsLogger.compare_runs(
        runs=runs,
        labels=[spec.run_name for spec in run_specs],
        output_dir=config.base_path,
        show=True
    )

    print(f"\nComparación guardada en:")
    print(comparison_path)


# ==================================================================
# CONFIGURACIÓN
# ==================================================================
VERSION = 1
N_BATCH = 512
TRAIN_EPISODES = math.ceil(100000 / N_BATCH)
EVAL_EPISODES = math.ceil(10000 / N_BATCH)

RUN_SELF_PLAY = True
RUN_EVALUATION = True

HUMAN_OPPONENT = "none"
HUMAN_EPISODES = 20
HUMAN_EPSILON = 0.1

PLAY_AGAINST_AI = False
PLAY_EPISODES = 20
PLAY_EPSILON = 0.0

# NUEVO: activa el modo comparación en vez de un único run
RUN_COMPARISON = False

def build_steps(config: RunConfig):
    steps = []

    if RUN_SELF_PLAY:
        steps.append(TrainingStep(
            name="Entrenamiento self-play",
            action="train",
            episodes=config.train_episodes,
            opponent_factory=PlayerAIV,
        ))

    if RUN_EVALUATION:
        steps.append(TrainingStep(
            name="Evaluación final",
            action="evaluate",
            episodes=config.eval_episodes,
            opponent_factory=PlayerAIV,
            load_opponent_checkpoint=(config.path_p2_sel, config.path_p2_turn),
        ))

    if HUMAN_OPPONENT != "none":
        player1_checkpoint = None
        if HUMAN_OPPONENT == "ia2":
            player1_checkpoint = (config.path_p2_sel, config.path_p2_turn)
        elif HUMAN_OPPONENT != "ia1":
            raise ValueError(f"HUMAN_OPPONENT debe ser 'none', 'ia1' o 'ia2', no {HUMAN_OPPONENT!r}")

        steps.append(TrainingStep(
            name=f"Fine-tuning contra humano ({HUMAN_OPPONENT.upper()})",
            action="train",
            episodes=HUMAN_EPISODES,
            opponent_factory=PlayerNoAIV,
            player1_checkpoint=player1_checkpoint,
            learn_p1=True,
            learn_p2=False,
            epsilon_turn=HUMAN_EPSILON,
        ))

    if PLAY_AGAINST_AI:
        steps.append(TrainingStep(
            name="Jugar contra IA",
            action="evaluate",
            episodes=PLAY_EPISODES,
            opponent_factory=PlayerNoAIV,
            player1_checkpoint=(config.path_p1_sel, config.path_p1_turn),
            learn_p1=False,
            learn_p2=False,
            epsilon_turn=PLAY_EPSILON,
        ))

    return steps


if __name__ == "__main__":
    config = RunConfig(version=VERSION, train_episodes=TRAIN_EPISODES, eval_episodes=EVAL_EPISODES)

    if RUN_COMPARISON:
        run_specs = [
            RunSpec(run_name="baseline", N=N_BATCH, train_batches=TRAIN_EPISODES, eval_batches=EVAL_EPISODES),
            RunSpec(run_name="gamma_alto", N=N_BATCH, train_batches=TRAIN_EPISODES, eval_batches=EVAL_EPISODES,
                    constants_overrides={"DISCOUNT_FACTOR": 0.99}),
            RunSpec(run_name="replays_40", N=N_BATCH, train_batches=TRAIN_EPISODES, eval_batches=EVAL_EPISODES,
                    constants_overrides={"TURN_REPLAYS_PER_BATCH": 40}),
        ]
        run_comparison(config, run_specs)
    else:
        steps = build_steps(config)
        n_efectivo = 1 if HUMAN_OPPONENT != "none" or PLAY_AGAINST_AI else N_BATCH
        MainV(config, steps, N=n_efectivo).run()