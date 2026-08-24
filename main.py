# main.py
import os
import time

from AI.Agent.opponent_pool import OpponentPool
from AI.Agent.playerAI import PlayerAI
from AI.Agent.playerNoIA import PlayerNoAI
from AI.Environment.environment import Environment
from AI.Agent.trainer import Trainer
from AI.Logging.metrics_logger import MetricsLogger
from config import RunConfig
from training_step import TrainingStep
import constants


class Main:
    def __init__(self, config: RunConfig, steps: list):
        self.config = config
        self.steps = steps
        self.player1 = None
        self.environment = None
        self.logger = None
        self.opponent_pool = OpponentPool(self.config.path_opp_pool)

    def setup(self):
        os.makedirs(self.config.p1_path, exist_ok=True)
        os.makedirs(self.config.p2_path, exist_ok=True)
        self.player1 = PlayerAI()
        self.environment = Environment()
        self.logger = MetricsLogger(
            output_dir=self.config.base_path,
            run_name=f"v{self.config.version}",
        )
        self.logger.dump_config(constants, extra={
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
        self.logger.plot_progress(show=True)

    def _run_step(self, step: TrainingStep):
        print(f"\n{'=' * 65}\nSTEP: {step.name} ({step.action}, {step.episodes} partidas)\n{'-' * 65}")

        if step.player1_checkpoint:
            active_player1 = PlayerAI()
            sel_path, turn_path = step.player1_checkpoint
            if os.path.exists(sel_path) and os.path.exists(turn_path):
                active_player1.load_model(sel_path, turn_path)
        else:
            active_player1 = self.player1

        opponent = step.opponent_factory()
        if step.load_opponent_checkpoint and hasattr(opponent, "load_model"):
            sel_path, turn_path = step.load_opponent_checkpoint
            if os.path.exists(sel_path) and os.path.exists(turn_path):
                opponent.load_model(sel_path, turn_path)
        
        trainer = Trainer(
            active_player1, opponent, self.environment,self.opponent_pool,
            train_episodes=step.episodes if step.action == "train" else 0,
            eval_episodes=step.episodes if step.action == "evaluate" else 0,
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
            trainer.run_asymmetric(
                episodes=step.episodes,
                epsilon_turn=epsilon_turn,
                epsilon_sel=step.epsilon_sel,
                learn_p1=learn_p1,
                learn_p2=learn_p2,
                stats_path=stats_path,
            )

            if learn_p1:
                p1_paths = step.player1_checkpoint if step.player1_checkpoint else (self.config.path_p1_sel, self.config.path_p1_turn)
                trainer._save_if_supported(active_player1, *p1_paths)
            if learn_p2:
                p2_paths = step.load_opponent_checkpoint if step.load_opponent_checkpoint else (self.config.path_p2_sel, self.config.path_p2_turn)
                trainer._save_if_supported(opponent, *p2_paths)

        elapsed = time.time() - start
        print(f"{step.name} terminado en {self._format_time(elapsed)}")

    def _print_configuration(self):
        print("=" * 65)
        print("                         CASTLE GAME")
        print("=" * 65)
        print(f"Versión:  IA V{self.config.version}")
        for step in self.steps:
            print(f"  - {step.name}: {step.action}, {step.episodes} partidas")
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
# CONFIGURACIÓN 
# ==================================================================
VERSION = 1
TRAIN_EPISODES = 100000
EVAL_EPISODES = 10000

# Steps de entrenamiento/evaluación IA vs IA que siempre se ejecutan
RUN_SELF_PLAY = True # Si es False, se salta el step de entrenamiento self-play (IA vs IA con aprendizaje)
RUN_EVALUATION = True # Si es False, se salta el step de evaluación final (IA vs IA sin aprender, solo mide winrate/stats)

# Enfrentamiento contra humano: "none" | "ia1" | "ia2"
HUMAN_OPPONENT = "none"
HUMAN_EPISODES = 20
# Probabilidad de que la IA elija una acción aleatoria en vez de la mejor
# conocida durante las partidas contra el humano (exploración). Valores bajos
# (0.05-0.15) hacen que casi siempre juegue su mejor jugada pero sin quedarse
# ciega a corregir errores; 0.0 la volvería completamente determinista.
HUMAN_EPSILON = 0.1

# Jugar contra la IA entrenada (después del entrenamiento/evaluación)
PLAY_AGAINST_AI = False  # Cambiar a True para jugar algunas partidas contra la IA
PLAY_EPISODES = 20       # Número de partidas a jugar
PLAY_EPSILON = 0.0       # Exploración de la IA durante la partida (0.0 = determinista)

def build_steps(config: RunConfig):
    steps = []

    if RUN_SELF_PLAY:
        steps.append(TrainingStep(
            name="Entrenamiento self-play",
            action="train",
            episodes=config.train_episodes,
            opponent_factory=PlayerAI,
        ))

    if RUN_EVALUATION:
        steps.append(TrainingStep(
            name="Evaluación final",
            action="evaluate",
            episodes=config.eval_episodes,
            opponent_factory=PlayerAI,
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
            opponent_factory=PlayerNoAI,
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
            opponent_factory=PlayerNoAI,
            player1_checkpoint=(config.path_p1_sel, config.path_p1_turn),
            learn_p1=False,
            learn_p2=False,
            epsilon_turn=PLAY_EPSILON,
        ))

    return steps


if __name__ == "__main__":
    config = RunConfig(version=VERSION, train_episodes=TRAIN_EPISODES, eval_episodes=EVAL_EPISODES)
    steps = build_steps(config)
    Main(config, steps).run()