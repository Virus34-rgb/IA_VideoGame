"""
MainV: orquesta la ejecución de una única configuración (secuencia de
TrainingStep, gestión de la pool de oponentes, logging). Extraído de
mainV.py para que el punto de entrada del proyecto se limite a resolver
configuración y delegar aquí el trabajo real.
"""
import os
import pstats
import re
import shutil
import time
from typing import Callable, List, Optional

from AI.Agent.opponent_poolV import OpponentPoolV
from AI.Agent.playerAIV import PlayerAIV
from AI.Agent.playerNoIAV import PlayerNoAIV
from AI.Agent.player_rusher import PlayerRusherV
from AI.Environment.vectorizedEnvironment import VectorizedEnvironment
from AI.Agent.trainerV import TrainerV
from AI.Logging.metrics_logger import MetricsLogger
from config import RunConfig
from training_step import TrainingStep
import constants

# OJO: esta condición se evalúa en el momento en que ESTE módulo se importa
# (import-time), que ocurre ANTES de que mainV.py aplique los overrides de
# config.yaml con setattr (ver _apply_yaml_overrides en mainV.py). Es decir,
# usa el USE_GUI por defecto de constants.py, no el que pudiera sobreescribir
# el YAML -- comportamiento idéntico al mainV.py original, no se cambia aquí.
if constants.USE_GUI:
    from AI.Agent.playerGUI import PlayerGUIV

try:
    from AI.Logging import wandb_setup
except ImportError:
    wandb_setup = None
    print("Advertencia: wandb_setup no encontrado. Desactivando wandb.")


class MainV:
    """
    Orquesta una secuencia de TrainingStep sobre una única configuración
    (RunConfig). Para comparar varias configuraciones ver
    AI/Agent/orchestration/run_spec.py (RunSpec/run_comparison), y para
    repetir la misma configuración con varias seeds ver multi_seed_runner.py.
    """

    def __init__(
        self,
        config: RunConfig,
        steps: List[TrainingStep],
        N: int,
        player_class: Optional[Callable] = None,
        log_dir: Optional[str] = None,
    ) -> None:
        self.config = config
        self.steps = steps
        self.N = N
        self.player_class = player_class or PlayerAIV
        self.log_dir = log_dir or self.config.base_path

        self.player1: Optional[PlayerAIV] = None
        self.playerRusher: Optional[PlayerRusherV] = None
        self.environment: Optional[VectorizedEnvironment] = None
        self.logger: Optional[MetricsLogger] = None
        self.opponent_pool: OpponentPoolV = OpponentPoolV(self.config.path_opp_pool)

    # ------------------------------------------------------------
    # Configuración y ejecución principal
    # ------------------------------------------------------------

    def setup(self) -> None:
        self.run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        if constants.DELETE_DIRECTORIES:
            shutil.rmtree(self.config.p1_path, ignore_errors=True)
            shutil.rmtree(self.config.p2_path, ignore_errors=True)

        os.makedirs(self.config.p1_path, exist_ok=True)
        os.makedirs(self.config.p2_path, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.environment = VectorizedEnvironment(self.N)
        self.player1 = self.player_class(self.N, self.environment)
        self.playerRusher = PlayerRusherV(self.N, self.environment)

        clean_suffix = self.sanitize_filename(self.config.suffix) if self.config.suffix else ""
        run_nameA = f"v{self.config.version}_{clean_suffix}" if clean_suffix else f"v{self.config.version}"

        self.logger = MetricsLogger(output_dir=self.log_dir, run_name=run_nameA)
        self.logger.dump_config(
            constants,
            extra={
                "N": self.N,
                "steps": [
                    {"name": s.name, "action": s.action, "episodes": s.episodes}
                    for s in self.steps
                ],
            },
        )

        if constants.USE_WANDB and wandb_setup is not None:
            wandb_setup.init_wandb(
                project_name="castle-game-rl",
                run_name=run_nameA,
                config={
                    "N": self.N,
                    "learning_rate_selection": constants.SELECTION_LEARNING_RATE,
                    "learning_rate_turn": constants.TURN_LEARNING_RATE,
                    "discount_factor": constants.DISCOUNT_FACTOR,
                    "n_step": constants.N_STEP,
                    "epsilon_turn": constants.EPSILON_TURN,
                    "epsilon_sel": constants.EPSILON_SELECTION,
                    "batch_size": constants.BATCH_SIZE,
                    "use_dueling": constants.USE_DUELING_DQN,
                    "use_meta": constants.USE_META_GAME,
                    "max_turns": constants.MAX_TURNS,
                    "win_reward": constants.WIN_REWARD,
                    "turn_penalty_base": constants.TURN_PENALTY_BASE,
                    "turn_penalty_max": constants.TURN_PENALTY_MAX,
                    "shaping_weight": constants.REWARD_WEIGHTS["shaping_weight"],
                    "deaths_weight": constants.REWARD_WEIGHTS["deaths"],
                    "blocks_weight": constants.REWARD_WEIGHTS["blocks"],
                    "heal_weight": constants.REWARD_WEIGHTS["heal"],
                    "rusher_opponent_percentage": constants.RUSHER_OPPONENT_PERCENTAGE,
                }
            )

        self._print_configuration()

    def sanitize_filename(self, name: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

    def run(self) -> None:
        self.setup()
        for step in self.steps:
            self._run_step(step)
        self.logger.flush_loss_buffer()
        self._print_summary()
        self.logger.plot_progress(show=False)

        if constants.PROFILE_CPROFILE:
            self._print_profile_stats()

    # ------------------------------------------------------------
    # Ejecución de un paso individual
    # ------------------------------------------------------------

    def _run_step(self, step: TrainingStep) -> None:
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

        stats2_path_for_step = step.stats_path if step.stats_path else self.config.stats2_path

        trainer = TrainerV(
            active_player1,
            opponent,
            self.playerRusher,
            self.environment,
            self.opponent_pool,
            train_batches=step.episodes if step.action == "train" else 0,
            eval_batches=step.episodes if step.action == "evaluate" else 0,
            pathp1_1=self.config.path_p1_sel,
            pathp1_2=self.config.path_p1_turn,
            pathp2_1=self.config.path_p2_sel,
            pathp2_2=self.config.path_p2_turn,
            path_stats=self.config.stats_path,
            path_stats2=stats2_path_for_step,
            logger=self.logger,
            profile_cprofile=constants.PROFILE_CPROFILE,
            profile_torch=constants.PROFILE_TORCH,
            profile_torch_batches=constants.PROFILE_TORCH_BATCHES,
        )

        start_time = time.time()

        if step.learn_p1 is None and step.learn_p2 is None:
            # Modo automático: entrenar si action=="train", evaluar si "evaluate"
            if step.action == "train":
                trainer.train()
            else:
                trainer.evaluate(fixed_rusher_aggression=step.rusher_aggression)
        else:
            # Modo personalizado con flags explícitos (fine-tuning vs rusher,
            # fine-tuning vs humano, etc. -- ver step_builder.build_steps)
            default_learn = step.action == "train"
            learn_p1 = step.learn_p1 if step.learn_p1 is not None else default_learn
            learn_p2 = step.learn_p2 if step.learn_p2 is not None else default_learn
            epsilon_turn = step.epsilon_turn if step.epsilon_turn is not None else (
                0.5 if default_learn else 0.02
            )
            stats_path = step.stats_path if step.stats_path else (
                self.config.stats_path if step.action == "train" else self.config.stats2_path
            )

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
                fixed_rusher_aggression=step.rusher_aggression,
                fixed_rusher_aggression_min=step.rusher_aggression_min,
                fixed_rusher_aggression_max=step.rusher_aggression_max,
                profile_cprofile_output=None,
                profile_torch_this_step=False,
            )

            if learn_p1:
                p1_paths = step.player1_checkpoint if step.player1_checkpoint else (
                    self.config.path_p1_sel, self.config.path_p1_turn
                )
                trainer._save_if_supported(active_player1, *p1_paths)
            if learn_p2:
                p2_paths = step.load_opponent_checkpoint if step.load_opponent_checkpoint else (
                    self.config.path_p2_sel, self.config.path_p2_turn
                )
                trainer._save_if_supported(opponent, *p2_paths)

        elapsed = time.time() - start_time
        print(f"{step.name} terminado en {self._format_time(elapsed)}")

    def _build_opponent(self, step: TrainingStep) -> object:
        if step.opponent_factory is PlayerNoAIV:
            if constants.USE_GUI:
                return PlayerGUIV(self.environment)
            else:
                return PlayerNoAIV(self.environment)
        if step.opponent_factory is PlayerRusherV:
            return self.playerRusher
        return self.player_class(self.N, self.environment)

    # ------------------------------------------------------------
    # Impresión de configuración y resumen
    # ------------------------------------------------------------

    def _print_configuration(self) -> None:
        print("=" * 65)
        print("                    CASTLE GAME (VECTORIZADO)")
        print("=" * 65)
        print(f"Versión:  IA V{self.config.version}   |   N (partidas por lote): {self.N}")
        for step in self.steps:
            print(f"  - {step.name}: {step.action}, {step.episodes} lotes")
        print(f"Logs en:  {self.log_dir}")
        print(f"Wandb:    {'Activo (revisa el dashboard)' if constants.USE_WANDB else 'Desactivado'}")
        print("=" * 65)

    def _print_summary(self) -> None:
        print("=" * 65)
        print("                         FINALIZADO")
        print("=" * 65)
        print(f"Modelos guardados en: {self.config.base_path}")
        print(f"Logs (loss/progreso/config): {self.log_dir}")
        if constants.USE_WANDB:
            print(f"Wandb:    Revisa el dashboard para gráficas en tiempo real")

    @staticmethod
    def _format_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        if hours > 0:
            return f"{hours}h {minutes}m {secs:.2f}s"
        if minutes > 0:
            return f"{minutes}m {secs:.2f}s"
        return f"{secs:.2f}s"

    def _print_profile_stats(self):
        profile_path = constants.PROFILE_CPROFILE_OUTPUT
        if not os.path.exists(profile_path):
            print("⚠️ No se encontró archivo de perfil de cProfile.")
            return

        print("\n" + "=" * 70)
        print("                    ESTADÍSTICAS DE PERFIL (cProfile)")
        print("=" * 70)

        stats = pstats.Stats(profile_path)

        print("\n🔹 TOP 40 POR TIEMPO ACUMULADO (cumulative)")
        print("-" * 70)
        stats.sort_stats("cumulative").print_stats(40)

        print("\n🔹 TOP 40 POR TIEMPO PROPIO (tottime)")
        print("-" * 70)
        stats.sort_stats("tottime").print_stats(40)

        print("\n" + "=" * 70)
        print("✅ Análisis de perfil completado.")