"""
Punto de entrada principal del proyecto Castle Game.

Orquesta la ejecución de pasos de entrenamiento/evaluación,
gestión de la pool de oponentes, y comparación de runs con diferentes
hiperparámetros (usando RunSpec).
"""
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, List

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
    """
    Clase principal que gestiona la ejecución del entrenamiento y evaluación.

    Permite definir una secuencia de pasos (TrainingStep) y ejecutarlos,
    con soporte para comparación de runs mediante RunSpec.
    """

    def __init__(
        self,
        config: RunConfig,
        steps: List[TrainingStep],
        N: int,
        player_class: Optional[Callable] = None,
        log_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            config: Configuración del run (versión, rutas, etc.).
            steps: Lista de pasos a ejecutar (entrenamiento, evaluación, etc.).
            N: Número de partidas paralelas.
            player_class: Clase del jugador (por defecto PlayerAIV).
            log_dir: Directorio para logs (si es None, usa config.base_path).
        """
        self.config = config
        self.steps = steps
        self.N = N
        self.player_class = player_class or PlayerAIV
        self.log_dir = log_dir or self.config.base_path

        self.player1: Optional[PlayerAIV] = None
        self.environment: Optional[VectorizedEnvironment] = None
        self.logger: Optional[MetricsLogger] = None
        self.opponent_pool: OpponentPoolV = OpponentPoolV(self.config.path_opp_pool)

    # ------------------------------------------------------------
    # Configuración y ejecución principal
    # ------------------------------------------------------------

    def setup(self) -> None:
        """
        Prepara el entorno, el jugador, el logger y la pool de oponentes.
        """
        # Limpiar directorios antiguos si está configurado
        if constants.DELETE_DIRECTORIES:
            shutil.rmtree(self.config.p2_path, ignore_errors=True)
            shutil.rmtree(self.config.p2_path, ignore_errors=True)  # Nota: podría ser p1_path? Revisar.

        # Crear directorios necesarios
        os.makedirs(self.config.p1_path, exist_ok=True)
        os.makedirs(self.config.p2_path, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Inicializar entorno y jugador
        self.environment = VectorizedEnvironment(self.N)
        self.player1 = self.player_class(self.N, self.environment)

        # Inicializar logger
        self.logger = MetricsLogger(
            output_dir=self.log_dir,
            run_name=f"v{self.config.version}",
        )
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
        self._print_configuration()

    def run(self) -> None:
        """Ejecuta todos los pasos definidos y genera gráficos finales."""
        self.setup()
        for step in self.steps:
            self._run_step(step)
        self._print_summary()
        self.logger.plot_progress(show=False)

    # ------------------------------------------------------------
    # Ejecución de un paso individual
    # ------------------------------------------------------------

    def _run_step(self, step: TrainingStep) -> None:
        """
        Ejecuta un paso (entrenamiento o evaluación) según la configuración.

        Args:
            step: El TrainingStep a ejecutar.
        """
        print(f"\n{'=' * 65}\nSTEP: {step.name} ({step.action}, {step.episodes} lotes de {self.N})\n{'-' * 65}")

        # Validación: humano solo con N=1
        if step.opponent_factory is PlayerNoAIV and self.N != 1:
            raise ValueError(
                f"El step '{step.name}' usa PlayerNoAIV (jugador humano), que solo "
                f"admite N=1. MainV está configurado con N={self.N}."
            )

        # Preparar jugador activo (P1) - puede cargar un checkpoint si se especifica
        if step.player1_checkpoint:
            active_player1 = self.player_class(self.N, self.environment)
            sel_path, turn_path = step.player1_checkpoint
            if os.path.exists(sel_path) and os.path.exists(turn_path):
                active_player1.load_model(sel_path, turn_path)
        else:
            active_player1 = self.player1

        # Preparar oponente (P2)
        opponent = self._build_opponent(step)
        if step.load_opponent_checkpoint and hasattr(opponent, "load_model"):
            sel_path, turn_path = step.load_opponent_checkpoint
            if os.path.exists(sel_path) and os.path.exists(turn_path):
                opponent.load_model(sel_path, turn_path)

        # Crear entrenador
        trainer = TrainerV(
            active_player1,
            opponent,
            self.environment,
            self.opponent_pool,
            train_batches=step.episodes if step.action == "train" else 0,
            eval_batches=step.episodes if step.action == "evaluate" else 0,
            pathp1_1=self.config.path_p1_sel,
            pathp1_2=self.config.path_p1_turn,
            pathp2_1=self.config.path_p2_sel,
            pathp2_2=self.config.path_p2_turn,
            path_stats=self.config.stats_path,
            path_stats2=self.config.stats2_path,
            logger=self.logger,
        )

        start_time = time.time()

        # Determinar parámetros de ejecución
        if step.learn_p1 is None and step.learn_p2 is None:
            # Modo automático: entrenar si action=="train", evaluar si "evaluate"
            if step.action == "train":
                trainer.train()
            else:
                trainer.evaluate()
        else:
            # Modo personalizado con flags explícitos
            default_learn = step.action == "train"
            learn_p1 = step.learn_p1 if step.learn_p1 is not None else default_learn
            learn_p2 = step.learn_p2 if step.learn_p2 is not None else default_learn
            epsilon_turn = step.epsilon_turn if step.epsilon_turn is not None else (
                0.5 if default_learn else 0.02
            )
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

            # Guardar modelos si se aprendió
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
        """
        Construye el oponente según la factoría especificada en el paso.

        Args:
            step: El TrainingStep actual.

        Returns:
            Instancia del oponente (PlayerAIV o PlayerNoAIV).
        """
        if step.opponent_factory is PlayerNoAIV:
            return PlayerNoAIV()
        return self.player_class(self.N, self.environment)

    # ------------------------------------------------------------
    # Impresión de configuración y resumen
    # ------------------------------------------------------------

    def _print_configuration(self) -> None:
        """Muestra la configuración del run por consola."""
        print("=" * 65)
        print("                    CASTLE GAME (VECTORIZADO)")
        print("=" * 65)
        print(f"Versión:  IA V{self.config.version}   |   N (partidas por lote): {self.N}")
        for step in self.steps:
            print(f"  - {step.name}: {step.action}, {step.episodes} lotes")
        print(f"Logs en:  {self.log_dir}")
        print("=" * 65)

    def _print_summary(self) -> None:
        """Muestra el resumen final del run."""
        print("=" * 65)
        print("                         FINALIZADO")
        print("=" * 65)
        print(f"Modelos guardados en: {self.config.base_path}")
        print(f"Logs (loss/progreso/config): {self.log_dir}")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Formatea un tiempo en segundos a formato legible."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        if hours > 0:
            return f"{hours}h {minutes}m {secs:.2f}s"
        if minutes > 0:
            return f"{minutes}m {secs:.2f}s"
        return f"{secs:.2f}s"


# ================================================================
# SISTEMA DE COMPARACIÓN DE RUNS (RunSpec)
# ================================================================

@dataclass
class RunSpec:
    """
    Especificación de un run para comparación.

    Permite sobrescribir constantes y usar una clase de jugador diferente.
    """
    run_name: str
    N: int
    train_batches: int
    eval_batches: int
    constants_overrides: dict = field(default_factory=dict)
    player_class: Optional[Callable] = None


def run_single(
    config: RunConfig,
    run_spec: RunSpec,
    shared_log_dir: Optional[str] = None,
) -> tuple[str, str]:
    """
    Ejecuta un único run con la configuración y especificaciones dadas.

    Args:
        config: Configuración base del run.
        run_spec: Especificaciones particulares (nombre, N, overrides, etc.).
        shared_log_dir: Directorio compartido para logs (si es None, usa el de config).

    Returns:
        Tuple (log_dir, version_name) del run ejecutado.
    """
    # Guardar valores originales de constantes para restaurarlos después
    original_values = {}
    for key, value in run_spec.constants_overrides.items():
        original_values[key] = getattr(constants, key)
        setattr(constants, key, value)

    try:
        # Crear configuración específica para este run
        run_config = RunConfig(
            version=f"{config.version}_{run_spec.run_name}",
            train_episodes=run_spec.train_batches,
            eval_episodes=run_spec.eval_batches,
        )
        steps = build_steps(run_config)

        main = MainV(
            run_config,
            steps,
            N=run_spec.N,
            player_class=run_spec.player_class,
            log_dir=shared_log_dir,
        )
        main.run()
        return main.log_dir, f"v{run_config.version}"
    finally:
        # Restaurar constantes originales
        for key, value in original_values.items():
            setattr(constants, key, value)


def run_comparison(config: RunConfig, run_specs: List[RunSpec]) -> None:
    """
    Ejecuta una comparación de varios runs y genera gráficos comparativos.

    Args:
        config: Configuración base (usada para la carpeta compartida).
        run_specs: Lista de especificaciones de runs a comparar.
    """
    shared_log_dir = config.base_path
    run_names = []

    for spec in run_specs:
        print(f"\n{'#' * 65}\n RUN: {spec.run_name}\n{'#' * 65}")
        _, versioned_name = run_single(config, spec, shared_log_dir=shared_log_dir)
        run_names.append(versioned_name)

    # Generar gráfico comparativo
    MetricsLogger.compare_runs(
        shared_log_dir,
        run_names,
        labels=[s.run_name for s in run_specs],
        show=False
    )


# ================================================================
# CONFIGURACIÓN DEL RUN PRINCIPAL
# ================================================================

# Parámetros globales del run
VERSION = 4
N_BATCH = 2048
TRAIN_EPISODES = math.ceil(100_000 / N_BATCH)
EVAL_EPISODES = math.ceil(10_000 / N_BATCH)

# Flags para activar/desactivar partes del pipeline
RUN_SELF_PLAY = True
RUN_EVALUATION = True

HUMAN_OPPONENT = "none"  # "none", "ia1", "ia2"
HUMAN_EPISODES = 20
HUMAN_EPSILON = 0.1

PLAY_AGAINST_AI = False
PLAY_EPISODES = 20
PLAY_EPSILON = 0.0

RUN_COMPARISON = False


def build_steps(config: RunConfig) -> List[TrainingStep]:
    """
    Construye la lista de pasos según las flags de configuración.

    Args:
        config: Configuración del run (versión, número de episodios).

    Returns:
        Lista de TrainingStep a ejecutar.
    """
    steps = []

    # Self-play
    if RUN_SELF_PLAY:
        steps.append(TrainingStep(
            name="Entrenamiento self-play",
            action="train",
            episodes=config.train_episodes,
            opponent_factory=PlayerAIV,
        ))

    # Evaluación final
    if RUN_EVALUATION:
        steps.append(TrainingStep(
            name="Evaluación final",
            action="evaluate",
            episodes=config.eval_episodes,
            opponent_factory=PlayerAIV,
            load_opponent_checkpoint=(config.path_p2_sel, config.path_p2_turn),
        ))

    # Fine-tuning contra humano
    if HUMAN_OPPONENT != "none":
        player1_checkpoint = None
        if HUMAN_OPPONENT == "ia2":
            player1_checkpoint = (config.path_p2_sel, config.path_p2_turn)
        elif HUMAN_OPPONENT != "ia1":
            raise ValueError(
                f"HUMAN_OPPONENT debe ser 'none', 'ia1' o 'ia2', no {HUMAN_OPPONENT!r}"
            )

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

    # Jugar contra IA (modo humano vs IA)
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


# ================================================================
# PUNTO DE ENTRADA
# ================================================================

if __name__ == "__main__":
    config = RunConfig(
        version=VERSION,
        train_episodes=TRAIN_EPISODES,
        eval_episodes=EVAL_EPISODES,
    )

    if RUN_COMPARISON:
        # Definir runs a comparar
        run_specs = [
            RunSpec(
                run_name="baseline",
                N=N_BATCH,
                train_batches=TRAIN_EPISODES,
                eval_batches=EVAL_EPISODES,
            ),
            RunSpec(
                run_name="NSteps_2",
                N=N_BATCH,
                train_batches=TRAIN_EPISODES,
                eval_batches=EVAL_EPISODES,
                constants_overrides={"N_STEP": 2},
            ),
            RunSpec(
                run_name="NSteps_3",
                N=N_BATCH,
                train_batches=TRAIN_EPISODES,
                eval_batches=EVAL_EPISODES,
                constants_overrides={"N_STEP": 3},
            ),
            RunSpec(
                run_name="NSteps_4",
                N=N_BATCH,
                train_batches=TRAIN_EPISODES,
                eval_batches=EVAL_EPISODES,
                constants_overrides={"N_STEP": 4},
            ),
            RunSpec(
                run_name="NSteps_5",
                N=N_BATCH,
                train_batches=TRAIN_EPISODES,
                eval_batches=EVAL_EPISODES,
                constants_overrides={"N_STEP": 5},
            ),
        ]
        run_comparison(config, run_specs)
    else:
        steps = build_steps(config)
        n_efectivo = 1 if HUMAN_OPPONENT != "none" or PLAY_AGAINST_AI else N_BATCH
        MainV(config, steps, N=n_efectivo).run()