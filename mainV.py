"""
Punto de entrada principal del proyecto Castle Game.

Orquesta la ejecución de pasos de entrenamiento/evaluación,
gestión de la pool de oponentes, y comparación de runs con diferentes
hiperparámetros (usando RunSpec).
"""
import math
import os
import random
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, List
import numpy
import torch
import yaml

from AI.Agent.opponent_poolV import OpponentPoolV
from AI.Agent.playerAIV import PlayerAIV
from AI.Agent.playerNoIAV import PlayerNoAIV
from AI.Environment.vectorizedEnvironment import VectorizedEnvironment
from AI.Agent.trainerV import TrainerV
from AI.Logging.metrics_logger import MetricsLogger
from config import RunConfig
from training_step import TrainingStep
import constants
if constants.USE_GUI:
    from AI.Agent.playerGUI import PlayerGUIV

# Importar el módulo de wandb (si no existe, se puede desactivar)
try:
    from AI.Logging import wandb_setup
except ImportError:
    wandb_setup = None
    print("Advertencia: wandb_setup no encontrado. Desactivando wandb.")

def set_seed(seed: Optional[int]) -> None:
    """Fija las semillas de torch, numpy y random para reproducibilidad.
    Si seed es None, no hace nada (comportamiento no determinista, por defecto)."""
    if seed is None:
        return
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    
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
        self.run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Limpiar directorios antiguos si está configurado
        if constants.DELETE_DIRECTORIES:
            shutil.rmtree(self.config.p1_path, ignore_errors=True)
            shutil.rmtree(self.config.p2_path, ignore_errors=True) 

        # Crear directorios necesarios
        os.makedirs(self.config.p1_path, exist_ok=True)
        os.makedirs(self.config.p2_path, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Inicializar entorno y jugador
        self.environment = VectorizedEnvironment(self.N)
        self.player1 = self.player_class(self.N, self.environment)
        
        clean_suffix = self.sanitize_filename(self.config.suffix) if self.config.suffix else ""
        if clean_suffix:
            run_nameA = f"v{self.config.version}_{clean_suffix}"
        else:
            run_nameA = f"v{self.config.version}"

        # Inicializar logger (CSV)
        self.logger = MetricsLogger(
            output_dir=self.log_dir,
            run_name = run_nameA,
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

        # Inicializar wandb si está activo
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
                }
            )

        self._print_configuration()
        
    def sanitize_filename(self,name: str) -> str:
        """Reemplaza caracteres no válidos en nombres de archivo por '_'."""
        return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

    def run(self) -> None:
        """Ejecuta todos los pasos definidos y genera gráficos finales."""
        self.setup()
        for step in self.steps:
            self._run_step(step)
        self.logger.flush_loss_buffer()
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
        if step.opponent_factory is PlayerNoAIV:
            if constants.USE_GUI:
                return PlayerGUIV()
            else:
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
        if constants.USE_WANDB:
            print(f"Wandb:    Activo (revisa el dashboard)")
        else:
            print(f"Wandb:    Desactivado")
        print("=" * 65)

    def _print_summary(self) -> None:
        """Muestra el resumen final del run."""
        print("=" * 65)
        print("                         FINALIZADO")
        print("=" * 65)
        print(f"Modelos guardados en: {self.config.base_path}")
        print(f"Logs (loss/progreso/config): {self.log_dir}")
        if constants.USE_WANDB:
            print(f"Wandb:    Revisa el dashboard para gráficas en tiempo real")

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
    seed: Optional[int] = None


def load_config_yaml(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        print(f"Advertencia: {path} no encontrado. Usando valores por defecto.")
        return {}
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

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
    if run_spec.seed is not None and constants.SEED != 'None':
        set_seed(run_spec.seed)
    
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

def build_steps(config: RunConfig) -> List[TrainingStep]:
    """
    Construye la lista de pasos según las flags de configuración (de constants).

    Args:
        config: Configuración del run (versión, número de episodios).

    Returns:
        Lista de TrainingStep a ejecutar.
    """
    steps = []

    # Self-play
    if constants.RUN_SELF_PLAY:
        steps.append(TrainingStep(
            name="Entrenamiento self-play",
            action="train",
            episodes=config.train_episodes,
            opponent_factory=PlayerAIV,
        ))

    # Evaluación final
    if constants.RUN_EVALUATION:
        steps.append(TrainingStep(
            name="Evaluación final",
            action="evaluate",
            episodes=config.eval_episodes,
            opponent_factory=PlayerAIV,
            load_opponent_checkpoint=(config.path_p2_sel, config.path_p2_turn),
        ))

    # Fine-tuning contra humano
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
            episodes=constants.HUMAN_EPISODES,
            opponent_factory=PlayerNoAIV,
            player1_checkpoint=player1_checkpoint,
            learn_p1=True,
            learn_p2=False,
            epsilon_turn=constants.HUMAN_EPSILON,
        ))

    # Jugar contra IA (modo humano vs IA)
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


# ================================================================
# PUNTO DE ENTRADA
# ================================================================

if __name__ == "__main__":
    # 1. Cargar configuración desde YAML
    yaml_config = load_config_yaml()

    # 2. Sobrescribir constants con los valores del YAML
    NON_CONSTANT_YAML_KEYS = {"comparisons"}

    for key, value in yaml_config.items():
        if key in NON_CONSTANT_YAML_KEYS:
            continue
        if hasattr(constants, key):
            setattr(constants, key, value)
        else:
            print(f"Advertencia: {key} no existe en constants, se omite")
    # 3. Crear configuración del run
    config = RunConfig(
        version=constants.VERSION,
        train_episodes=constants.TRAIN_EPISODES,
        eval_episodes=constants.EVAL_EPISODES,
        suffix=constants.RUN_NAME_SUFFIX,   # <--- NUEVO
    )

    # 4. Ejecutar comparación o run simple
    if constants.RUN_COMPARISON and "comparisons" in yaml_config:
        run_specs = []
        for comp in yaml_config["comparisons"]:
            run_specs.append(RunSpec(
                run_name=comp.get("run_name", "unnamed"),
                N=comp.get("N", constants.N_BATCH),
                train_batches=comp.get("train_batches", constants.TRAIN_EPISODES),
                eval_batches=comp.get("eval_batches", constants.EVAL_EPISODES),
                constants_overrides=comp.get("overrides", {}),
                seed=comp.get("seed", constants.SEED),   # NUEVO: usa la seed del run o la global como fallback
                # player_class se puede añadir si se quiere, pero por ahora no
            ))
        run_comparison(config, run_specs)
    else:
        if(constants.SEED is not None and constants.SEED != 'None'):
            set_seed(constants.SEED) 
        steps = build_steps(config)
        n_efectivo = 1 if constants.HUMAN_OPPONENT != "none" or constants.PLAY_AGAINST_AI else constants.N_BATCH
        MainV(config, steps, N=n_efectivo).run()