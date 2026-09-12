"""
Punto de entrada principal del proyecto Castle Game.

Orquesta la ejecución de pasos de entrenamiento/evaluación,
gestión de la pool de oponentes, y comparación de runs con diferentes
hiperparámetros (usando RunSpec).
"""
import math
import os
import pstats
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
from AI.Agent.player_rusher import PlayerRusherV
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
        self.playerRusher: Optional[PlayerRusherV] = None
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
        self.playerRusher = PlayerRusherV(self.N, self.environment)

        clean_suffix = self.sanitize_filename(self.config.suffix) if self.config.suffix else ""
        if clean_suffix:
            run_nameA = f"v{self.config.version}_{clean_suffix}"
        else:
            run_nameA = f"v{self.config.version}"

        # Inicializar logger (CSV)
        self.logger = MetricsLogger(
            output_dir=self.log_dir,
            run_name=run_nameA,
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
                    "rusher_opponent_percentage": constants.RUSHER_OPPONENT_PERCENTAGE,
                }
            )

        self._print_configuration()

    def sanitize_filename(self, name: str) -> str:
        """Reemplaza caracteres no válidos en nombres de archivo por '_'."""
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

        stats2_path_for_step = step.stats_path if step.stats_path else self.config.stats2_path

        # Crear entrenador
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

        # Determinar parámetros de ejecución
        if step.learn_p1 is None and step.learn_p2 is None:
            # Modo automático: entrenar si action=="train", evaluar si "evaluate"
            if step.action == "train":
                trainer.train()
            else:
                trainer.evaluate(fixed_rusher_aggression=step.rusher_aggression)

        else:
            # Modo personalizado con flags explícitos
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

    def _print_profile_stats(self):
        """Carga el archivo de perfil de cProfile y muestra las estadísticas."""
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
    if run_spec.seed is not None and constants.SEED is not None:
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

        if constants.RUN_RUSHER_FINETUNE:
            finetune_stats = (config.stats_rusher_finetune_low, config.stats_rusher_finetune_medium, config.stats_rusher_finetune_hight)
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

    # Evaluación final
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

    # Fine-tuning contra humano (aprendizaje)
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

    # Jugar contra IA (modo humano vs IA) (Evaluacion)
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
# MULTI-SEED + COMPARACIÓN CON BASELINE
# ================================================================
# Las constantes MULTI_SEED_* vienen de constants.py / config.yaml.


def _parse_stats_file(path: str) -> dict:
    """
    Extrae {nombre_metrica: valor_numerico} de un archivo stats.txt.
    Acepta líneas tipo:
        "Partidas:                  40960"
        "Victorias P1:              26911 (65.70%)"
        "Victorias IA:              12490 -> 0.6098"
    """
    import re as _re
    metrics = {}
    if not os.path.exists(path):
        return metrics
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip()
            if ":" not in line:
                continue
            name, rest = line.split(":", 1)
            name = name.strip()
            if not name or name.startswith("-") or name.startswith("="):
                continue
            nums = _re.findall(r"[-+]?\d+(?:\.\d+)?", rest)
            if not nums:
                continue
            try:
                vals = [float(x) for x in nums]
            except ValueError:
                continue
            metrics[name] = vals[0]
            if len(vals) >= 2:
                # Segunda cifra: útil para "12490 -> 0.6098" y "26911 (65.70%)"
                metrics[f"{name} (2nd)"] = vals[1]
    return metrics


def _find_seed_dirs(parent_dir: str, suffix_filter: str | None = None) -> List[str]:
    """
    Devuelve las subcarpetas de `parent_dir` que corresponden a una seed.
    Reconoce dos layouts:
      1. Nuevo: 'parent_dir/s42/', 'parent_dir/s43/', ...  (subcarpetas directas)
      2. Antiguo: 'parent_dir/IAV*_s42/', '..._s43/', ... (carpetas hermanas)
    Si `suffix_filter` se pasa, solo aplica al layout 2.
    """
    import glob
    candidates: List[str] = []

    # Layout 1: subcarpetas directas s<num>
    candidates += glob.glob(os.path.join(parent_dir, "s[0-9]*"))

    # Layout 2: IAV*_s<num> (compatibilidad con carpetas antiguas)
    if suffix_filter:
        pattern = os.path.join(parent_dir, f"IAV*_{suffix_filter}_s[0-9]*")
    else:
        pattern = os.path.join(parent_dir, "IAV*_s[0-9]*")
    candidates += glob.glob(pattern)

    valid = []
    for p in candidates:
        base = os.path.basename(os.path.normpath(p))
        if re.fullmatch(r"s\d+", base) or re.search(r"_s\d+$", base):
            valid.append(p)

    def _extract_seed(p):
        base = os.path.basename(os.path.normpath(p))
        m = re.fullmatch(r"s(\d+)", base) or re.search(r"_s(\d+)$", base)
        return int(m.group(1)) if m else 0

    seen = set()
    unique = []
    for p in valid:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return sorted(unique, key=_extract_seed)

def _load_stats_entry(run_dir: str, label: str) -> dict:
    """
    Carga stats2.txt y stats_rusher_*.txt de `run_dir`. `label` es la etiqueta
    corta que aparecerá como nombre de columna (p. ej. 's42', 'base', 'b1').
    """
    entry = {"seed": label, "main": {}, "rusher_0": {}, "rusher_05": {}, "rusher_1": {}}
    for fname in ("stats2.txt", "stats.txt"):
        p = os.path.join(run_dir, fname)
        if os.path.exists(p):
            entry["main"] = _parse_stats_file(p)
            break
    for tag, fname in (("rusher_0", "stats_rusher_aggr_0.txt"),
                       ("rusher_05", "stats_rusher_aggr_05.txt"),
                       ("rusher_1", "stats_rusher_aggr_1.txt")):
        p = os.path.join(run_dir, fname)
        if os.path.exists(p):
            entry[tag] = _parse_stats_file(p)
    return entry
def _load_baseline_entries(baseline_dir) -> tuple[list, bool]:
    """
    Devuelve (lista_de_entries, es_multi).

    Acepta:
      - None → ([], False)
      - str apuntando a un fichero de stats → un único entry
      - str apuntando a una carpeta con stats2.txt → un único entry
      - str apuntando a una carpeta que contiene subcarpetas IAV*_s<seed> →
        un entry por cada subcarpeta encontrada
      - list/tuple de str, cada uno con un run distinto → un entry por elemento
    """
    if baseline_dir is None:
        return [], False

    # ---------- Lista explícita de rutas ----------
    if isinstance(baseline_dir, (list, tuple)):
        entries = []
        for i, path in enumerate(baseline_dir):
            if not os.path.exists(path):
                print(f"[compare] Baseline path no existe: {path}")
                continue
            name = os.path.basename(os.path.normpath(path))
            m = re.search(r"_s(\d+)$", name)
            label = f"s{m.group(1)}" if m else f"b{i+1}"
            entries.append(_load_stats_entry(path, label))
        return entries, len(entries) > 1

    # ---------- Ruta única ----------
    if not os.path.exists(baseline_dir):
        print(f"[compare] Baseline no existe: {baseline_dir}")
        return [], False

    if os.path.isfile(baseline_dir):
        entry = {"seed": "base", "main": _parse_stats_file(baseline_dir),
                 "rusher_0": {}, "rusher_05": {}, "rusher_1": {}}
        return [entry], False

    # ¿Carpeta contenedora con subcarpetas IAV*_s<seed>?
    seed_subdirs = _find_seed_dirs(baseline_dir)
    if seed_subdirs:
        entries = []
        for sd in seed_subdirs:
            name = os.path.basename(sd)
            # Acepta tanto "s42" (layout nuevo) como "IAV..._s42" (layout antiguo)
            m = re.fullmatch(r"s(\d+)", name) or re.search(r"_s(\d+)$", name)
            short = f"s{m.group(1)}" if m else name
            entries.append(_load_stats_entry(sd, short))
        return entries, True

    # Carpeta de un solo run
    entry = _load_stats_entry(baseline_dir, "base")
    return [entry], False

def compare_seeds_against_baseline(
    seeds_dir: str,
    baseline_dir=None,
    output_json: str | None = None,
    output_txt: str | None = None,
    suffix_filter: str | None = None,
) -> str:
    """
    Compara stats por seed contra uno o varios baselines.

    Layout de columnas:
      Métrica | BL_s42 | BL_s43 | BL_s44 | BL_μ | NW_s42 | NW_s43 | NW_s44 | NW_μ | Δμ
    donde:
      - BL_* son los baselines (una columna por seed encontrada).
      - NW_* son las seeds nuevas.
      - BL_μ / NW_μ son medias y solo aparecen si hay ≥2 seeds en el grupo.
      - Δμ = NW_μ − BL_μ, solo si ambos grupos tienen ≥2 seeds.
    """
    import json

    # --- Cargar seeds nuevas ---
    new_entries = []
    for sd in _find_seed_dirs(seeds_dir, suffix_filter=suffix_filter):
        name = os.path.basename(sd)
        m = re.search(r"_s(\d+)$", name)
        short = f"s{m.group(1)}" if m else name
        new_entries.append(_load_stats_entry(sd, short))

    if not new_entries:
        msg = f"[compare] No se encontraron subcarpetas IAV*_s<seed> en {seeds_dir}"
        if output_txt:
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write(msg + "\n")
        else:
            print(msg)
        return msg

    # --- Cargar baselines ---
    baseline_entries, baseline_is_multi = _load_baseline_entries(baseline_dir)

    # --- Métricas a comparar ---
    key_metrics = [
        ("main",      "Win ratio P1 (sin empates)",      "{:.2f}"),
        ("main",      "Turnos medios por partida",       "{:.2f}"),
        ("main",      "Elo P1",                          "{:.1f}"),
        ("main",      "Elo P2",                          "{:.1f}"),
        ("main",      "Daño por sobrekill medio P1",     "{:.2f}"),
        ("main",      "Kill confirmed medio P1",         "{:.2f}"),
        ("main",      "Defensas desperdiciadas P1",      "{:.2f}"),
        ("main",      "Movimientos estratégicos P1 (%)", "{:.2f}"),
        ("rusher_0",  "Victorias IA vs rusher 0.0",      "{:.4f}"),
        ("rusher_05", "Victorias IA vs rusher 0.5",      "{:.4f}"),
        ("rusher_1",  "Victorias IA vs rusher 1.0",      "{:.4f}"),
    ]

    n_bl = len(baseline_entries)
    n_nw = len(new_entries)
    show_bl_mean = n_bl >= 2
    show_nw_mean = n_nw >= 2
    show_delta = show_bl_mean and show_nw_mean

    # --- Cabeceras de columna en orden ---
    col_headers: List[str] = []
    for e in baseline_entries:
        col_headers.append(f"BL_{e['seed']}")
    if show_bl_mean:
        col_headers.append("BL_μ")
    for e in new_entries:
        col_headers.append(f"NW_{e['seed']}")
    if show_nw_mean:
        col_headers.append("NW_μ")
    if show_delta:
        col_headers.append("Δμ")

    n_val_cols = len(col_headers)

    # --- Construir matriz de valores formateados: filas × columnas ---
    table_rows: List[List[str]] = []  # una lista por métrica
    mean_rows: List[List[str]] = []   # solo si show_*_mean

    def _fmt(value, fmt):
        return fmt.format(value) if value is not None else "N/A"

    def _get(entry, section, name):
        src = entry["main"] if section == "main" else entry[section]
        return src.get(name)

    for section, name, fmt in key_metrics:
        row = []
        # Baselines
        bl_vals = []
        for e in baseline_entries:
            v = _get(e, section, name)
            bl_vals.append(v)
            row.append(_fmt(v, fmt))
        if show_bl_mean:
            nums = [v for v in bl_vals if v is not None]
            row.append(_fmt(sum(nums) / len(nums) if nums else None, fmt))
        # Nuevas
        nw_vals = []
        for e in new_entries:
            v = _get(e, section, name)
            nw_vals.append(v)
            row.append(_fmt(v, fmt))
        if show_nw_mean:
            nums = [v for v in nw_vals if v is not None]
            row.append(_fmt(sum(nums) / len(nums) if nums else None, fmt))
        # Delta de medias
        if show_delta:
            bl_nums = [v for v in bl_vals if v is not None]
            nw_nums = [v for v in nw_vals if v is not None]
            if bl_nums and nw_nums:
                delta = sum(nw_nums) / len(nw_nums) - sum(bl_nums) / len(bl_nums)
                row.append(fmt.format(delta))
            else:
                row.append("N/A")
        table_rows.append(row)

    # --- Calcular anchos de columna dinámicos ---
    metric_width = max(
        len("Métrica"),
        max((len(name) for _, name, _ in key_metrics), default=0),
    ) + 2
    value_widths = []
    for j, h in enumerate(col_headers):
        content_max = max((len(r[j]) for r in table_rows), default=0)
        value_widths.append(max(len(h), content_max) + 2)

    # --- Construir líneas de la tabla ---
    lines: List[str] = []

    def _border(left, mid, right, fill="─"):
        parts = [fill * metric_width] + [fill * w for w in value_widths]
        return left + mid.join(parts) + right

    def _row(cells):
        # cells[0] = métrica (izquierda), resto = valores (derecha)
        parts = [f" {cells[0]:<{metric_width - 1}}"]
        for i, c in enumerate(cells[1:]):
            parts.append(f" {c:>{value_widths[i] - 1}}")
        return "│" + "│".join(parts) + "│"

    total_width = sum([metric_width] + value_widths) + n_val_cols + 1

    # Cabecera decorativa
    lines.append(_border("┌", "┬", "┐"))
    title = f"COMPARACIÓN MULTI-SEED vs BASELINE  (baseline: {n_bl} seed(s), nuevo: {n_nw} seed(s))"
    if len(title) > total_width - 3:
        title = title[: total_width - 6] + "..."
    lines.append(f"│ {title:<{total_width - 3}} │")
    prueba_actual = f"NW source: {seeds_dir}"
    lines.append(f"│ {prueba_actual:<{total_width - 3}} │")
    if baseline_dir:
        bl_label = baseline_dir if isinstance(baseline_dir, str) else f"{len(baseline_dir)} rutas"
        bl_text = f"Baseline source: {bl_label}"
        if len(bl_text) > total_width - 4:
            bl_text = bl_text[: total_width - 7] + "..."
        lines.append(f"│ {bl_text:<{total_width - 3}} │")
    lines.append(_border("├", "┼", "┤"))

    # Fila de encabezado
    lines.append(_row(["Métrica"] + col_headers))
    lines.append(_border("├", "┼", "┤"))

    # Filas de métricas
    for (section, name, fmt), row in zip(key_metrics, table_rows):
        lines.append(_row([name] + row))

    lines.append(_border("└", "┴", "┘"))

    report_text = "\n".join(lines)

    if output_txt:
        os.makedirs(os.path.dirname(output_txt), exist_ok=True)
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")
    else:
        print(report_text)

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "baseline_entries": baseline_entries,
                    "new_entries": new_entries,
                    "baseline_source": baseline_dir if isinstance(baseline_dir, str) else list(baseline_dir or []),
                },
                f, indent=2,
            )

    return report_text
def append_experiment_log_entry(
    log_path: str,
    new_entries: list,
    config,
    overrides: dict | None = None,
    baseline_dir=None,
        ) -> None:
        """
        Añade una fila a docs/experiment_log.csv con las medias de las seeds
        recién corridas. Crea el fichero con cabecera si no existe. Evita duplicar
        filas con el mismo (suffix, lotes, seeds).

        Parámetros:
            log_path: ruta del csv (p. ej. "docs/experiment_log.csv").
            new_entries: lista de dicts como los que devuelve _load_stats_entry.
            config: RunConfig con .suffix, .train_episodes, .base_path.
            overrides: dict con constantes cambiadas respecto al baseline (opcional).
            baseline_dir: str | list | None. Solo se usa para el campo 'baseline'.
        """
        import csv
        import datetime

        if not new_entries:
            return

        # --- Medias sobre seeds ---
        def _mean(key, section="main"):
            vals = []
            for e in new_entries:
                v = e.get(section, {}).get(key)
                if v is not None:
                    vals.append(v)
            return sum(vals) / len(vals) if vals else None

        win_self = _mean("Win ratio P1 (sin empates)")
        wr_00 = _mean("Victorias IA (2nd)", "rusher_0")
        wr_05 = _mean("Victorias IA (2nd)", "rusher_05")
        wr_10 = _mean("Victorias IA (2nd)", "rusher_1")

        # --- Veredicto derivado (cualitativo, se puede sobreescribir luego) ---
        # Comparamos solo self-play porque es la métrica más ruidosa y la más
        # correlacionada con "vale la pena adoptar esto".
        veredicto = "referencia"
        if baseline_dir is not None:
            bl_entries, _ = _load_baseline_entries(baseline_dir)
            if bl_entries:
                bl_self = [e["main"].get("Win ratio P1 (sin empates)") for e in bl_entries]
                bl_self = [v for v in bl_self if v is not None]
                if bl_self and win_self is not None:
                    delta = win_self - (sum(bl_self) / len(bl_self))
                    if delta >= 3.0:
                        veredicto = "positivo"
                    elif delta <= -3.0:
                        veredicto = "negativo"
                    else:
                        veredicto = "neutro"

        # --- Id, seeds, artefactos ---
        seeds = [e["seed"].replace("s", "") for e in new_entries]
        seeds_str = ";".join(seeds)

        exp_id = (constants.EXPERIMENT_ID or "").strip()
        if not exp_id:
            # Autoincremento: leemos el csv si existe
            next_n = 1
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # header
                    for row in reader:
                        if row and row[0].startswith("EXP-"):
                            try:
                                n = int(row[0].split("-")[1])
                                next_n = max(next_n, n + 1)
                            except (IndexError, ValueError):
                                pass
            exp_id = f"EXP-{next_n:04d}"

        # --- Deduplicación simple: mismo suffix + lotes + seeds → no añadir ---
        dedup_key = (config.suffix, config.train_episodes, seeds_str)
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        if (row["suffix"], int(row["lotes"]), row["seeds"]) == dedup_key:
                            print(f"[exp-log] Ya existe fila para {dedup_key}, se omite.")
                            return
                    except (KeyError, ValueError):
                        continue

        # --- Overrides como string compacto ---
        if overrides:
            ov_str = ";".join(f"{k}={v}" for k, v in overrides.items())
        else:
            ov_str = ""

        # --- Baseline como string ---
        if baseline_dir is None:
            bl_str = ""
        elif isinstance(baseline_dir, (list, tuple)):
            bl_str = ";".join(baseline_dir)
        else:
            bl_str = str(baseline_dir)

        # --- Artefactos: rutas de las carpetas de las seeds ---
        artefactos = ";".join(
            os.path.join(config.base_path, f"s{s}") for s in seeds
        )

        # --- Escribir ---
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        write_header = not os.path.exists(log_path)

        row = {
            "id": exp_id,
            "fecha": datetime.date.today().isoformat(),
            "objetivo": constants.EXPERIMENT_OBJETIVO,
            "suffix": config.suffix,
            "baseline": bl_str,
            "overrides": ov_str,
            "seeds": seeds_str,
            "lotes": config.train_episodes,
            "winrate_self": f"{win_self:.2f}" if win_self is not None else "",
            "wr_rusher_00": f"{wr_00:.4f}" if wr_00 is not None else "",
            "wr_rusher_05": f"{wr_05:.4f}" if wr_05 is not None else "",
            "wr_rusher_10": f"{wr_10:.4f}" if wr_10 is not None else "",
            "veredicto": veredicto,
            "nota": constants.EXPERIMENT_NOTA,
            "artefactos": artefactos,
        }

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        print(f"[exp-log] Añadida fila {exp_id} a {log_path}")
        
def run_multi_seed(base_config, seeds, baseline_dir=None):
    """
    Ejecuta el pipeline completo para cada seed en `seeds`. Cada seed se guarda
    en una subcarpeta 's<seed>/' DENTRO de la carpeta base del experimento.

    Estructura resultante:
        models/IAV{version}_{suffix}/
        ├── s42/
        │   ├── P1/  P2/
        │   ├── stats.txt  stats2.txt  stats_rusher_*.txt
        │   └── ...
        ├── s43/
        ├── s44/
        ├── comparison_report.txt
        └── comparison_report.json
    """
    import time as _time

    base_root = base_config.base_path         # models/IAV2_TRPD_200K (sin _sX)
    os.makedirs(base_root, exist_ok=True)

    print("=" * 70)
    print(f"MODO MULTI-SEED  |  seeds = {seeds}")
    print(f"Directorio raíz: {base_root}")
    print("=" * 70)

    for seed in seeds:
        seed_dir = os.path.join(base_root, f"s{seed}")   # models/.../s42
        os.makedirs(seed_dir, exist_ok=True)

        # Mismo suffix y version que la config base; el override fuerza que
        # base_path (y por tanto p1_path, stats_path, etc.) apunten a la
        # subcarpeta de esta seed.
        seed_config = RunConfig(
            version=base_config.version,
            train_episodes=base_config.train_episodes,
            eval_episodes=base_config.eval_episodes,
            suffix=base_config.suffix,
            base_dir=base_config.base_dir,
            base_path_override=seed_dir,
        )

        print(f"\n{'#' * 65}\n# SEED {seed} → {seed_dir}\n{'#' * 65}")

        # Fijar seed en constants (por si algún módulo lo lee)
        constants.SEED = seed
        set_seed(seed)

        steps = build_steps(seed_config)
        n_efectivo = (1 if constants.HUMAN_OPPONENT != "none"
                        or constants.PLAY_AGAINST_AI else constants.N_BATCH)

        t0 = _time.time()
        MainV(seed_config, steps, N=n_efectivo).run()
        elapsed = _time.time() - t0
        print(f"[seed {seed}] terminado en {MainV._format_time(elapsed)}")

    # Comparación: las seeds viven dentro de base_root, y el reporte va a la
    # raíz del experimento (también base_root).
    compare_seeds_against_baseline(
        seeds_dir=base_root,
        baseline_dir=baseline_dir,
        output_txt=os.path.join(base_root, "comparison_report.txt"),
        output_json=os.path.join(base_root, "comparison_report.json"),
        suffix_filter=None,   # nuevo layout: se buscan subcarpetas 'sN'
    )
    print(f"[compare] Reporte guardado en {os.path.join(base_root, 'comparison_report.txt')}")
        # --- Registrar en el log maestro ---
    new_entries = []
    for sd in _find_seed_dirs(base_root, suffix_filter=None):
        name = os.path.basename(sd)
        m = re.search(r"_s(\d+)$", name) or re.fullmatch(r"s(\d+)", name)
        short = f"s{m.group(1)}" if m else name
        new_entries.append(_load_stats_entry(sd, short))

    append_experiment_log_entry(
        log_path=constants.EXPERIMENT_LOG_PATH,
        new_entries=new_entries,
        config=base_config,
        overrides=constants.EXPERIMENT_OVERRIDES if hasattr(constants, "EXPERIMENT_OVERRIDES") else None,
        baseline_dir=baseline_dir,
    )
    

# ================================================================
# PUNTO DE ENTRADA
# ================================================================

if __name__ == "__main__":
    torch.set_num_threads(2)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"

    yaml_config = load_config_yaml()
    NON_CONSTANT_YAML_KEYS = {"comparisons"}
    for key, value in yaml_config.items():
        if key in NON_CONSTANT_YAML_KEYS:
            continue
        if hasattr(constants, key):
            setattr(constants, key, value)
        else:
            print(f"Advertencia: {key} no existe en constants, se omite")

    config = RunConfig(
        version=constants.VERSION,
        train_episodes=constants.TRAIN_EPISODES,
        eval_episodes=constants.EVAL_EPISODES,
        suffix=constants.RUN_NAME_SUFFIX,
    )

    if constants.RUN_COMPARISON and "comparisons" in yaml_config:
        # Modo comparación de RunSpecs (sin cambios)
        run_specs = []
        for comp in yaml_config["comparisons"]:
            run_specs.append(RunSpec(
                run_name=comp.get("run_name", "unnamed"),
                N=comp.get("N", constants.N_BATCH),
                train_batches=comp.get("train_batches", constants.TRAIN_EPISODES),
                eval_batches=comp.get("eval_batches", constants.EVAL_EPISODES),
                constants_overrides=comp.get("overrides", {}),
                seed=comp.get("seed", constants.SEED),
            ))
        run_comparison(config, run_specs)

    elif constants.MULTI_SEED_ENABLED:
        run_multi_seed(
            config,
            constants.MULTI_SEED_LIST,
            baseline_dir=constants.MULTI_SEED_BASELINE,
        )

    else:
        # Comportamiento original: una seed
        if constants.SEED is not None and constants.SEED != 'None':
            set_seed(constants.SEED)
        steps = build_steps(config)
        n_efectivo = (1 if constants.HUMAN_OPPONENT != "none"
                        or constants.PLAY_AGAINST_AI else constants.N_BATCH)
        MainV(config, steps, N=n_efectivo).run()