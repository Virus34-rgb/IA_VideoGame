"""
Módulo de logging para Castle Game.

Guarda métricas de pérdida (loss) y progreso (snapshots) en formato CSV,
y genera gráficas de evolución al final del entrenamiento.
"""
import csv
import json
import os
import time
from dataclasses import dataclass, field, asdict
import csv as csv_module

import constants  # Necesario para ELO_INITIAL


@dataclass
class LossRecord:
    """Registro de una pérdida (loss) en un paso de replay."""
    episode: int
    replayed_count: int
    player: str
    network: str
    loss: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class SnapshotRecord:
    episode: int
    p1_winrate: float
    p2_winrate: float
    drawrate: float
    p1_damage_avg: float
    p2_damage_avg: float
    avg_turns: float
    p1_reward_avg: float
    p2_reward_avg: float
    elo_p1: float
    elo_p2: float
    pool_elo_mean: float
    pool_elo_max: float
    pool_elo_min: float
    # NUEVO: diagnóstico de exploración (NoisyNets) y de perfil de oponente
    sigma_sel_p1: float = 0.0
    sigma_turn_p1: float = 0.0
    sigma_sel_p2: float = 0.0
    sigma_turn_p2: float = 0.0
    profile_p1_aggression: float = 0.5
    profile_p1_movement: float = 0.5
    profile_p1_defense: float = 0.5
    profile_p1_dmgratio: float = 0.5
    profile_p2_aggression: float = 0.5
    profile_p2_movement: float = 0.5
    profile_p2_defense: float = 0.5
    profile_p2_dmgratio: float = 0.5
    timestamp: float = field(default_factory=time.time)


class MetricsLogger:
    """
    Logger para métricas de entrenamiento.

    Guarda pérdidas (loss) y snapshots de progreso en archivos CSV,
    y genera gráficas al final del entrenamiento.
    """

    def __init__(self, output_dir: str, run_name: str) -> None:
        """
        Args:
            output_dir: Directorio donde guardar los archivos.
            run_name: Nombre del run (se usa en nombres de archivo).
        """
        self.output_dir = output_dir
        self.run_name = run_name
        os.makedirs(output_dir, exist_ok=True)

        self.loss_path = os.path.join(output_dir, f"{run_name}_loss.csv")
        self.snapshot_path = os.path.join(output_dir, f"{run_name}_progress.csv")
        self.config_path = os.path.join(output_dir, f"{run_name}_config.json")

        self._loss_header_written = os.path.exists(self.loss_path)
        self._snapshot_header_written = os.path.exists(self.snapshot_path)
        self._loss_buffer: list = []
        self._loss_buffer_max = 500

    def dump_config(self, config_module, extra=None) -> None:
        """
        Guarda la configuración actual en un archivo JSON.

        Args:
            config_module: Módulo de constantes (normalmente 'constants').
            extra: Diccionario con información adicional (ej. N, steps).
        """
        values = {
            name: getattr(config_module, name)
            for name in dir(config_module)
            if name.isupper() and not name.startswith("_")
        }
        payload = {
            "run_name": self.run_name,
            "timestamp": time.time(),
            "constants": values,
        }
        if extra:
            payload["extra"] = extra

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    def log_loss(self, episode: int, replayed_count: int, player: str, network: str, loss_value: float) -> None:
        if loss_value is None:
            return
        record = LossRecord(episode, replayed_count, player, network, float(loss_value))
        self._loss_buffer.append(record)
        if len(self._loss_buffer) >= self._loss_buffer_max:
            self.flush_loss_buffer()

    def flush_loss_buffer(self) -> None:
        """Vuelca el buffer de loss acumulado a disco en una sola apertura de archivo."""
        if not self._loss_buffer:
            return
        write_header = not self._loss_header_written
        with open(self.loss_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(self._loss_buffer[0]).keys())
            if write_header:
                writer.writeheader()
            for record in self._loss_buffer:
                writer.writerow(asdict(record))
        self._loss_header_written = True
        self._loss_buffer.clear()

    def log_snapshot(
        self,
        episode: int,
        player1: any,
        player2: any,
        stats: any,
        elo_p1: float = None,
        elo_p2: float = None,
        pool_elos: dict = None,
        sigmas: dict = None,
        profile_p1: list = None,
        profile_p2: list = None,
    ) -> None:
        """
        Registra un snapshot de progreso (métricas y Elo).

        Args:
            episode: Número de lote.
            player1, player2: Agentes (se usan para obtener epsilon si existiera, pero ya no se usa).
            stats: Objeto StatsV con estadísticas acumuladas.
            elo_p1: Elo actual de P1 (opcional, se obtiene de player1.elo si no se da).
            elo_p2: Elo actual de P2 (opcional, se obtiene de player2.elo si no se da).
            pool_elos: Diccionario {checkpoint_id: elo} de la pool de oponentes.
        """
        partidas = max(stats.partidas, 1)

        # Obtener Elos si no se proporcionan explícitamente
        if elo_p1 is None:
            elo_p1 = getattr(player1, "elo", constants.ELO_INITIAL)
        if elo_p2 is None:
            elo_p2 = getattr(player2, "elo", constants.ELO_INITIAL)

        # Calcular estadísticas de la pool
        if pool_elos and len(pool_elos) > 0:
            elos_list = list(pool_elos.values())
            pool_mean = sum(elos_list) / len(elos_list)
            pool_max = max(elos_list)
            pool_min = min(elos_list)
        else:
            pool_mean = pool_max = pool_min = constants.ELO_INITIAL

        sigmas = sigmas or {}
        profile_p1 = profile_p1 or [0.5, 0.5, 0.5, 0.5]
        profile_p2 = profile_p2 or [0.5, 0.5, 0.5, 0.5]

        record = SnapshotRecord(
            episode=episode,
            p1_winrate=stats.p1_victories / partidas * 100,
            p2_winrate=stats.p2_victories / partidas * 100,
            drawrate=stats.empates / partidas * 100,
            p1_damage_avg=stats.p1_damage / partidas,
            p2_damage_avg=stats.p2_damage / partidas,
            avg_turns=stats.total_turns / partidas,
            p1_reward_avg=stats.total_reward_p1 / partidas,
            p2_reward_avg=stats.total_reward_p2 / partidas,
            elo_p1=elo_p1,
            elo_p2=elo_p2,
            pool_elo_mean=pool_mean,
            pool_elo_max=pool_max,
            pool_elo_min=pool_min,
            sigma_sel_p1=sigmas.get("sel_p1", 0.0),
            sigma_turn_p1=sigmas.get("turn_p1", 0.0),
            sigma_sel_p2=sigmas.get("sel_p2", 0.0),
            sigma_turn_p2=sigmas.get("turn_p2", 0.0),
            profile_p1_aggression=profile_p1[0],
            profile_p1_movement=profile_p1[1],
            profile_p1_defense=profile_p1[2],
            profile_p1_dmgratio=profile_p1[3],
            profile_p2_aggression=profile_p2[0],
            profile_p2_movement=profile_p2[1],
            profile_p2_defense=profile_p2[2],
            profile_p2_dmgratio=profile_p2[3],
        )
        self._append_csv(self.snapshot_path, record, self._snapshot_header_written)
        self._snapshot_header_written = True

    @staticmethod
    def _append_csv(path: str, record: any, header_written: bool) -> None:
        """Añade un registro a un archivo CSV, escribiendo cabecera si es necesario."""
        row = asdict(record)
        write_header = not header_written
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def plot_progress(self, show: bool = True) -> list:
        """
        Genera gráficas de progreso a partir del archivo de snapshots.

        Incluye: winrate, daño medio, reward medio, y evolución de Elo.
        Las gráficas de epsilon han sido eliminadas (Noisy Networks las reemplaza).

        Args:
            show: Si se debe mostrar la figura en pantalla.

        Returns:
            Lista de rutas de las imágenes guardadas.
        """
        import matplotlib.pyplot as plt
        fig_paths = []

        # Graficar progreso (snapshots)
        if os.path.exists(self.snapshot_path):
            rows = self._read_csv(self.snapshot_path)
            if rows:
                episodes = [int(r["episode"]) for r in rows]
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle(f"Progreso — {self.run_name}")

                # 1. Winrate (arriba izquierda)
                axes[0, 0].plot(episodes, [float(r["p1_winrate"]) for r in rows], label="P1", color="blue")
                axes[0, 0].plot(episodes, [float(r["p2_winrate"]) for r in rows], label="P2", color="orange")
                axes[0, 0].plot(episodes, [float(r["drawrate"]) for r in rows], label="Empates", color="gray", linestyle="--")
                axes[0, 0].set_title("Winrate (%)")
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

                # 2. Daño medio (arriba derecha)
                axes[0, 1].plot(episodes, [float(r["p1_damage_avg"]) for r in rows], label="P1", color="blue")
                axes[0, 1].plot(episodes, [float(r["p2_damage_avg"]) for r in rows], label="P2", color="orange")
                axes[0, 1].set_title("Daño medio por partida")
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

                # 3. Reward medio (abajo izquierda)
                axes[1, 0].plot(episodes, [float(r["p1_reward_avg"]) for r in rows], label="P1", color="blue")
                axes[1, 0].plot(episodes, [float(r["p2_reward_avg"]) for r in rows], label="P2", color="orange")
                axes[1, 0].set_title("Reward media por partida")
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

                # 4. Evolución de Elo (abajo derecha) — REEMPLAZA A EPSILON
                axes[1, 1].plot(episodes, [float(r["elo_p1"]) for r in rows], label="P1", color="blue", linewidth=2)
                axes[1, 1].plot(episodes, [float(r["elo_p2"]) for r in rows], label="P2", color="orange", linewidth=2)
                axes[1, 1].plot(episodes, [float(r["pool_elo_mean"]) for r in rows], label="Pool media", color="green", linestyle="--", linewidth=1.5)
                # Opcional: banda de máx/mín de la pool
                if "pool_elo_max" in rows[0] and "pool_elo_min" in rows[0]:
                    pool_max = [float(r["pool_elo_max"]) for r in rows]
                    pool_min = [float(r["pool_elo_min"]) for r in rows]
                    axes[1, 1].fill_between(episodes, pool_min, pool_max, alpha=0.15, color="green", label="Rango pool")
                axes[1, 1].axhline(y=constants.ELO_INITIAL, color="gray", linestyle=":", label="Elo inicial")
                axes[1, 1].set_title("Evolución Elo")
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

                plt.tight_layout()
                path = os.path.join(self.output_dir, f"{self.run_name}_progress.png")
                fig.savefig(path, dpi=150)
                fig_paths.append(path)

        # Graficar exploración (sigma NoisyNets) y perfil de oponente observado
        if os.path.exists(self.snapshot_path):
            rows = self._read_csv(self.snapshot_path)
            if rows and "sigma_turn_p1" in rows[0]:
                episodes = [int(r["episode"]) for r in rows]
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle(f"Exploración y perfil de oponente — {self.run_name}")

                # 1. Sigma medio de TurnNetwork (arriba izquierda)
                axes[0, 0].plot(episodes, [float(r["sigma_turn_p1"]) for r in rows], label="P1", color="blue")
                axes[0, 0].plot(episodes, [float(r["sigma_turn_p2"]) for r in rows], label="P2", color="orange")
                axes[0, 0].axhline(y=constants.SIGMA_MIN, color="red", linestyle=":", label="Sigma mínimo")
                axes[0, 0].set_title("Sigma medio — TurnNetwork")
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

                # 2. Sigma medio de SelectionNetwork (arriba derecha)
                axes[0, 1].plot(episodes, [float(r["sigma_sel_p1"]) for r in rows], label="P1", color="blue")
                axes[0, 1].plot(episodes, [float(r["sigma_sel_p2"]) for r in rows], label="P2", color="orange")
                axes[0, 1].axhline(y=constants.SIGMA_MIN, color="red", linestyle=":", label="Sigma mínimo")
                axes[0, 1].set_title("Sigma medio — SelectionNetwork")
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

                # 3. Perfil de rival observado por P1 (abajo izquierda)
                axes[1, 0].plot(episodes, [float(r["profile_p1_aggression"]) for r in rows], label="Agresividad")
                axes[1, 0].plot(episodes, [float(r["profile_p1_movement"]) for r in rows], label="Movimiento")
                axes[1, 0].plot(episodes, [float(r["profile_p1_defense"]) for r in rows], label="Defensa/cura")
                axes[1, 0].plot(episodes, [float(r["profile_p1_dmgratio"]) for r in rows], label="Ratio daño")
                axes[1, 0].axhline(y=0.5, color="gray", linestyle=":")
                axes[1, 0].set_ylim(0, 1)
                axes[1, 0].set_title("Perfil rival observado por P1")
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

                # 4. Perfil de rival observado por P2 (abajo derecha)
                axes[1, 1].plot(episodes, [float(r["profile_p2_aggression"]) for r in rows], label="Agresividad")
                axes[1, 1].plot(episodes, [float(r["profile_p2_movement"]) for r in rows], label="Movimiento")
                axes[1, 1].plot(episodes, [float(r["profile_p2_defense"]) for r in rows], label="Defensa/cura")
                axes[1, 1].plot(episodes, [float(r["profile_p2_dmgratio"]) for r in rows], label="Ratio daño")
                axes[1, 1].axhline(y=0.5, color="gray", linestyle=":")
                axes[1, 1].set_ylim(0, 1)
                axes[1, 1].set_title("Perfil rival observado por P2")
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

                plt.tight_layout()
                path = os.path.join(self.output_dir, f"{self.run_name}_exploration_profile.png")
                fig.savefig(path, dpi=150)
                fig_paths.append(path)

        # Graficar pérdidas (loss) si existen
        if os.path.exists(self.loss_path):
            rows = self._read_csv(self.loss_path)
            if rows:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                fig.suptitle(f"Loss — {self.run_name}")

                for network, ax in [("turn", axes[0]), ("selection", axes[1])]:
                    for player in ("p1", "p2"):
                        subset = [r for r in rows if r["network"] == network and r["player"] == player]
                        if subset:
                            ax.plot(
                                [int(r["replayed_count"]) for r in subset],
                                [float(r["loss"]) for r in subset],
                                label=player,
                                alpha=0.6,
                            )
                    ax.set_title(f"Loss — {network}")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                plt.tight_layout()
                path = os.path.join(self.output_dir, f"{self.run_name}_loss.png")
                fig.savefig(path, dpi=150)
                fig_paths.append(path)

        if show and fig_paths:
            plt.show()
        return fig_paths

    @staticmethod
    def _read_csv(path: str) -> list:
        """Lee un archivo CSV y devuelve una lista de diccionarios."""
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv_module.DictReader(f))

    # ------------------------------------------------------------
    # Comparación de runs
    # ------------------------------------------------------------

    @staticmethod
    def compare_runs(output_dir: str, run_names: list, labels: list = None, show: bool = True) -> str:
        """
        Compara múltiples runs superponiendo sus curvas de progreso.

        Args:
            output_dir: Directorio donde están los archivos de logs.
            run_names: Lista de nombres de runs (cada uno con su *_progress.csv).
            labels: Etiquetas para la leyenda (por defecto usa run_names).
            show: Si mostrar la figura en pantalla.

        Returns:
            Ruta de la imagen guardada.
        """
        import matplotlib.pyplot as plt

        runs = [(output_dir, run_name) for run_name in run_names]
        
        if not runs:
            return None

        if labels is None:
            labels = [run_name for _, run_name in runs]

        if len(labels) != len(runs):
            raise ValueError("Debe haber una label por cada run.")

        # Si no se especifica dónde guardar la comparación,
        # usamos el directorio padre común.
        if output_dir is None:
            output_dir = os.path.dirname(runs[0][0])

        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("Comparación de runs")

        any_data = False

        for (run_dir, run_name), label in zip(runs, labels):

            path = os.path.join(
                run_dir,
                f"{run_name}_progress.csv"
            )

            if not os.path.exists(path):
                print(f"[AVISO] No existe: {path}")
                continue

            rows = MetricsLogger._read_csv(path)

            if not rows:
                print(f"[AVISO] CSV vacío: {path}")
                continue

            any_data = True
            episodes = [int(r["episode"]) for r in rows]

            # Winrate P1
            axes[0, 0].plot(episodes, [float(r["p1_winrate"]) for r in rows], label=label)
            # Daño medio P1
            axes[0, 1].plot(episodes, [float(r["p1_damage_avg"]) for r in rows], label=label)
            # Reward medio P1
            axes[1, 0].plot(episodes, [float(r["p1_reward_avg"]) for r in rows], label=label)
            # Turnos medios
            axes[1, 1].plot(episodes, [float(r["avg_turns"]) for r in rows], label=label)

        axes[0, 0].set_title("Winrate P1 (%)")
        axes[0, 1].set_title("Daño medio P1")
        axes[1, 0].set_title("Reward medio P1")
        axes[1, 1].set_title("Turnos medios por partida")

        for ax in axes.flat:
            ax.set_xlabel("Lote")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, "comparison_progress.png")
        fig.savefig(path, dpi=150)
        if show and any_data:
            plt.show()

        return path