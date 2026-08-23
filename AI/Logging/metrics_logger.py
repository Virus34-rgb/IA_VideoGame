import csv
import json
import os
import time
from dataclasses import dataclass, field, asdict
import csv as csv_module

@dataclass
class LossRecord:
    episode: int
    replayed_count: int
    player: str          # "p1" o "p2"
    network: str          # "selection" o "turn"
    loss: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class SnapshotRecord:
    episode: int
    epsilon_turn_p1: float
    epsilon_turn_p2: float
    epsilon_sel_p1: float
    epsilon_sel_p2: float
    p1_winrate: float
    p2_winrate: float
    drawrate: float
    p1_damage_avg: float
    p2_damage_avg: float
    avg_turns: float
    p1_reward_avg: float
    p2_reward_avg: float
    timestamp: float = field(default_factory=time.time)


class MetricsLogger:
    """
    Centraliza el logging de una corrida de entrenamiento/evaluación:
    - loss por replay (CSV, para graficar convergencia)
    - snapshots periódicos de progreso (CSV, para graficar la curva de aprendizaje)
    - volcado de configuración/hiperparámetros usados (JSON, para reproducir/comparar corridas)
    """

    def __init__(self, output_dir, run_name):
        self.output_dir = output_dir
        self.run_name = run_name
        os.makedirs(output_dir, exist_ok=True)

        self.loss_path = os.path.join(output_dir, f"{run_name}_loss.csv")
        self.snapshot_path = os.path.join(output_dir, f"{run_name}_progress.csv")
        self.config_path = os.path.join(output_dir, f"{run_name}_config.json")

        self._loss_header_written = os.path.exists(self.loss_path)
        self._snapshot_header_written = os.path.exists(self.snapshot_path)

    # ------------------------------------------------------------
    # Configuración / hiperparámetros
    # ------------------------------------------------------------
    def dump_config(self, config_module, extra=None):
        """
        Guarda todas las constantes en mayúsculas de un módulo (p.ej. constants.py)
        como JSON, junto con metadata de cuándo se lanzó la corrida.
        `extra` permite añadir datos adicionales (p.ej. nombre del step, episodios).
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

    # ------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------
    def log_loss(self, episode, replayed_count, player, network, loss_value):
        if loss_value is None:
            return
        record = LossRecord(episode, replayed_count, player, network, float(loss_value))
        self._append_csv(self.loss_path, record, self._loss_header_written)
        self._loss_header_written = True

    # ------------------------------------------------------------
    # Snapshots periódicos de progreso
    # ------------------------------------------------------------
    def log_snapshot(self, episode, player1, player2, stats):
        partidas = max(stats.partidas, 1)
        record = SnapshotRecord(
            episode=episode,
            epsilon_turn_p1=player1.epsilon_turn,
            epsilon_turn_p2=player2.epsilon_turn,
            epsilon_sel_p1=player1.epsilon_sel,
            epsilon_sel_p2=player2.epsilon_sel,
            p1_winrate=stats.p1_victories / partidas * 100,
            p2_winrate=stats.p2_victories / partidas * 100,
            drawrate=stats.empates / partidas * 100,
            p1_damage_avg=stats.p1_damage / partidas,
            p2_damage_avg=stats.p2_damage / partidas,
            avg_turns=stats.total_turns / partidas,
            p1_reward_avg=stats.total_reward_p1 / partidas,
            p2_reward_avg=stats.total_reward_p2 / partidas,
        )
        self._append_csv(self.snapshot_path, record, self._snapshot_header_written)
        self._snapshot_header_written = True

    # ------------------------------------------------------------
    # Utilidad interna
    # ------------------------------------------------------------
    @staticmethod
    def _append_csv(path, record, header_written):
        row = asdict(record)
        write_header = not header_written
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def plot_progress(self, show=True):
        """Genera gráficas a partir de los CSV de progreso y loss, y las guarda como PNG."""
        import matplotlib.pyplot as plt

        fig_paths = []

        if os.path.exists(self.snapshot_path):
            rows = self._read_csv(self.snapshot_path)
            if rows:
                episodes = [int(r["episode"]) for r in rows]
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle(f"Progreso — {self.run_name}")

                axes[0, 0].plot(episodes, [float(r["p1_winrate"]) for r in rows], label="P1")
                axes[0, 0].plot(episodes, [float(r["p2_winrate"]) for r in rows], label="P2")
                axes[0, 0].plot(episodes, [float(r["drawrate"]) for r in rows], label="Empates", linestyle="--")
                axes[0, 0].set_title("Winrate (%)")
                axes[0, 0].legend()

                axes[0, 1].plot(episodes, [float(r["p1_damage_avg"]) for r in rows], label="P1")
                axes[0, 1].plot(episodes, [float(r["p2_damage_avg"]) for r in rows], label="P2")
                axes[0, 1].set_title("Daño medio por partida")
                axes[0, 1].legend()

                axes[1, 0].plot(episodes, [float(r["p1_reward_avg"]) for r in rows], label="P1")
                axes[1, 0].plot(episodes, [float(r["p2_reward_avg"]) for r in rows], label="P2")
                axes[1, 0].set_title("Reward media por partida")
                axes[1, 0].legend()

                axes[1, 1].plot(episodes, [float(r["epsilon_turn_p1"]) for r in rows], label="epsilon_turn P1")
                axes[1, 1].plot(episodes, [float(r["epsilon_sel_p1"]) for r in rows], label="epsilon_sel P1")
                axes[1, 1].set_title("Epsilon")
                axes[1, 1].legend()

                plt.tight_layout()
                path = os.path.join(self.output_dir, f"{self.run_name}_progress.png")
                fig.savefig(path)
                fig_paths.append(path)

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
                                label=player, alpha=0.6,
                            )
                    ax.set_title(f"Loss — {network}")
                    ax.legend()
                plt.tight_layout()
                path = os.path.join(self.output_dir, f"{self.run_name}_loss.png")
                fig.savefig(path)
                fig_paths.append(path)

        if show and fig_paths:
            plt.show()

        return fig_paths

    @staticmethod
    def _read_csv(path):
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv_module.DictReader(f))