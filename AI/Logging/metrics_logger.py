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
    player: str
    network: str
    loss: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class SnapshotRecord:
    episode: int
    epsilon_turn_p1: float | None
    epsilon_turn_p2: float | None
    epsilon_sel_p1: float | None
    epsilon_sel_p2: float | None
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
    def __init__(self, output_dir, run_name):
        self.output_dir = output_dir
        self.run_name = run_name
        os.makedirs(output_dir, exist_ok=True)
        self.loss_path = os.path.join(output_dir, f"{run_name}_loss.csv")
        self.snapshot_path = os.path.join(output_dir, f"{run_name}_progress.csv")
        self.config_path = os.path.join(output_dir, f"{run_name}_config.json")
        self._loss_header_written = os.path.exists(self.loss_path)
        self._snapshot_header_written = os.path.exists(self.snapshot_path)

    def dump_config(self, config_module, extra=None):
        values = {
            name: getattr(config_module, name)
            for name in dir(config_module)
            if name.isupper() and not name.startswith("_")
        }
        payload = {"run_name": self.run_name, "timestamp": time.time(), "constants": values}
        if extra:
            payload["extra"] = extra
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    def log_loss(self, episode, replayed_count, player, network, loss_value):
        if loss_value is None:
            return
        record = LossRecord(episode, replayed_count, player, network, float(loss_value))
        self._append_csv(self.loss_path, record, self._loss_header_written)
        self._loss_header_written = True

    def log_snapshot(self, episode, player1, player2, stats):
        partidas = max(stats.partidas, 1)
        record = SnapshotRecord(
            episode=episode,
            epsilon_turn_p1=getattr(player1, "epsilon_turn", None),
            epsilon_turn_p2=getattr(player2, "epsilon_turn", None),
            epsilon_sel_p1=getattr(player1, "epsilon_sel", None),
            epsilon_sel_p2=getattr(player2, "epsilon_sel", None),
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

    # ------------------------------------------------------------
    # NUEVO: comparación de varios runs — funcionalidad acordada como
    # prioridad inmediatamente posterior a la vectorización.
    # ------------------------------------------------------------
    @staticmethod
    def compare_runs(runs, labels=None, show=True, output_dir=None):
        """
        Compara varios runs aunque cada uno esté almacenado en un
        directorio diferente.

        runs:
            Lista de tuplas (output_dir, run_name)

        labels:
            Nombres que aparecerán en las gráficas.
        """
        import matplotlib.pyplot as plt

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

            episodes = [
                int(r["episode"])
                for r in rows
            ]

            axes[0, 0].plot(
                episodes,
                [float(r["p1_winrate"]) for r in rows],
                label=label
            )

            axes[0, 1].plot(
                episodes,
                [float(r["p1_damage_avg"]) for r in rows],
                label=label
            )

            axes[1, 0].plot(
                episodes,
                [float(r["p1_reward_avg"]) for r in rows],
                label=label
            )

            axes[1, 1].plot(
                episodes,
                [float(r["avg_turns"]) for r in rows],
                label=label
            )

        axes[0, 0].set_title("Winrate P1 (%)")
        axes[0, 1].set_title("Daño medio P1")
        axes[1, 0].set_title("Reward medio P1")
        axes[1, 1].set_title("Turnos medios por partida")

        for ax in axes.flat:
            ax.set_xlabel("Lote")
            ax.legend()
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        path = os.path.join(output_dir,"comparison_progress.png")

        fig.savefig(path)

        if show and any_data:
            plt.show()

        return path