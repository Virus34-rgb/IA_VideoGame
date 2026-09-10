"""
Verifica que StatsReportWriter produce el mismo formato/valores que el
StatsV monolítico original, usando un StatsAccumulator poblado de forma
determinista con datos sintéticos.
"""
import os
import tempfile
import torch

from AI.Logging.stats_accumulator import StatsAccumulator
from AI.Logging.stats_report_writer import StatsReportWriter
from AI.Environment.warriorFactory import get_warriors_classes


def _populate_deterministic(stats: StatsAccumulator, N: int = 4):
    stats.start_batch(N)
    damage_p1 = torch.tensor([10.0, 5.0, 0.0, 20.0])
    damage_p2 = torch.tensor([8.0, 3.0, 0.0, 15.0])
    zeros = torch.zeros(N)
    stats.accumulate_turn(
        damage_p1, damage_p2, zeros, zeros, zeros, zeros, zeros, zeros,
        zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros,
        torch.zeros(N, dtype=torch.bool),
    )
    termina_ahora = torch.tensor([True, True, False, True])
    winner = torch.tensor([0, 1, -1, 2])
    p1_deaths = torch.tensor([3, 1, 0, 2])
    p2_deaths = torch.tensor([1, 3, 0, 2])
    turn_number = torch.tensor([5, 8, 0, 20])
    por_muerte = torch.tensor([True, True, False, False])
    por_turnos = torch.tensor([False, False, False, True])
    stats.close_finished_games(termina_ahora, winner, p1_deaths, p2_deaths, turn_number, por_muerte, por_turnos)
    stats.total_reward_p1 = -12.5
    stats.total_reward_p2 = 34.2


def test_write_produces_valid_file_with_expected_values():
    stats = StatsAccumulator()
    _populate_deterministic(stats)
    warriors_classes = get_warriors_classes()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "stats_test.txt")
        StatsReportWriter.write(path, stats, warriors_classes, p1_elo=1050.0, p2_elo=980.0, pool_elos={1: 1100.0, 2: 950.0})

        assert os.path.exists(path)
        content = open(path, encoding="utf-8").read()

    # Valores deterministas conocidos -> deben aparecer literalmente en el texto
    assert "Partidas:                  3" in content   # 3 partidas cerradas (termina_ahora True x3)
    assert "Victorias P1:              1" in content   # winner==0 solo en la primera
    assert "Victorias P2:              1" in content   # winner==1 en la segunda
    assert "Elo P1:                    1050.0" in content
    assert "Snapshots en la pool:      2" in content


def test_summary_matches_accumulator_state():
    stats = StatsAccumulator()
    _populate_deterministic(stats)
    summary = stats.build_summary()

    assert summary.partidas == 3
    assert summary.p1_victories == 1
    assert summary.p2_victories == 1
    assert summary.total_reward_p1 == -12.5


def test_reset_clears_all_state():
    stats = StatsAccumulator()
    _populate_deterministic(stats)
    stats.reset()
    assert stats.partidas == 0
    assert stats.p1_victories == 0
    assert stats.total_reward_p1 == 0.0