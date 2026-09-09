"""
Micro-benchmark aislado para torch.compile sobre TurnNetwork/SelectionNetwork.
Ejecutar de forma independiente al entrenamiento completo (NO usar torch.profiler
sobre el pipeline completo, ver diagnóstico de bloqueo documentado en el proyecto).

Uso: python benchmarks/compile_benchmark.py
"""
import torch
from torch.utils.benchmark import Timer

from AI.Agent.turnNetwork import TurnNetwork
from AI.Agent.selectionNetwork import SelectionNetwork
import constants


def _bench(label, fn, num_threads=6):
    timer = Timer(stmt="fn()", globals={"fn": fn}, num_threads=num_threads)
    result = timer.blocked_autorange(min_run_time=2.0)
    print(f"{label}: median={result.median*1000:.3f} ms | iqr={result.iqr*1000:.3f} ms")


def benchmark_turn_network(batch_size: int, compiled: bool):
    net = TurnNetwork(sigma_init=constants.NOISY_SIGMA_INIT)
    net.train()
    if compiled:
        net = torch.compile(net, mode="reduce-overhead", fullgraph=False)

    x = torch.randn(batch_size, constants.TURN_STATE_DIM)
    action_mask = torch.ones(batch_size, 3, 6, dtype=torch.bool)

    def step():
        net.reset_noise()
        with torch.no_grad():
            net(x, action_mask=action_mask)

    # Warm-up: la primera llamada compilada incluye el tiempo de compilación,
    # nunca debe medirse como representativa.
    for _ in range(5):
        step()

    label = f"TurnNetwork batch={batch_size} compiled={compiled}"
    _bench(label, step)


def diagnose_graph_breaks(batch_size: int):
    net = TurnNetwork(sigma_init=constants.NOISY_SIGMA_INIT)
    net.train()
    x = torch.randn(batch_size, constants.TURN_STATE_DIM)
    action_mask = torch.ones(batch_size, 3, 6, dtype=torch.bool)
    try:
        explanation = torch._dynamo.explain(net)(x, action_mask)
        print(explanation)
    except Exception as e:
        print(f"No se pudo generar explain(): {e}")


if __name__ == "__main__":
    torch.set_num_threads(6)

    print("=== Diagnóstico de graph breaks (TurnNetwork, batch=128) ===")
    diagnose_graph_breaks(batch_size=128)

    print("\n=== Benchmark inferencia (N=2048) ===")
    benchmark_turn_network(batch_size=2048, compiled=False)
    benchmark_turn_network(batch_size=2048, compiled=True)

    print("\n=== Benchmark replay (BATCH_SIZE=128) ===")
    benchmark_turn_network(batch_size=constants.BATCH_SIZE, compiled=False)
    benchmark_turn_network(batch_size=constants.BATCH_SIZE, compiled=True)