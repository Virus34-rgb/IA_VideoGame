"""
Verifica que TurnNetwork compilada produce salidas numéricamente equivalentes
a la versión sin compilar, para las mismas entradas y el mismo ruido.
"""
import torch
from AI.Agent.turnNetwork import TurnNetwork
import constants


def test_compiled_output_matches_eager():
    torch.manual_seed(42)
    net_eager = TurnNetwork(sigma_init=constants.NOISY_SIGMA_INIT)
    net_eager.eval()   # eval: sin ruido, elimina la fuente de aleatoriedad para la comparación
    net_eager._sync_noise_free = True  # no-op, solo documenta la intención

    net_compiled = torch.compile(
        TurnNetwork(sigma_init=constants.NOISY_SIGMA_INIT), mode="reduce-overhead", fullgraph=False,
    )
    net_compiled.eval()
    net_compiled.load_state_dict(net_eager.state_dict())

    x = torch.randn(16, constants.TURN_STATE_DIM)
    action_mask = torch.ones(16, 3, 6, dtype=torch.bool)

    with torch.no_grad():
        out_eager = net_eager(x, action_mask=action_mask)
        out_compiled = net_compiled(x, action_mask=action_mask)

    assert torch.allclose(out_eager, out_compiled, atol=1e-4), (
        "La salida compilada difiere de la eager más allá de la tolerancia esperada por redondeo"
    )
    
    if __name__ == "__main__":
        test_compiled_output_matches_eager()
        print("✅ Test de equivalencia pasado correctamente.")