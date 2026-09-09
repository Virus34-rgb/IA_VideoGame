"""
Test de equivalencia numérica para la cache de NoisyLinear.
Verifica que forward() con cache produce el mismo resultado que el cálculo
directo, y que la cache se invalida correctamente tras reset_noise()/clamp_sigma().
"""
import torch
from AI.Agent.noisy_linear import NoisyLinear


def test_weight_matches_manual_computation_training_mode():
    layer = NoisyLinear(10, 5, std_init=0.5)
    layer.train()
    expected_weight = layer.weight_mu + layer.weight_sigma * layer.weight_epsilon
    expected_bias = layer.bias_mu + layer.bias_sigma * layer.bias_epsilon
    assert torch.allclose(layer.weight, expected_weight)
    assert torch.allclose(layer.bias, expected_bias)


def test_weight_matches_manual_computation_eval_mode():
    layer = NoisyLinear(10, 5, std_init=0.5)
    layer.eval()
    layer._refresh_cache()  # simula lo que hace TurnNetwork/PlayerAIV al poner eval()
    assert torch.allclose(layer.weight, layer.weight_mu)
    assert torch.allclose(layer.bias, layer.bias_mu)


def test_cache_changes_after_reset_noise():
    layer = NoisyLinear(10, 5, std_init=0.5)
    layer.train()
    w1 = layer.weight.clone()
    torch.manual_seed(123)
    layer.reset_noise()
    w2 = layer.weight.clone()
    assert not torch.allclose(w1, w2), "El peso debería cambiar tras reset_noise() (nuevo ruido)"


def test_cache_updates_after_clamp_sigma():
    layer = NoisyLinear(10, 5, std_init=0.5)
    layer.train()
    layer.weight_sigma.data.fill_(0.001)  # fuerza sigma por debajo del min_sigma
    layer.clamp_sigma(min_sigma=0.05)
    expected_weight = layer.weight_mu + layer.weight_sigma * layer.weight_epsilon
    assert torch.allclose(layer.weight, expected_weight), (
        "Tras clamp_sigma, la cache debe reflejar el sigma recortado, no el anterior"
    )


def test_forward_output_matches_uncached_reference():
    """Compara la salida completa de forward() contra una implementación
    de referencia que no usa cache (recalcula manualmente cada vez)."""
    layer = NoisyLinear(8, 4, std_init=0.5)
    layer.train()
    x = torch.randn(16, 8)

    out_cached = layer(x)

    ref_weight = layer.weight_mu + layer.weight_sigma * layer.weight_epsilon
    ref_bias = layer.bias_mu + layer.bias_sigma * layer.bias_epsilon
    out_reference = torch.nn.functional.linear(x, ref_weight, ref_bias)

    assert torch.allclose(out_cached, out_reference, atol=1e-6)


def test_eval_mode_ignores_noise():
    layer = NoisyLinear(8, 4, std_init=0.5)
    layer.eval()
    layer._refresh_cache()
    x = torch.randn(16, 8)
    out1 = layer(x)
    layer.reset_noise()  # cambia epsilon, pero en eval no debería importar
    layer._refresh_cache()
    out2 = layer(x)
    assert torch.allclose(out1, out2), "En eval, el ruido no debe afectar la salida"