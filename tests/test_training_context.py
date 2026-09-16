"""
Verifica que TrainingContext captura correctamente los valores de constants
tras aplicar overrides, y que AgentTrainer respeta el context inyectado en
vez de leer constants directamente cuando se le proporciona.

Ejecutar: python -m pytest tests/test_training_context.py -v
"""
import torch
import pytest

import constants
from AI.Agent.training.training_context import TrainingContext
from AI.Agent.training.agent_trainer import AgentTrainer
from AI.Agent.turnNetwork import TurnNetwork
from AI.Agent.selectionNetwork import SelectionNetwork
from AI.Agent.replayMemoryAN import ReplayMemoryAN
from AI.Agent.replayMemoryPM import ReplayMemoryPM


def test_from_constants_snapshots_current_values():
    """TrainingContext.from_constants() debe capturar los valores ACTUALES
    de constants, no los del momento de import del módulo. Un override
    temporal de constants.BATCH_SIZE debe reflejarse en el context capturado."""
    original = constants.BATCH_SIZE
    try:
        constants.BATCH_SIZE = 512
        ctx = TrainingContext.from_constants()
        assert ctx.batch_size == 512
    finally:
        constants.BATCH_SIZE = original


def test_context_is_frozen():
    """El dataclass es frozen=True: los campos no se pueden reasignar."""
    ctx = TrainingContext(
        batch_size=256, turn_replays_per_batch=240,
        selection_replays_per_batch=72,
        rusher_aggression_band_weights=(0.25, 0.35, 0.40),
        copy_dqn_sel=50, copy_dqn_turn=150,
    )
    with pytest.raises(Exception):
        ctx.batch_size = 999  # type: ignore[misc]


def test_context_band_weights_are_tuple():
    """rusher_aggression_band_weights debe almacenarse como tuple para que
    el objeto sea realmente inmutable (frozen=True solo congela la referencia)."""
    ctx = TrainingContext.from_constants()
    assert isinstance(ctx.rusher_aggression_band_weights, tuple)


def _make_trainer(context=None):
    """Helper: construye un AgentTrainer mínimo con redes/buffers recién creados."""
    sel_net = SelectionNetwork(sigma_init=0.5, input_size=20)
    target_sel_net = SelectionNetwork(sigma_init=0.5, input_size=20)
    opt_sel = torch.optim.Adam(sel_net.parameters(), lr=0.0001)
    replay_sel = ReplayMemoryPM(1000, state_dim=20)

    turn_net = TurnNetwork(sigma_init=0.5, input_size=constants.TURN_STATE_DIM)
    target_turn_net = TurnNetwork(sigma_init=0.5, input_size=constants.TURN_STATE_DIM)
    opt_turn = torch.optim.Adam(turn_net.parameters(), lr=0.0001)
    replay_turn = ReplayMemoryAN(1000, state_dim=constants.TURN_STATE_DIM)

    return AgentTrainer(
        sel_net, target_sel_net, opt_sel, replay_sel,
        turn_net, target_turn_net, opt_turn, replay_turn,
        context=context,
    ), replay_sel


def test_agent_trainer_uses_context_batch_size_not_constants():
    """Con un batch_size del context MAYOR que el nº de muestras en el buffer,
    replay_selection() debe devolver None -- confirma que respeta el context
    y NO cae al batch_size de constants.py (que podría ser menor)."""
    context = TrainingContext(
        batch_size=5000, turn_replays_per_batch=10, selection_replays_per_batch=10,
        rusher_aggression_band_weights=(0.25, 0.35, 0.40),
        copy_dqn_sel=50, copy_dqn_turn=150,
    )
    trainer, replay_sel = _make_trainer(context=context)

    # 10 transiciones en el buffer, muy por debajo de context.batch_size=5000.
    states = torch.randn(10, 20)
    actions = torch.randint(0, 30, (10,))
    rewards = torch.randn(10)
    next_states = torch.randn(10, 20)
    dones = torch.zeros(10, dtype=torch.bool)
    replay_sel.push_batch(states, actions, rewards, next_states, dones)

    result = trainer.replay_selection()
    assert result is None, (
        "Con batch_size=5000 del context y solo 10 muestras, debe devolver None. "
        f"Devolvió: {result}"
    )


def test_agent_trainer_falls_back_to_constants_without_context():
    """Sin context inyectado, _batch_size()/_copy_dqn_sel()/_copy_dqn_turn()
    deben leer de constants.py (comportamiento previo, compatibilidad atrás)."""
    trainer, _ = _make_trainer(context=None)
    assert trainer._batch_size() == constants.BATCH_SIZE
    assert trainer._copy_dqn_sel() == constants.COPY_DQN_SEL
    assert trainer._copy_dqn_turn() == constants.COPY_DQN_TURN


def test_agent_trainer_uses_context_copy_dqn_not_constants():
    """Con context, _copy_dqn_sel()/_copy_dqn_turn() deben devolver los
    valores del context, no los de constants.py."""
    context = TrainingContext(
        batch_size=256, turn_replays_per_batch=240, selection_replays_per_batch=72,
        rusher_aggression_band_weights=(0.25, 0.35, 0.40),
        copy_dqn_sel=999, copy_dqn_turn=888,
    )
    trainer, _ = _make_trainer(context=context)
    assert trainer._copy_dqn_sel() == 999
    assert trainer._copy_dqn_turn() == 888