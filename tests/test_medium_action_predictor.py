"""Fast unit checks for action-batch shape and differentiability."""
import torch

from scripts.medium_action_predictor import (
    STRIDE, ActionAdapter, PositionHead, normalize_actions,
)


def test_action_adapter_starts_as_identity():
    model = ActionAdapter()
    actions = torch.randn(3, STRIDE, 2)
    assert torch.allclose(model(actions), actions, atol=1e-6)


def test_action_adapter_has_gradient():
    model = ActionAdapter()
    x = torch.ones(2, STRIDE, 2)
    model(x).square().sum().backward()
    assert model.delta.weight.grad is not None


def test_reject_incorrect_stride():
    try:
        ActionAdapter()(torch.zeros(2, STRIDE - 1, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected stride mismatch")


def test_position_head_accepts_meta_visual_token_shapes():
    h = PositionHead(384)
    z = torch.randn(2, 1, 1, 16, 16, 384)
    assert h(z).shape == (2, 2)
    assert h(z[:, 0]).shape == (2, 2)


def test_action_normalization_broadcasts_over_stride():
    actions = torch.ones(2, STRIDE, 2)
    mean = torch.tensor([0.2, -0.1])
    std = torch.tensor([0.5, 0.25])
    actual = normalize_actions(actions, mean, std)
    expected = torch.tensor([1.6, 4.4])
    assert torch.allclose(actual, expected.expand_as(actual))
