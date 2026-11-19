"""Unit tests for the visual/state residual merger (no simulator needed)."""
import torch

from scripts.visual_medium_bridge import fuse_latents


def test_alpha_zero_keeps_exact_teacher_state():
    teacher = torch.randn(3, 128)
    visual = torch.randn(3, 128) * 10
    fused = fuse_latents(teacher, visual, alpha=0, max_ratio=0.5)
    assert torch.equal(fused, teacher)


def test_uncapped_alpha_one_equals_visual():
    state = torch.randn(3, 128)
    visual = torch.randn(3, 128)
    fused = fuse_latents(state, visual, alpha=1, max_ratio=1e6)
    assert torch.allclose(fused, visual, atol=1e-6)


def test_visual_residual_is_limited():
    state = torch.ones(2, 64)
    visual = torch.ones(2, 64) * 100
    alpha, ratio = 0.2, 0.5
    fused = fuse_latents(state, visual, alpha=alpha, max_ratio=ratio)
    residual_rms = (fused - state).square().mean(dim=-1).sqrt()
    state_rms = state.square().mean(dim=-1).sqrt()
    assert torch.all(residual_rms <= alpha * ratio * state_rms + 1e-6)


def test_wrong_shapes_raise():
    try:
        fuse_latents(torch.zeros(1, 3), torch.zeros(1, 4), alpha=0.2)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a shape mismatch error")
