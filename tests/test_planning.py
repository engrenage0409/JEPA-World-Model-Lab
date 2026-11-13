import torch
from jepa.models import ActionDynamics
from jepa.planning import cem_plan

def test_cem_shapes():
    torch.manual_seed(1)
    dynamics=ActionDynamics(dim=16)
    actions,stats=cem_plan(dynamics,torch.zeros(16),torch.ones(16),
                           horizon=3,candidates=12,elite=3,iterations=2)
    assert actions.shape==(3,3)
    assert torch.isfinite(actions).all()
    assert stats["best_terminal_latent_mse"]>=0
