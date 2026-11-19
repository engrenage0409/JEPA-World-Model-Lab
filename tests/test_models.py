import torch
from jepa.models import MaskedJEPA,ActionDynamics,random_patch_mask

def test_masked_gradients_and_ema():
    torch.manual_seed(0)
    model=MaskedJEPA(dim=32,depth=1)
    frame=torch.randn(2,3,96,96)
    mask=random_patch_mask(2,36,0.5,frame.device)
    before=next(model.target.parameters()).clone()
    loss,pred,target=model(frame,frame,mask)
    assert pred.shape==(2,36,32)
    assert target.shape==(2,36,32)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.predictor.head.weight.grad is not None
    assert all(p.grad is None for p in model.target.parameters())
    with torch.no_grad(): next(model.encoder.parameters()).add_(1.)
    model.update_target(0.9)
    assert not torch.allclose(next(model.target.parameters()),before)

def test_action_rollout():
    model=ActionDynamics(dim=32)
    pred=model.rollout(torch.randn(2,32),torch.randn(2,4,3))
    assert pred.shape==(2,4,32)
