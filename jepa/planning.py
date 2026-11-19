"""CEM latent planning demo. Needs task-specific learned goal/reward model for control."""
import torch

@torch.no_grad()
def cem_plan(dynamics,start,goal,horizon=6,candidates=128,elite=16,iterations=5,
             action_dim=3,seed=42):
    if not (0<elite<=candidates and horizon>0 and iterations>0):
        raise ValueError("Invalid CEM settings")
    if start.ndim!=1 or goal.shape!=start.shape:
        raise ValueError("start and goal must be [D] latent vectors")
    generator=torch.Generator(device="cpu").manual_seed(seed)
    device=start.device
    mean=torch.zeros(horizon,action_dim,device=device)
    std=torch.ones_like(mean)*0.5
    best=None
    for _ in range(iterations):
        samples=mean+std*torch.randn(candidates,horizon,action_dim,generator=generator).to(device)
        # CarRacing continuous controls: steering [-1,1], gas/brake [0,1].
        steer=samples[...,0:1].clamp(-1,1)
        throttle=samples[...,1:].clamp(0,1)
        samples=torch.cat([steer,throttle],dim=-1)
        prediction=dynamics.rollout(start.expand(candidates,-1),samples)[:,-1]
        costs=(prediction-goal).square().mean(-1)
        chosen=costs.topk(elite,largest=False).indices
        elites=samples[chosen]
        mean=elites.mean(0)
        std=elites.std(0,unbiased=False).clamp(min=0.03)
        best=float(costs.min().item())
    return mean,{"best_terminal_latent_mse":best,
                 "warning":"CEM optimizes arbitrary supplied latent goal, not environment reward."}
