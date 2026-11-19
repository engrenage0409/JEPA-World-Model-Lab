"""Train a Mini-JEPA-compatible goal-conditioned low-level rectified-flow walker.

Reuses upstream jepa_robotics networks, pretrained-independent: load YOUR
self-trained State JEPA, train flow from scratch on your collected trajectories.
The original author's directed Flow has a different curated training setup;
this is a reproducible replacement with the same eval checkpoint interface.

Hindsight goal relabel: for each t select achieved position at t+h within
the same episode, condition the action chunk on that future waypoint.
Never copies actions at inference.

Example:
  python scripts/train_mini_medium_low_flow.py \
    --jepa outputs/mini_scratch/full/point_medium_jepa_model.pt \
    --episodes-npz outputs/mini_scratch/medium_episodes.npz \
    --out outputs/mini_scratch/full/medium_low_flow.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "mini-jepa"))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn.functional as F

from jepa_robotics.algos.control.flow import build_flow_net
from jepa_robotics.data import load_episodes_npz
from jepa_robotics.evaluate import load_jepa_artifact


def make_samples(episodes, spec, norm, chunk, horizon, seed):
    rng = np.random.default_rng(seed)
    states, chunks = [], []
    # All goals are future achieved states in the SAME episode.
    a, b = spec.obs_dim, spec.obs_dim + spec.goal_dim
    c, d = b, b + spec.goal_dim
    for ep in episodes:
        s, actions = ep.states.astype(np.float32), ep.actions.astype(np.float32)
        if len(actions) < chunk + 1:
            continue
        for t in range(0, len(actions) - chunk + 1, 2):
            h = int(rng.integers(chunk, max(chunk + 1, min(horizon, len(actions) - t) + 1)))
            tf = min(len(s) - 1, t + h)
            state = s[t].copy()
            state[c:d] = s[tf, a:b]
            states.append(state)
            chunks.append(actions[t:t + chunk].reshape(-1))
    if not states:
        raise ValueError("No valid action chunks. Collect longer maze trajectories.")
    return norm.encode(np.asarray(states, np.float32)), np.asarray(chunks, np.float32)


@torch.no_grad()
def build_conditions(model, states, device, batch=1024):
    out = []
    for k in range(0, len(states), batch):
        s = torch.from_numpy(states[k:k + batch]).to(device)
        z = model.encode(s)
        out.append(torch.cat((z, s), dim=-1).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jepa", type=Path, required=True)
    ap.add_argument("--episodes-npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--goal-horizon", type=int, default=90)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    wm, norm, spec, cfg = load_jepa_artifact(args.jepa, dev)
    wm.eval().requires_grad_(False)
    if spec.goal_dim != 2 or spec.action_dim != 2:
        raise ValueError(f"This trainer expects PointMaze 2D goal/actions; got {spec}")
    episodes = load_episodes_npz(args.episodes_npz)
    if len(episodes) < 3:
        raise ValueError("Need at least 3 episodes")
    # Episode-level validation split; no time-adjacent leakage.
    n_val = max(1, len(episodes) // 10)
    train_s, train_a = make_samples(episodes[:-n_val], spec, norm, args.chunk, args.goal_horizon, args.seed)
    valid_s, valid_a = make_samples(episodes[-n_val:], spec, norm, args.chunk, args.goal_horizon, args.seed+1)
    cond_train = build_conditions(wm, train_s, dev)
    cond_valid = build_conditions(wm, valid_s, dev)
    actions_train = torch.from_numpy(train_a)
    actions_valid = torch.from_numpy(valid_a)
    del train_s, valid_s
    net = build_flow_net(args.chunk * spec.action_dim, cond_train.shape[-1],
                         args.hidden, architecture="mlp").to(dev)
    optim = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    best = float("inf")
    print(json.dumps({"event": "low_flow_data", "train_chunks": len(cond_train),
                      "valid_chunks": len(cond_valid), "cond_dim": cond_train.shape[-1],
                      "chunk": args.chunk}), flush=True)

    def flow_loss(cond, target):
        noise = torch.randn_like(target)
        t = torch.rand(target.shape[0], 1, device=dev)
        xt = (1 - t) * noise + t * target
        return F.mse_loss(net(xt, t, cond), target - noise)

    for step in range(1, args.train_steps+1):
        net.train()
        idx = torch.randint(0, len(cond_train), (args.batch_size,))
        x = actions_train[idx].to(dev)
        z = cond_train[idx].to(dev)
        loss = flow_loss(z, x)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optim.step()
        if step == 1 or step % 500 == 0 or step == args.train_steps:
            net.eval()
            with torch.no_grad():
                # Fixed RNG for comparable validation-noise estimates.
                rng_state = torch.get_rng_state()
                cuda_state = torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None
                torch.manual_seed(9000)
                if dev.type == "cuda":
                    torch.cuda.manual_seed_all(9000)
                v_losses = []
                for start in range(0, min(len(cond_valid), 2048), 256):
                    v = flow_loss(cond_valid[start:start+256].to(dev),
                                  actions_valid[start:start+256].to(dev))
                    v_losses.append(v.item())
                valid = float(np.mean(v_losses))
                torch.set_rng_state(rng_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state(cuda_state, dev)
            print(json.dumps({"event": "low_flow_train", "step": step,
                              "train_loss": float(loss.item()), "valid_loss": valid}), flush=True)
            if valid < best:
                best = valid
                args.out.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "flow": net.state_dict(),
                    "config": {
                        "chunk": args.chunk, "action_dim": spec.action_dim,
                        "chunk_dim": args.chunk * spec.action_dim,
                        "cond_dim": int(cond_train.shape[-1]), "hidden": args.hidden,
                        "concat_raw": True, "flow_architecture": "mlp",
                        "flow_blocks": 4, "emphasis_repeat": 0,
                        "progress_cond": False,
                        "train_source": "from-scratch flow with hindsight relabeled future achieved goals",
                    },
                    "best_valid_flow_loss": best,
                    "jepa_path": str(args.jepa),
                }, args.out)
    print(json.dumps({"event": "low_flow_saved", "path": str(args.out),
                      "best_valid_loss": best}), flush=True)


if __name__ == "__main__":
    main()
