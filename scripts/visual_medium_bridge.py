"""Visual-to-HWM bridge for the existing Meta JEPA-WM + Mini-JEPA project.

Stages:
  collect: run the known working Mini-JEPA Medium controller and save synchronized
           RGB frames and the teacher's state-JEPA latent vectors.
  train:   freeze the EXISTING Meta JEPA-WM encoder and learn a small projection
           into Mini-JEPA's latent space.
  eval:    replace the HIGH-LEVEL state latent with the projected visual latent.
           The Mini low-level walker remains state-based.

This is an experimental hybrid, NOT a full end-to-end visual world-model policy.
Run from JEPA-World-Model-Lab root. No existing source files are modified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "external" / "mini-jepa"
META = ROOT / "external" / "jepa-wms"
if str(MINI) not in sys.path:
    sys.path.insert(0, str(MINI))

from jepa_robotics.envs import make_env, flatten_obs
from jepa_robotics.tasks import resolve_task
from jepa_robotics.algos.maze_low_level import LowLevelFlow

# Reuse the OFFICIAL Mini-JEPA planning code; no reimplementation of macro-flow.
sys.path.insert(0, str(MINI / "scripts" / "eval"))
from eval_hjepa_hwm import load_hwm, load_macro_flow, flow_macro_subgoal


def paths():
    ck = MINI / "runs" / "point_medium" / "checkpoints"
    return dict(
        jepa=ck / "point_medium_jepa_model.pt",
        hwm=ck / "point_medium_hwm_s30.pt",
        macro=ck / "point_medium_hwm_macroflow.pt",
        low=ck / "point_medium_flow_directed.pt",
    )


class VisualBridge(nn.Module):
    """Map pooled Meta image JEPA features to Mini-JEPA state latent space."""

    def __init__(self, input_dim: int, target_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, target_dim),
        )

    def forward(self, x):
        return self.net(x)


def open_meta(device):
    if not META.exists():
        raise FileNotFoundError(f"Meta source missing: {META}")
    model, _ = torch.hub.load(
        str(META), "jepa_wm_pointmaze", source="local",
        pretrained=True, device=str(device),
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.inference_mode()
def meta_features(meta, frames: np.ndarray, device, batch_size=2):
    """Input RGB uint8 [N,H,W,3]; Meta.encode accepts [B,T,C,H,W]."""
    outs = []
    for start in range(0, len(frames), batch_size):
        chunk = frames[start:start + batch_size]
        tensor = torch.from_numpy(np.ascontiguousarray(chunk))
        tensor = tensor.permute(0, 3, 1, 2).unsqueeze(1).to(device)
        latent = meta.encode(tensor)
        if isinstance(latent, dict):
            latent = latent["visual"]
        # Preserve maze location: Meta [B,T,V,H,W,D] -> spatial 4x4 grid.
        # Pool only time and views, NEVER discard spatial coordinates.
        if latent.ndim != 6:
            raise ValueError(f"Expected Meta tokens [B,T,V,H,W,D], got {tuple(latent.shape)}")
        spatial = latent.float().mean(dim=(1, 2)).permute(0, 3, 1, 2)
        feature = F.adaptive_avg_pool2d(spatial, (4, 4)).flatten(1)
        outs.append(feature.cpu().numpy())
    return np.concatenate(outs, axis=0)


def make_components(device):
    for name, path in paths().items():
        if not path.is_file() or path.stat().st_size < 100_000:
            raise RuntimeError(f"Missing real checkpoint ({name}): {path}")
    task = resolve_task("point_medium", None)
    env = make_env(task.env_id, seed=20000, max_episode_steps=task.max_episode_steps,
                   render_mode="rgb_array", width=320, height=240)
    low = LowLevelFlow(str(paths()["jepa"]), str(paths()["low"]),
                       env.action_space.low, env.action_space.high,
                       device=str(device))
    psi, _, g, dec, _ = load_hwm(str(paths()["hwm"]), device)
    macro_flow = load_macro_flow(str(paths()["macro"]), device)
    macro_flow[2]["n_samples"] = 16
    macro_flow[2]["horizon"] = 1
    return task, env, low, psi, g, dec, macro_flow


def teacher_latent(low, obs, device):
    x = torch.from_numpy(low.norm.encode(flatten_obs(obs))).unsqueeze(0).to(device)
    with torch.inference_mode():
        return low.model.encode(x)


def run_controller(args, mode, bridge=None, meta=None):
    task, env, low, psi, g, dec, (fnet, ddpm, config) = make_components(args.device)
    frames, previous_frames, labels, episode_ids, successes = [], [], [], [], []
    video_frames = []
    # The flow sampler is stochastic. Seeds are fixed for repeatable evaluation.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        goal = np.asarray(obs["desired_goal"], np.float32)
        subgoal = goal.copy()
        needs_replan, elapsed = True, 0
        terminated = truncated = False
        info = {}
        step = 0
        initial_image = np.asarray(env.render(), dtype=np.uint8)
        history = deque([initial_image] * (args.history + 1), maxlen=args.history + 1)
        while not (terminated or truncated):
            current_image = np.asarray(env.render(), dtype=np.uint8)
            prev_image = history[0]
            if mode == "eval" and ep == 0 and args.video_out and step % 2 == 0:
                video_frames.append(np.asarray(env.render(), dtype=np.uint8))
            if mode == "collect" and step % args.every == 0:
                frames.append(current_image.copy())
                previous_frames.append(prev_image.copy())
                labels.append(teacher_latent(low, obs, args.device)
                              .squeeze(0).cpu().numpy())
                episode_ids.append(ep)

            achieved = np.asarray(obs["achieved_goal"], np.float32)
            if (needs_replan or
                np.linalg.norm(achieved - subgoal) < args.reach_radius or
                elapsed >= args.low_timeout):
                if mode == "eval":
                    current_features = meta_features(meta, current_image[None], args.device)
                    previous_features = meta_features(meta, prev_image[None], args.device)
                    features = np.concatenate((current_features, previous_features), axis=1)
                    with torch.inference_mode():
                        z = bridge(torch.from_numpy(features).to(args.device))
                else:
                    z = teacher_latent(low, obs, args.device)
                with torch.inference_mode():
                    z_high = psi(z)
                    subgoal = flow_macro_subgoal(
                        fnet, ddpm, g, dec, z_high, goal, args.device,
                        n_samples=config.get("n_samples", 16),
                        flow_steps=int(ddpm["T"]),
                        horizon=config.get("horizon", 1),
                    )
                if np.linalg.norm(subgoal - goal) < args.reach_radius:
                    subgoal = goal
                elapsed, needs_replan = 0, False
            action = low.act(obs, subgoal)
            obs, _, terminated, truncated, info = env.step(action)
            history.append(current_image)
            step += 1
            elapsed += 1
        success = float(info.get("is_success", info.get("success", 0.0)))
        successes.append(success)
        print(json.dumps({"episode": ep, "seed": args.seed + ep,
                          "steps": step, "success": success}), flush=True)
    env.close()
    if video_frames:
        import imageio.v2 as imageio
        Path(args.video_out).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video_out, video_frames, fps=30)
        print(json.dumps({"video_saved": args.video_out}), flush=True)
    print(json.dumps({"mode": mode, "episodes": args.episodes,
                      "success_rate": float(np.mean(successes))}), flush=True)
    if mode == "collect":
        output = Path(args.data)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, frames=np.stack(frames),
                            prev_frames=np.stack(previous_frames),
                            target_latents=np.stack(labels),
                            episode_ids=np.array(episode_ids))
        print(f"Saved {len(frames)} aligned RGB/latent pairs to {output}")


def train_bridge(args):
    raw = np.load(args.data)
    images = raw["frames"]
    targets = np.asarray(raw["target_latents"], dtype=np.float32)
    if "prev_frames" not in raw:
        raise RuntimeError("Old dataset lacks prev_frames. Re-run collect with the updated script.")
    previous_images = raw["prev_frames"]
    episodes = raw["episode_ids"]
    if len(np.unique(episodes)) < 2:
        raise ValueError("Collect at least two episodes to split train and validation.")

    print(f"Encoding {len(images)} images using frozen Meta JEPA-WM...")
    meta = open_meta(args.device)
    features = meta_features(meta, images, args.device, args.batch_size)
    previous_features = meta_features(meta, previous_images, args.device, args.batch_size)
    features = np.concatenate([features, previous_features], axis=1)
    del meta
    if args.device.type == "cuda":
        torch.cuda.empty_cache()

    # Split by WHOLE episodes to avoid frame leakage across validation.
    held_out = np.unique(episodes)[-max(1, len(np.unique(episodes)) // 5):]
    valid = np.isin(episodes, held_out)
    train = ~valid
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    bridge = VisualBridge(x.shape[1], y.shape[1], args.hidden).to(args.device)
    opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=1e-4)
    best = float("inf")
    for epoch in range(args.epochs):
        bridge.train()
        permutation = np.random.permutation(np.where(train)[0])
        for pos in range(0, len(permutation), args.batch_size):
            ids = permutation[pos:pos + args.batch_size]
            pred = bridge(x[ids].to(args.device))
            target = y[ids].to(args.device)
            loss = F.mse_loss(pred, target) + 0.1 * (
                1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        bridge.eval()
        with torch.inference_mode():
            val_x = x[valid].to(args.device)
            val_y = y[valid].to(args.device)
            val_mse = F.mse_loss(bridge(val_x), val_y).item()
        print(f"epoch={epoch + 1} validation_mse={val_mse:.6f}", flush=True)
        if val_mse < best:
            best = val_mse
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": bridge.state_dict(),
                "input_dim": x.shape[1],
                "target_dim": y.shape[1],
                "hidden": args.hidden,
                "validation_mse": best,
                "teacher": "Mini-JEPA Medium state latent",
                "visual_encoder": "Meta JEPA-WM PointMaze (frozen)",
                "representation": "paired temporal spatial 4x4 Meta tokens",
            }, out)
    print(json.dumps({"bridge_saved": args.output, "best_val_mse": best}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["collect", "train", "eval"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data", default="outputs/visual_medium/paired_data.npz")
    parser.add_argument("--output", default="outputs/visual_medium/visual_bridge.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--video-out", default="outputs/visual_medium/visual_nav.mp4")
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--every", type=int, default=4)
    parser.add_argument("--history", type=int, default=4,
                        help="Number of past simulation steps for the second visual frame")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--reach-radius", type=float, default=1.0)
    parser.add_argument("--low-timeout", type=int, default=90)
    args = parser.parse_args()
    if args.history < 1:
        parser.error("--history must be >= 1")
    args.device = torch.device(args.device)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    if args.mode == "collect":
        run_controller(args, "collect")
    elif args.mode == "train":
        train_bridge(args)
    else:
        ck = torch.load(args.output, map_location="cpu", weights_only=True)
        bridge = VisualBridge(ck["input_dim"], ck["target_dim"], ck["hidden"])
        bridge.load_state_dict(ck["model"])
        bridge.to(args.device).eval()
        meta = open_meta(args.device)
        run_controller(args, "eval", bridge, meta)


if __name__ == "__main__":
    main()
