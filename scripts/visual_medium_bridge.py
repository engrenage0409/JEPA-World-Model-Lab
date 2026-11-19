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


def plan_subgoal(fnet, ddpm, g, dec, z_high, goal, device, config):
    """Use the same official Mini flow scorer for visual and teacher latents."""
    return flow_macro_subgoal(
        fnet, ddpm, g, dec, z_high, goal, device,
        n_samples=config.get("n_samples", 16),
        flow_steps=int(ddpm["T"]),
        horizon=config.get("horizon", 1),
    )


def snapshot_rng(device):
    """Capture both Torch and NumPy RNG so a shadow plan cannot alter the policy."""
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state(device).clone()
                if device.type == "cuda" else None,
        "numpy": np.random.get_state(),
    }


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)
    np.random.set_state(state["numpy"])


def fuse_latents(z_state, z_visual, alpha: float, max_ratio: float = 0.5):
    """Conservative residual interpolation into the pretrained Mini state space.

    alpha=0 reproduces the original Mini state latent exactly.
    The optional residual cap prevents a badly aligned visual embedding from
    taking control of the HWM in one step. No new weights are trained here.
    """
    if alpha == 0.0:
        return z_state
    if z_state.shape != z_visual.shape:
        raise ValueError(
            f"Visual/state latent mismatch: {tuple(z_visual.shape)} vs {tuple(z_state.shape)}"
        )
    residual = z_visual - z_state
    state_scale = z_state.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
    delta_scale = residual.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    cap = (max_ratio * state_scale / delta_scale).clamp(max=1.0)
    return z_state + alpha * residual * cap


def run_controller(args, mode, bridge=None, meta=None):
    task, env, low, psi, g, dec, (fnet, ddpm, config) = make_components(args.device)
    frames, previous_frames, labels, episode_ids, successes = [], [], [], [], []
    saved_video_types = set()
    comparisons = []
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
        # Record candidate episodes until one success and one failure are saved.
        record_episode = (mode == "eval" and bool(args.video_out)
                          and len(saved_video_types) < 2)
        episode_video_frames = []
        initial_image = np.asarray(env.render(), dtype=np.uint8)
        history = deque([initial_image] * (args.history + 1), maxlen=args.history + 1)
        while not (terminated or truncated):
            current_image = np.asarray(env.render(), dtype=np.uint8)
            prev_image = history[0]
            if record_episode and step % 2 == 0:
                episode_video_frames.append(current_image.copy())
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
                    with torch.inference_mode():
                        # Always retain the proven state JEPA representation.
                        z_state = teacher_latent(low, obs, args.device)
                        if args.fusion_alpha == 0.0:
                            # Exact Mini-state path, no Meta model needed.
                            z = z_state
                        else:
                            current_features = meta_features(meta, current_image[None], args.device)
                            previous_features = meta_features(meta, prev_image[None], args.device)
                            features = np.concatenate((current_features, previous_features), axis=1)
                            z_visual = bridge(torch.from_numpy(features).to(args.device))
                            z = fuse_latents(z_state, z_visual, args.fusion_alpha, args.fusion_max_ratio)
                else:
                    z = teacher_latent(low, obs, args.device)
                with torch.inference_mode():
                    z_high = psi(z)
                    # The selected (visual, state, or fused) planner controls the
                    # trajectory. Shadow teacher uses exactly the same observation.
                    # Restore the RNG after the shadow call to avoid consuming
                    # extra random samples in the actual rollout.
                    random_before = snapshot_rng(args.device) if args.compare_high_level and mode == "eval" else None
                    subgoal = plan_subgoal(fnet, ddpm, g, dec, z_high, goal, args.device, config)
                    if random_before is not None:
                        random_after = snapshot_rng(args.device)
                        try:
                            restore_rng(random_before, args.device)
                            z_teacher = teacher_latent(low, obs, args.device)
                            teacher_high = psi(z_teacher)
                            teacher_subgoal = plan_subgoal(
                                fnet, ddpm, g, dec, teacher_high, goal,
                                args.device, config,
                            )
                        finally:
                            restore_rng(random_after, args.device)
                        visual_xy = np.asarray(subgoal, dtype=np.float32).reshape(-1)
                        teacher_xy = np.asarray(teacher_subgoal, dtype=np.float32).reshape(-1)
                        row = {
                            "episode": ep,
                            "seed": args.seed + ep,
                            "step": step,
                            "achieved_xy": achieved.tolist(),
                            "goal_xy": goal.tolist(),
                            "visual_subgoal_xy": visual_xy.tolist(),
                            "teacher_subgoal_xy": teacher_xy.tolist(),
                            "subgoal_distance": float(np.linalg.norm(visual_xy - teacher_xy)),
                            "visual_latent_mse": float(F.mse_loss(z, z_teacher).item()),
                            "high_latent_mse": float(F.mse_loss(z_high, teacher_high).item()),
                            "high_latent_cosine": float(F.cosine_similarity(z_high, teacher_high, dim=-1).mean().item()),
                        }
                        comparisons.append(row)
                        if args.log_comparisons:
                            print(json.dumps({"event": "high_level_comparison", **row}), flush=True)
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
        if record_episode:
            category = "success" if success > 0.5 else "failure"
            if category not in saved_video_types and episode_video_frames:
                import imageio.v2 as imageio
                video_base = Path(args.video_out)
                video_path = video_base.with_name(
                    f"{video_base.stem}_{category}{video_base.suffix or '.mp4'}"
                )
                video_path.parent.mkdir(parents=True, exist_ok=True)
                final_frame = np.asarray(env.render(), dtype=np.uint8)
                episode_video_frames.append(final_frame.copy())
                imageio.mimsave(str(video_path), episode_video_frames, fps=30)
                saved_video_types.add(category)
                print(json.dumps({"event": "video_saved", "category": category,
                                  "episode": ep, "seed": args.seed + ep,
                                  "path": str(video_path), "frames": len(episode_video_frames)}),
                      flush=True)
    env.close()
    if comparisons:
        comparison_file = Path(args.comparison_out)
        comparison_file.parent.mkdir(parents=True, exist_ok=True)
        with comparison_file.open("w", encoding="utf-8") as fh:
            for row in comparisons:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        # Per-episode statistics make it easier to locate where plans diverge.
        by_episode = {}
        for row in comparisons:
            ep = row["episode"]
            by_episode.setdefault(ep, []).append(row)
        for ep, rows in by_episode.items():
            print(json.dumps({
                "event": "high_level_episode_summary",
                "episode": ep,
                "seed": args.seed + ep,
                "comparisons": len(rows),
                "mean_subgoal_distance": float(np.mean([x["subgoal_distance"] for x in rows])),
                "mean_high_latent_mse": float(np.mean([x["high_latent_mse"] for x in rows])),
                "mean_high_latent_cosine": float(np.mean([x["high_latent_cosine"] for x in rows])),
                "first_subgoal_distance": rows[0]["subgoal_distance"],
            }), flush=True)
        print(json.dumps({"event": "high_level_comparison_saved",
                          "path": str(comparison_file), "rows": len(comparisons)}), flush=True)
    if mode == "eval" and args.video_out:
        for category in ("success", "failure"):
            if category not in saved_video_types:
                print(json.dumps({"event": "video_not_available", "category": category,
                                  "reason": "no_matching_episode"}), flush=True)
    print(json.dumps({"mode": mode, "episodes": args.episodes,
                      "success_rate": float(np.mean(successes)),
                      "fusion_alpha": args.fusion_alpha if mode == "eval" else None,
                      "fusion_max_ratio": args.fusion_max_ratio if mode == "eval" else None}), flush=True)
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
    # Freeze Mini-JEPA HWM high-level encoder, while allowing gradients
    # through it to train ONLY the bridge toward planning-relevant features.
    psi, _, _, _, _ = load_hwm(str(paths()["hwm"]), args.device)
    psi.eval()
    for parameter in psi.parameters():
        parameter.requires_grad_(False)
    opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=1e-4)
    best = float("inf")
    for epoch in range(args.epochs):
        bridge.train()
        permutation = np.random.permutation(np.where(train)[0])
        for pos in range(0, len(permutation), args.batch_size):
            ids = permutation[pos:pos + args.batch_size]
            pred = bridge(x[ids].to(args.device))
            target = y[ids].to(args.device)
            with torch.no_grad():
                target_high = psi(target)
            high_pred = psi(pred)
            loss = (F.mse_loss(pred, target)
                    + 0.1 * (1.0 - F.cosine_similarity(pred, target, dim=-1).mean())
                    + args.planning_loss_weight * F.mse_loss(high_pred, target_high))
            opt.zero_grad()
            loss.backward()
            opt.step()
        bridge.eval()
        with torch.inference_mode():
            val_x = x[valid].to(args.device)
            val_y = y[valid].to(args.device)
            val_pred = bridge(val_x)
            val_mse = F.mse_loss(val_pred, val_y).item()
            val_high_mse = F.mse_loss(psi(val_pred), psi(val_y)).item()
            val_score = val_mse + args.planning_loss_weight * val_high_mse
        print(f"epoch={epoch + 1} validation_mse={val_mse:.6f} "
              f"validation_high_mse={val_high_mse:.6f} "
              f"validation_score={val_score:.6f}", flush=True)
        if val_score < best:
            best = val_score
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": bridge.state_dict(),
                "input_dim": x.shape[1],
                "target_dim": y.shape[1],
                "hidden": args.hidden,
                "validation_mse": val_mse,
                "validation_high_mse": val_high_mse,
                "validation_score": best,
                "planning_loss_weight": args.planning_loss_weight,
                "teacher": "Mini-JEPA Medium state latent",
                "visual_encoder": "Meta JEPA-WM PointMaze (frozen)",
                "representation": "paired temporal spatial 4x4 Meta tokens",
            }, out)
    print(json.dumps({"bridge_saved": args.output, "best_validation_score": best}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["collect", "train", "eval"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data", default="outputs/visual_medium/paired_data.npz")
    parser.add_argument("--output", default="outputs/visual_medium/visual_bridge.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--video-out", default="outputs/visual_medium/visual_nav.mp4")
    parser.add_argument("--compare-high-level", action="store_true",
                        help="Shadow compare visual vs Mini teacher subgoals without changing controls")
    parser.add_argument("--comparison-out",
                        default="outputs/visual_medium/high_level_comparison.jsonl")
    parser.add_argument("--log-comparisons", action="store_true",
                        help="Print every high-level comparison rather than only episode averages")
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--every", type=int, default=4)
    parser.add_argument("--history", type=int, default=4,
                        help="Number of past simulation steps for the second visual frame")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fusion-alpha", type=float, default=1.0,
                        help="0=Mini state only, 1=visual only (if uncapped); 0.1-0.2 for conservative fusion")
    parser.add_argument("--fusion-max-ratio", type=float, default=0.5,
                        help="Max RMS visual residual relative to state RMS; 0.5 is conservative")
    parser.add_argument("--planning-loss-weight", type=float, default=1.0,
                        help="Weight on HWM psi(z_visual) vs psi(z_teacher) loss")
    parser.add_argument("--reach-radius", type=float, default=1.0)
    parser.add_argument("--low-timeout", type=int, default=90)
    args = parser.parse_args()
    if args.history < 1:
        parser.error("--history must be >= 1")
    if not (0.0 <= args.fusion_alpha <= 1.0):
        parser.error("--fusion-alpha must be between 0 and 1")
    if args.fusion_max_ratio <= 0:
        parser.error("--fusion-max-ratio must be positive")
    args.device = torch.device(args.device)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    if args.mode == "collect":
        run_controller(args, "collect")
    elif args.mode == "train":
        train_bridge(args)
    else:
        if args.fusion_alpha == 0.0:
            # The pure state baseline requires neither bridge nor Meta checkpoint.
            run_controller(args, "eval")
        else:
            ck = torch.load(args.output, map_location="cpu", weights_only=True)
            bridge = VisualBridge(ck["input_dim"], ck["target_dim"], ck["hidden"])
            bridge.load_state_dict(ck["model"])
            bridge.to(args.device).eval()
            meta = open_meta(args.device)
            run_controller(args, "eval", bridge, meta)


if __name__ == "__main__":
    main()
