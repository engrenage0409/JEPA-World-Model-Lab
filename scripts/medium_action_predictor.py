"""Meta JEPA-WM action-conditioned medium-maze adapter.

Three explicit stages:
  collect: teacher Mini hierarchical rollouts => RGB + five executed actions + xy.
  train: frozen Meta visual encoder/predictor + trainable affine action adapter and
         latent-to-position readout. Validates held-out dynamics and position errors.
  eval:  Mini HWM/Macro Flow selects a long-term subgoal. Mini low-level Flow
         proposes action chunks; Meta action-conditioned Predictor forecasts
         each chunk, and an adapted position head selects the chunk closest
         to the subgoal. --planner flow disables Meta ranking for ablation.

The low-level controller and long-term planner both consume true state; this is
NOT a pure-vision agent and NOT fine-tuning of the Meta Predictor backbone.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .visual_medium_bridge import (
        ROOT, make_components, open_meta, plan_subgoal, teacher_latent,
    )
except ImportError:
    from visual_medium_bridge import (
        ROOT, make_components, open_meta, plan_subgoal, teacher_latent,
    )
from jepa_robotics.envs import flatten_obs

# Official JEPA-WM PointMaze pretraining used 5 actions per RGB frame.
STRIDE = 5


def meta_action_stats():
    # Bundle the 2-D PointMaze action statistics from the upstream Meta config.
    from app.plan_common.datasets import get_data_stats
    stats = get_data_stats("pointmaze")
    return np.asarray(stats["action_mean"], np.float32), np.asarray(stats["action_std"], np.float32)


class ActionAdapter(nn.Module):
    """Small residual affine correction on 5x2 normalized Mini actions."""
    def __init__(self):
        super().__init__()
        self.delta = nn.Linear(STRIDE * 2, STRIDE * 2)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, actions):
        if actions.shape[-2:] != (STRIDE, 2):
            raise ValueError(f"Expected [B,{STRIDE},2], got {tuple(actions.shape)}")
        flat = actions.reshape(-1, STRIDE * 2)
        return flat + 0.1 * torch.tanh(self.delta(flat))


class PositionHead(nn.Module):
    """Visual latent readout to world XY (trained, never ground-truth at eval)."""
    def __init__(self, d=384):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128),
                                 nn.GELU(), nn.Linear(128, 2))

    def forward(self, z):
        # z [B, 1, V, H, W, D] or [B, V, H, W, D]
        if z.ndim == 6:
            z = z[:, -1]
        return self.net(z.mean(dim=(1, 2, 3)))


def encode_images(meta, images, device, proprio=None):
    """Meta visual encoder; supply matching 4D proprio when using AdaLN Predictor.

    The official PointMaze AdaLN expects 384 visual + 16 proprio features.
    Omitting proprio passes 384D tokens to a 400D LayerNorm and crashes.
    """
    arr = np.ascontiguousarray(images)
    x = torch.from_numpy(arr).permute(0, 1, 4, 2, 3).to(device)
    with torch.no_grad():
        if proprio is None:
            return meta.encode(x)
        p = torch.as_tensor(np.ascontiguousarray(proprio), dtype=torch.float32)
        if p.shape != (x.shape[0], x.shape[1], 4):
            raise ValueError(f"Expected proprio [B,T,4], got {tuple(p.shape)}")
        return meta.encode({"visual": x, "proprio": p})


def env_proprio(obs):
    """PointMaze simulator state: current xy and xy velocity (four floats)."""
    p = np.asarray(obs["observation"], dtype=np.float32).reshape(-1)
    if p.shape[0] < 4:
        raise ValueError(f"Expected position + velocity, got {p.shape}")
    return p[:4].copy()


def normalize_actions(actions, mean, std):
    return (actions - mean) / std


def action_predict(meta, ctx, actions, adapter):
    """Differentiable through adapter, frozen official Meta predictor."""
    a = adapter(actions)
    prediction = meta.unroll(ctx, a.unsqueeze(0))
    # Multimodal official model returns TensorDict[T,B,...], not a visual tensor.
    return prediction["visual"][-1] if isinstance(ctx, dict) or hasattr(ctx, "keys") else prediction[-1]


def select_low_chunks(low, obs, subgoal, count, device):
    """Reuse EXACT pretrained Mini low-level rectified-flow sampler."""
    o = {k: np.array(v, copy=True) for k, v in obs.items()}
    o["desired_goal"] = np.asarray(subgoal, dtype=np.float32)
    s = torch.from_numpy(low.norm.encode(flatten_obs(o))).unsqueeze(0).to(device)
    with torch.no_grad():
        z = low.model.encode(s)
        cond = torch.cat([z, s], dim=1) if low.concat_raw else z
        if low.emphasis_repeat:
            a0, a1 = low.agent_dims
            g0, g1 = low.goal_dims
            cond = torch.cat([cond, (s[:, g0:g1] - s[:, a0:a1])
                              .repeat(1, low.emphasis_repeat)], dim=1)
        if low.progress_cond:
            tp = torch.full((1, 1), low.target_progress,
                            dtype=cond.dtype, device=device)
            cond = torch.cat([cond, tp], dim=1)
        samples = low.net.sample(cond.repeat(count, 1),
                                 low.chunk_dim, low.flow_steps)
        chunks = samples.reshape(count, low.chunk, low.action_dim)
    if low.chunk < STRIDE:
        raise RuntimeError(f"Mini low-level chunk length {low.chunk} < {STRIDE}")
    return np.clip(chunks[:, :STRIDE].cpu().numpy(), low.low, low.high)


def collect(args):
    _, env, low, psi, g, dec, (fnet, ddpm, config) = make_components(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = dict(context=[], context_proprio=[], future=[], actions=[],
                   xy=[], next_xy=[], episode=[])
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            goal = np.asarray(obs["desired_goal"], np.float32)
            subgoal = goal.copy()
            need, elapsed, step = True, 0, 0
            done, truncated = False, False
            images = [np.asarray(env.render(), dtype=np.uint8).copy()]
            actions = []
            positions = [np.asarray(obs["achieved_goal"], np.float32).copy()]
            proprios = [env_proprio(obs)]
            while not (done or truncated):
                reached = np.asarray(obs["achieved_goal"], np.float32)
                if need or np.linalg.norm(reached - subgoal) < args.reach_radius or elapsed >= args.low_timeout:
                    with torch.inference_mode():
                        z = psi(teacher_latent(low, obs, args.device))
                        subgoal = plan_subgoal(fnet, ddpm, g, dec, z, goal, args.device, config)
                    if np.linalg.norm(subgoal - goal) < args.reach_radius:
                        subgoal = goal
                    elapsed, need = 0, False
                action = low.act(obs, subgoal)
                obs, _, done, truncated, info = env.step(action)
                actions.append(np.asarray(action, np.float32))
                images.append(np.asarray(env.render(), dtype=np.uint8).copy())
                positions.append(np.asarray(obs["achieved_goal"], np.float32).copy())
                proprios.append(env_proprio(obs))
                step += 1
                elapsed += 1
                # Context is images[t-5], images[t]; target image[t+5].
                # Only store nonoverlapping 5-step chunks at image index t.
                if step >= 2 * STRIDE and step % STRIDE == 0:
                    t = step - STRIDE
                    records["context"].append(np.stack((images[t-STRIDE], images[t])))
                    records["context_proprio"].append(np.stack((proprios[t-STRIDE], proprios[t])))
                    records["future"].append(images[t+STRIDE])
                    records["actions"].append(np.stack(actions[t:t+STRIDE]))
                    records["xy"].append(positions[t].copy())
                    records["next_xy"].append(positions[t+STRIDE].copy())
                    records["episode"].append(ep)
                    if args.max_pairs and len(records["episode"]) >= args.max_pairs:
                        break
            print(json.dumps({"episode": ep, "steps": step,
                              "pairs": len(records["episode"]),
                              "success": float(info.get("is_success", 0))}), flush=True)
            if args.max_pairs and len(records["episode"]) >= args.max_pairs:
                break
    finally:
        env.close()
    if not records["episode"]:
        raise RuntimeError("No transitions collected")
    out = Path(args.data); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in records.items()})
    print(json.dumps({"dataset": str(out), "pairs": len(records["episode"])}))


def train(args):
    # npz indexing decompresses a whole array each time; load each array ONCE.
    # Otherwise per-example DINO encoding repeatedly decompresses ~hundreds MB.
    with np.load(args.data) as packed:
        data = {name: np.asarray(packed[name]) for name in packed.files}
    if "context_proprio" not in data:
        raise RuntimeError("Dataset lacks 4D PointMaze proprio inputs; re-run collect with this version.")
    episodes = data["episode"]
    if len(np.unique(episodes)) < 2:
        raise ValueError("Collect at least two complete episodes for validation")
    holdout = np.unique(episodes)[-max(1, len(np.unique(episodes)) // 5):]
    train_ids = np.flatnonzero(~np.isin(episodes, holdout))
    valid_ids = np.flatnonzero(np.isin(episodes, holdout))
    if not len(train_ids) or not len(valid_ids):
        raise RuntimeError("Empty episode-grouped train/validation split")
    meta = open_meta(args.device)
    mean, std = meta_action_stats()
    mean = torch.tensor(mean, device=args.device)
    std = torch.tensor(std, device=args.device)
    head = PositionHead().to(args.device)
    adapter = ActionAdapter().to(args.device)
    # Stage A: supervised visual position readout. Frozen DINO representation.
    optim_head = torch.optim.AdamW(head.parameters(), lr=args.lr)
    for epoch in range(args.head_epochs):
        order = np.random.permutation(train_ids)
        losses = []
        for i in order:
            z = encode_images(meta, data["context"][i:i+1, -1:], args.device)
            xy = torch.as_tensor(np.asarray(data["xy"][i:i+1]), device=args.device, dtype=torch.float32)
            loss = F.smooth_l1_loss(head(z), xy)
            optim_head.zero_grad(); loss.backward(); optim_head.step()
            losses.append(loss.item())
        print(json.dumps({"stage": "position", "epoch": epoch+1,
                          "train_loss": float(np.mean(losses))}), flush=True)
    for p in head.parameters():
        p.requires_grad_(False)
    head.eval()

    # Stage B: learn Mini->Meta action correction against real future RGB latents.
    # Meta predictor is frozen, but autograd traverses it to the 10-D adapter.
    optim_adapter = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    best = float("inf")
    for epoch in range(args.epochs):
        adapter.train()
        order = np.random.permutation(train_ids)
        losses = []
        for i in order:
            ctx = encode_images(meta, data["context"][i:i+1], args.device,
                                proprio=data["context_proprio"][i:i+1])
            future = encode_images(meta, data["future"][i:i+1, None], args.device)[:, 0]
            actions = torch.as_tensor(np.asarray(data["actions"][i:i+1]),
                                      device=args.device, dtype=torch.float32)
            actions = normalize_actions(actions, mean, std)
            pred = action_predict(meta, ctx, actions, adapter)
            loss_latent = F.mse_loss(pred, future)
            loss_xy = F.smooth_l1_loss(head(pred), torch.as_tensor(
                np.asarray(data["next_xy"][i:i+1]), device=args.device, dtype=torch.float32))
            loss = loss_latent + args.position_weight * loss_xy
            optim_adapter.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optim_adapter.step()
            losses.append(loss.item())
        adapter.eval()
        val_pred, val_xy, val_base = [], [], []
        with torch.no_grad():
            for i in valid_ids[:args.max_valid]:
                ctx = encode_images(meta, data["context"][i:i+1], args.device,
                                    proprio=data["context_proprio"][i:i+1])
                future = encode_images(meta, data["future"][i:i+1, None], args.device)[:, 0]
                actions = torch.as_tensor(np.asarray(data["actions"][i:i+1]),
                                          device=args.device, dtype=torch.float32)
                actions = normalize_actions(actions, mean, std)
                z = action_predict(meta, ctx, actions, adapter)
                val_pred.append(F.mse_loss(z, future).item())
                val_xy.append(torch.linalg.vector_norm(head(z) - torch.as_tensor(
                    np.asarray(data["next_xy"][i:i+1]), device=args.device, dtype=torch.float32), dim=-1).item())
                # Identical observations, but shuffled actions as an action-conditioning ablation.
                invalid_action = -actions
                ablated = action_predict(meta, ctx, invalid_action, adapter)
                val_base.append(F.mse_loss(ablated, future).item())
        metric = float(np.mean(val_pred))
        print(json.dumps({"stage": "action_predictor", "epoch": epoch+1,
                          "train_loss": float(np.mean(losses)),
                          "valid_future_latent_mse": metric,
                          "valid_future_xy_error": float(np.mean(val_xy)),
                          "valid_negated_action_latent_mse": float(np.mean(val_base))}), flush=True)
        if metric < best:
            best = metric
            out = Path(args.checkpoint); out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"adapter": adapter.state_dict(), "head": head.state_dict(),
                        "validation_latent_mse": best,
                        "action_stride": STRIDE,
                        "source": "Meta JEPA-WM pretrained PointMaze, frozen",
                        "method": "supervised XY readout + action-input adapter",
                        "train_episodes": np.unique(episodes[train_ids]).tolist(),
                        "valid_episodes": holdout.tolist()}, out)
    print(json.dumps({"checkpoint": args.checkpoint, "best_valid_mse": best}))


def evaluate(args):
    if args.planner == "meta":
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if ck["action_stride"] != STRIDE:
            raise ValueError("Action stride mismatch")
        meta = open_meta(args.device)
        adapter = ActionAdapter().to(args.device).eval()
        adapter.load_state_dict(ck["adapter"])
        head = PositionHead().to(args.device).eval()
        head.load_state_dict(ck["head"])
        mean, std = meta_action_stats()
        mean = torch.tensor(mean, device=args.device)
        std = torch.tensor(std, device=args.device)
    else:
        meta = adapter = head = None
    _, env, low, psi, g, dec, (fnet, ddpm, config) = make_components(args.device)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    success = []
    decisions = []
    saved_video_types = set()
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            goal = np.asarray(obs["desired_goal"], np.float32)
            subgoal = goal.copy()
            prev = np.asarray(env.render(), np.uint8)
            history = deque([prev] * (STRIDE+1), maxlen=STRIDE+1)
            proprio_history = deque([env_proprio(obs)] * (STRIDE+1), maxlen=STRIDE+1)
            queue = deque()
            step, elapsed, need = 0, 0, True
            done = truncated = False
            info = {}
            record_video = bool(args.video_out) and len(saved_video_types) < 2
            video_frames = []
            while not (done or truncated):
                if record_video and step % 2 == 0:
                    video_frames.append(np.asarray(env.render(), np.uint8).copy())
                achieved = np.asarray(obs["achieved_goal"], np.float32)
                if need or np.linalg.norm(achieved - subgoal) < args.reach_radius or elapsed >= args.low_timeout:
                    with torch.inference_mode():
                        high = psi(teacher_latent(low, obs, args.device))
                        subgoal = plan_subgoal(fnet, ddpm, g, dec, high, goal, args.device, config)
                    if np.linalg.norm(subgoal - goal) < args.reach_radius:
                        subgoal = goal
                    need, elapsed = False, 0
                    queue.clear()
                if args.planner == "meta" and not queue:
                    chunks = select_low_chunks(low, obs, subgoal, args.candidates, args.device)
                    # Encode actual current and 5-steps-back frames; action
                    # predictor proposes future visual token for each chunk.
                    curr = np.asarray(env.render(), np.uint8)
                    context_p = np.stack((proprio_history[0], env_proprio(obs)))[None]
                    ctx = encode_images(meta, np.stack((history[0], curr))[None],
                                        args.device, proprio=context_p)
                    with torch.no_grad():
                        scores, predicted_positions = [], []
                        for chunk in chunks:
                            a = torch.as_tensor(chunk[None], dtype=torch.float32,
                                                device=args.device)
                            pred = action_predict(meta, ctx, normalize_actions(a, mean, std), adapter)
                            predicted_xy = head(pred)[0].cpu().numpy()
                            predicted_positions.append(predicted_xy.tolist())
                            scores.append(float(np.linalg.norm(predicted_xy - subgoal)))
                    idx = int(np.argmin(scores))
                    queue.extend(chunks[idx])
                    decisions.append({"episode": ep, "step": step,
                                      "chosen_candidate": idx, "scores": scores,
                                      "predicted_positions": predicted_positions,
                                      "subgoal": np.asarray(subgoal).tolist()})
                action = queue.popleft() if args.planner == "meta" else low.act(obs, subgoal)
                obs, _, done, truncated, info = env.step(action)
                history.append(np.asarray(env.render(), np.uint8).copy())
                proprio_history.append(env_proprio(obs))
                step += 1; elapsed += 1
            success.append(float(info.get("is_success", info.get("success", 0))))
            if record_video and video_frames:
                category = "success" if success[-1] > 0.5 else "failure"
                if category not in saved_video_types:
                    import imageio.v2 as imageio
                    path = Path(args.video_out)
                    path = path.with_name(f"{path.stem}_{args.planner}_{category}{path.suffix or '.mp4'}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    video_frames.append(np.asarray(env.render(), np.uint8).copy())
                    imageio.mimsave(str(path), video_frames, fps=30)
                    saved_video_types.add(category)
                    print(json.dumps({"event": "video_saved", "seed": args.seed+ep,
                                      "category": category, "path": str(path)}), flush=True)
            print(json.dumps({"episode": ep, "seed": args.seed+ep,
                              "steps": step, "success": success[-1],
                              "meta_decisions": sum(d["episode"] == ep for d in decisions)}), flush=True)
    finally:
        env.close()
    if decisions:
        out = Path(args.decision_log); out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for row in decisions:
                fh.write(json.dumps(row) + "\n")
    print(json.dumps({"planner": args.planner, "success_rate": float(np.mean(success)),
                      "episodes": len(success), "predictor_decisions": len(decisions)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["collect", "train", "eval"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=30000)
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--data", default="outputs/medium_action/paired_chunks.npz")
    ap.add_argument("--checkpoint", default="outputs/medium_action/meta_action_adapter.pt")
    ap.add_argument("--decision-log", default="outputs/medium_action/decisions.jsonl")
    ap.add_argument("--video-out", default="outputs/medium_action/nav.mp4",
                    help="Save first success and failure; empty string disables video")
    ap.add_argument("--max-pairs", type=int, default=600)
    ap.add_argument("--max-valid", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--head-epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--position-weight", type=float, default=0.1)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--planner", choices=["meta", "flow"], default="meta")
    ap.add_argument("--reach-radius", type=float, default=1.0)
    ap.add_argument("--low-timeout", type=int, default=90)
    args = ap.parse_args()
    args.device = torch.device(args.device)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    if args.candidates < 2 and args.planner == "meta" and args.mode == "eval":
        ap.error("--candidates must be >= 2 for model-based action ranking")
    if args.mode == "collect": collect(args)
    elif args.mode == "train": train(args)
    else: evaluate(args)


if __name__ == "__main__":
    main()
