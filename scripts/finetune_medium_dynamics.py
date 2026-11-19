"""Fine-tune the last Meta AdaLN JEPA-WM layers for Mini Medium Maze dynamics.

Uses the existing paired_chunks.npz dataset, frozen Meta DINO vision and
frozen spatial XY head. Only the last N AdaLN blocks, predictor projection,
10-D action adapter and a residual latent calibration head are trained.

The residual correction is initialized as identity dynamics. To claim useful
action prediction, the held-out true-action forecast must beat BOTH a static
identity baseline and an incorrect-action ablation. No such result is assumed.

Usage:
 python scripts/finetune_medium_dynamics.py --device cuda --epochs 12
 python scripts/medium_action_predictor.py eval --device cuda --planner meta \
   --checkpoint outputs/medium_action/meta_dynamics_finetuned.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .medium_action_predictor import (
        STRIDE, ActionAdapter, PositionHead, ResidualDynamicsHead,
        action_predict, encode_images, meta_action_stats, normalize_actions,
        previous_action_lookup,
    )
    from .visual_medium_bridge import open_meta
except ImportError:
    from medium_action_predictor import (
        STRIDE, ActionAdapter, PositionHead, ResidualDynamicsHead,
        action_predict, encode_images, meta_action_stats, normalize_actions,
        previous_action_lookup,
    )
    from visual_medium_bridge import open_meta


def configure_trainable_predictor(meta, last_blocks: int):
    """Unfreeze final AdaLN block(s) and output projector, never the DINO encoder."""
    meta.requires_grad_(False)
    predictor = meta.model.predictor
    blocks = predictor.predictor_blocks
    if not 1 <= last_blocks <= len(blocks):
        raise ValueError(f"last-blocks must be 1..{len(blocks)}")
    for block in list(blocks)[-last_blocks:]:
        block.requires_grad_(True)
    predictor.predictor_proj.requires_grad_(True)
    predictor.predictor_norm.requires_grad_(True)
    trainable = {
        name: p for name, p in predictor.named_parameters() if p.requires_grad
    }
    if not trainable:
        raise RuntimeError("No trainable predictor parameters")
    return trainable


def cache_encodings(meta, data, device, batch_size=2):
    """Encode once on GPU; store features in float16 on CPU to fit an 8 GB GPU."""
    vis, prop, target = [], [], []
    n = len(data["episode"])
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        ctx = encode_images(meta, data["context"][start:stop], device,
                            proprio=data["context_proprio"][start:stop])
        tgt = encode_images(meta, data["future"][start:stop, None], device)
        vis.append(ctx["visual"].detach().cpu().half())
        prop.append(ctx["proprio"].detach().cpu().half())
        target.append(tgt[:, 0].detach().cpu().half())
        if (start // batch_size + 1) % 50 == 0:
            print(json.dumps({"event": "feature_cache_progress", "rows": stop,
                              "total": n}), flush=True)
    return (torch.cat(vis), torch.cat(prop), torch.cat(target))


def prepared_batch(ids, cache, data, prev_idx, device, mean, std):
    vis, prop, target = cache
    idx = np.asarray(ids, dtype=np.int64)
    context = {
        "visual": vis[idx].to(device=device, dtype=torch.float32),
        "proprio": prop[idx].to(device=device, dtype=torch.float32),
    }
    future = target[idx].to(device=device, dtype=torch.float32)
    current_action = torch.from_numpy(data["actions"][idx]).to(device=device, dtype=torch.float32)
    previous_action = torch.from_numpy(data["actions"][prev_idx[idx]]).to(device=device, dtype=torch.float32)
    goal = torch.from_numpy(data["next_xy"][idx]).to(device=device, dtype=torch.float32)
    return (context, future, normalize_actions(current_action, mean, std),
            normalize_actions(previous_action, mean, std), goal)


def evaluate_subset(ids, cache, data, prev_idx, device,
                    meta, adapter, calibrator, head, mean, std, limit):
    meta.eval(); adapter.eval(); calibrator.eval()
    scores = {"true": [], "negative": [], "identity": [], "future_xy": []}
    # Strictly grouped by held-out EPISODES, not by adjacent video frames.
    with torch.no_grad():
        for i in ids[:limit]:
            ctx, future, a, previous, xy = prepared_batch(
                [int(i)], cache, data, prev_idx, device, mean, std,
            )
            z = action_predict(meta, ctx, a, adapter, previous, calibrator)
            wrong = action_predict(meta, ctx, -a, adapter, previous, calibrator)
            scores["true"].append(F.mse_loss(z, future).item())
            scores["negative"].append(F.mse_loss(wrong, future).item())
            scores["identity"].append(F.mse_loss(ctx["visual"][:, -1], future).item())
            scores["future_xy"].append(
                torch.linalg.vector_norm(head(z) - xy, dim=-1).item()
            )
    return {k: float(np.mean(v)) for k, v in scores.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data", default="outputs/medium_action/paired_chunks.npz")
    p.add_argument("--head-checkpoint",
                   default="outputs/medium_action/meta_action_adapter_spatial.pt")
    p.add_argument("--checkpoint",
                   default="outputs/medium_action/meta_dynamics_finetuned.pt")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--last-blocks", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--cache-batch-size", type=int, default=2)
    p.add_argument("--max-train-steps", type=int, default=0,
                   help="Limit optimizer batches per epoch; 0 uses all samples")
    p.add_argument("--max-valid", type=int, default=60)
    p.add_argument("--xy-weight", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    if a.batch_size < 1 or a.cache_batch_size < 1 or a.patience < 1:
        p.error("batch sizes and patience must be positive")
    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    with np.load(a.data) as packed:
        data = {name: np.asarray(packed[name]) for name in packed.files}
    required = {"episode", "actions", "context", "context_proprio", "future", "next_xy"}
    if not required <= set(data):
        raise ValueError(f"Dataset is missing fields: {sorted(required - set(data))}")
    episodes = data["episode"]
    if len(np.unique(episodes)) < 2:
        raise ValueError("Need at least two complete episode groups")
    holdout = np.unique(episodes)[-max(1, len(np.unique(episodes)) // 5):]
    prev_idx, has_prev = previous_action_lookup(episodes)
    train_idx = np.flatnonzero((~np.isin(episodes, holdout)) & has_prev)
    valid_idx = np.flatnonzero(np.isin(episodes, holdout) & has_prev)
    if len(train_idx) == 0 or len(valid_idx) == 0:
        raise ValueError("Insufficient contiguous five-step action transitions")

    ck_head = torch.load(a.head_checkpoint, map_location="cpu", weights_only=True)
    if ck_head.get("position_head") != "spatial_4x4":
        raise ValueError("Need the spatial head checkpoint from the preceding experiment")
    head = PositionHead().to(device)
    head.load_state_dict(ck_head["head"])
    head.requires_grad_(False).eval()
    adapter = ActionAdapter().to(device)
    if "adapter" in ck_head:
        adapter.load_state_dict(ck_head["adapter"])

    meta = open_meta(device)
    trainable = configure_trainable_predictor(meta, a.last_blocks)
    corrector = ResidualDynamicsHead().to(device)
    mean, std = meta_action_stats()
    mean = torch.tensor(mean, device=device)
    std = torch.tensor(std, device=device)
    print(json.dumps({"event": "dynamics_setup", "train_samples": len(train_idx),
                      "valid_samples": len(valid_idx),
                      "tuned_predictor_parameters": sum(x.numel() for x in trainable.values()),
                      "last_blocks": a.last_blocks}), flush=True)
    cache = cache_encodings(meta, data, device, a.cache_batch_size)
    print(json.dumps({"event": "features_cached",
                      "shape": list(cache[0].shape)}), flush=True)
    optimizer = torch.optim.AdamW(
        [{"params": list(trainable.values()), "lr": a.lr},
         {"params": adapter.parameters(), "lr": a.lr},
         {"params": corrector.parameters(), "lr": a.lr * 3}],
        weight_decay=1e-4,
    )
    best, stale = float("inf"), 0
    for epoch in range(1, a.epochs + 1):
        meta.model.predictor.train()
        adapter.train(); corrector.train()
        order = np.random.permutation(train_idx)
        losses = []
        for j, start in enumerate(range(0, len(order), a.batch_size)):
            if a.max_train_steps and j >= a.max_train_steps:
                break
            idx = order[start:start + a.batch_size]
            ctx, future, actions, previous, xy = prepared_batch(
                idx, cache, data, prev_idx, device, mean, std,
            )
            z = action_predict(meta, ctx, actions, adapter, previous, corrector)
            # Unlike raw pretrained output, residual output starts at identity.
            loss = F.mse_loss(z, future) + a.xy_weight * F.smooth_l1_loss(head(z), xy)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch {j}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(trainable.values()) + list(adapter.parameters())
                + list(corrector.parameters()), 1.0
            )
            optimizer.step()
            losses.append(float(loss.item()))
        metrics = evaluate_subset(
            valid_idx, cache, data, prev_idx, device,
            meta, adapter, corrector, head, mean, std, a.max_valid,
        )
        metrics["stage"] = "finetuned_dynamics"
        metrics["epoch"] = epoch
        metrics["train_loss"] = float(np.mean(losses))
        metrics["true_minus_negated"] = metrics["true"] - metrics["negative"]
        metrics["beats_identity"] = metrics["true"] < metrics["identity"]
        metrics["action_sensitive"] = metrics["true"] < metrics["negative"]
        print(json.dumps(metrics), flush=True)
        if metrics["true"] < best - 1e-6:
            best, stale = metrics["true"], 0
            out = Path(a.checkpoint)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "adapter": adapter.state_dict(),
                "head": head.state_dict(),
                "corrector": corrector.state_dict(),
                "predictor_tuned": {
                    name: tensor.detach().cpu()
                    for name, tensor in trainable.items()
                },
                "dynamics_finetuned": True,
                "position_head": "spatial_4x4",
                "action_stride": STRIDE,
                "validation_latent_mse": best,
                "validation_identity_mse": metrics["identity"],
                "validation_wrong_action_mse": metrics["negative"],
                "validation_xy_error": metrics["future_xy"],
                "train_episodes": np.unique(episodes[train_idx]).tolist(),
                "valid_episodes": holdout.tolist(),
                "method": "frozen Meta visual encoder + fine-tuned final AdaLN blocks + residual calibration",
            }, out)
        else:
            stale += 1
            if stale >= a.patience:
                print(json.dumps({"event": "early_stopping", "epoch": epoch,
                                  "best_valid_mse": best}), flush=True)
                break
    print(json.dumps({"checkpoint": a.checkpoint, "best_valid_mse": best,
                      "note": "Only claim model-based advantage if true < identity and true < wrong-action"}))


if __name__ == "__main__":
    main()
