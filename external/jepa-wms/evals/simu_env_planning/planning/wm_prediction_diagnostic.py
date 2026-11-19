"""Compare JEPA-WM one-action latent predictions with MuJoCo outcomes.

Diagnostic only: never changes planner actions or model weights.
All distances are measured in the *same frozen encoder's* visual latent space.
"""
import csv
from pathlib import Path

import torch

from evals.utils import prepare_obs


def _visual(z):
    return z["visual"] if isinstance(z, dict) or hasattr(z, "keys") else z


def _rmse(a, b):
    return float(torch.sqrt(torch.mean((a.float() - b.float()) ** 2)).item())


@torch.no_grad()
def evaluate_one_step_predictions(agent, cfg, observations, state_infos, action_records, output_path, max_checks=6):
    """Use identical starting observation and normalized actions for prediction.

    action_records: (physical_start_step, model_actions [T, A]); the actions are
    recorded immediately after CEM, before the preprocessor denormalizes them.
    Only first model action is tested, which maps to cfg.frameskip simulator steps
    when action_skip == 1.
    """
    if agent.model.action_skip != 1:
        raise ValueError("One-step diagnostic currently requires action_skip == 1")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    device = agent.device
    for step_start, model_actions in action_records[:max_checks]:
        step_end = step_start + cfg.frameskip
        if step_end >= len(observations):
            continue
        start_obs = prepare_obs(cfg.task_specification.obs, observations[step_start])
        next_obs = prepare_obs(cfg.task_specification.obs, observations[step_end])
        z0 = agent.model.encode(start_obs.to(device).unsqueeze(0), act=False)
        z_real = agent.model.encode(next_obs.to(device).unsqueeze(0), act=False)
        actions = model_actions[:1].to(device=device, dtype=torch.float32).unsqueeze(1)
        z_prediction = agent.model.unroll(z0, act_suffix=actions)
        before = _visual(z0)[:, -1]
        predicted = _visual(z_prediction)[-1]
        actual = _visual(z_real)[:, -1]
        target = _visual(agent.goal_state_enc)[:, -1]
        state0 = state_infos[step_start]
        state1 = state_infos[step_end]
        rows.append({
            "step_start": step_start,
            "action_norm_first": str(model_actions[0].tolist()),
            "pred_rmse": _rmse(predicted, actual),
            "static_rmse": _rmse(before, actual),
            "pred_to_goal_rmse": _rmse(predicted, target),
            "actual_to_goal_rmse": _rmse(actual, target),
            "start_to_goal_rmse": _rmse(before, target),
            "real_dx": float(state1[0] - state0[0]),
            "real_dy": float(state1[1] - state0[1]),
        })
    if not rows:
        print("[WM-DIAG] No full model-action window was observed")
        return
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[WM-DIAG] saved={output_path}")
    for r in rows:
        print(
            "[WM-DIAG] step={step_start} pred_rmse={pred_rmse:.4f} "
            "static_rmse={static_rmse:.4f} real_delta=({real_dx:+.3f},{real_dy:+.3f}) "
            "goal_start={start_to_goal_rmse:.4f} goal_pred={pred_to_goal_rmse:.4f} "
            "goal_real={actual_to_goal_rmse:.4f}".format(**r),
            flush=True,
        )
