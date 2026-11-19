"""Collect native Mini-JEPA PointMaze episodes for HWM/Macro Flow training.

Unlike the visual paired-chunk dataset, this preserves full episodes as
(states[T+1, state_dim], actions[T, 2]) with the official Mini-JEPA schema.
Uses Mini's existing scripted maze controller plus randomized exploration.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "mini-jepa"))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.data import collect_episodes, save_episodes_npz
from jepa_robotics.envs import make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("outputs/mini_scratch/medium_episodes.npz"))
    p.add_argument("--steps", type=int, default=150000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--scripted-fraction", type=float, default=0.9)
    p.add_argument("--noise", type=float, default=0.10)
    p.add_argument("--controller-gain", type=float, default=5.0)
    args = p.parse_args()
    env = make_env("PointMaze_Medium-v3", seed=args.seed, max_episode_steps=600)
    try:
        episodes, spec = collect_episodes(
            env, num_steps=args.steps, seed=args.seed,
            scripted_fraction=args.scripted_fraction,
            controller_gain=args.controller_gain,
            action_noise=args.noise, controller="maze", log_every=10000,
        )
    finally:
        env.close()
    valid = [ep for ep in episodes if ep.states.shape[0] == ep.actions.shape[0] + 1
             and ep.actions.shape[0] > 30]
    if len(valid) < 2:
        raise RuntimeError("Insufficient valid long episodes for HWM training.")
    save_episodes_npz(args.out, valid, spec)
    print(f"Saved {len(valid)} full episodes to {args.out}")
    print(f"Transitions: {sum(len(ep.actions) for ep in valid)}")


if __name__ == "__main__":
    main()
