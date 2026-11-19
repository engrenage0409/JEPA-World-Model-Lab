"""MuJoCo Medium Maze state-feedback diagnostic (NOT a JEPA-WM score).

Based on p_gain=10 and d_gain=-1 in the official waypoint_controller.py.
Intended for straight corridors; corner/detour need waypoints.
Run from external/jepa-wms:
  MUJOCO_GL=osmesa python scripts/check_medium_control_baseline.py
"""
import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np

from evals.simu_env_planning.envs.pointmaze_curriculum import sample_medium_task
from evals.simu_env_planning.envs.pointmaze_gym_wrap import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--output", type=Path, default=Path("diagnostics/medium_control"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cfg = SimpleNamespace(
        task_specification=SimpleNamespace(
            task="maze-medium-straight", max_episode_steps=args.steps
        )
    )
    env = make_env(cfg)
    try:
        for seed in args.seeds:
            start, goal = sample_medium_task(seed, level="straight")
            out = args.output / f"straight_seed{seed}"
            out.mkdir(parents=True, exist_ok=True)
            env.prepare(seed, goal)
            imageio.imwrite(out / "goal.png", np.asarray(env.render()))
            _, goal_info = env.prepare(seed, goal)
            actual_goal = np.asarray(goal_info["state"])
            _, info = env.prepare(seed, start)
            # prepare() is forwarded through TimeLimit without calling reset().
            # Initialize its counter explicitly before taking any actions.
            env.set_elapsed_steps(0)
            imageio.imwrite(out / "start.png", np.asarray(env.render()))
            assert actual_goal.shape == (4,), actual_goal.shape
            assert np.allclose(actual_goal[:2], goal[:2]), "Goal reset mismatch"
            assert np.allclose(np.asarray(info["state"])[:2], start[:2]), "Start reset mismatch"

            trajectory = []
            reached_at = None
            for step in range(args.steps):
                state = np.asarray(info["state"], dtype=np.float64)
                action = np.clip(
                    10.0 * (goal[:2] - state[:2]) - state[2:4], -1.0, 1.0
                )
                _, _, terminated, truncated, info = env.step(action)
                cur = np.asarray(info["state"], dtype=np.float64)
                distance = float(np.linalg.norm(goal[:2] - cur[:2]))
                trajectory.append((step + 1, *cur, *action, distance))
                if distance < 0.5:
                    reached_at = step + 1
                    break
                if terminated or truncated:
                    break

            imageio.imwrite(out / "final.png", np.asarray(env.render()))
            with (out / "trajectory.csv").open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    ["step", "x", "y", "vx", "vy", "action_x", "action_y", "distance"]
                )
                writer.writerows(trajectory)
            last_dist = trajectory[-1][-1] if trajectory else float(
                np.linalg.norm(start[:2] - goal[:2])
            )
            status = "PASS" if reached_at is not None else "FAIL"
            print(
                f"[{status}] straight seed={seed} start={start[:2].tolist()} "
                f"goal={goal[:2].tolist()} reached_at={reached_at} "
                f"last_dist={last_dist:.3f} output={out}",
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
