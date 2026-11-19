"""CPU-only smoke test of Medium Maze states and diagnostic difficulty.

Run from external/jepa-wms:
    MUJOCO_GL=osmesa python scripts/check_medium_maze_tasks.py
"""
from types import SimpleNamespace
import numpy as np

from evals.simu_env_planning.envs.pointmaze_gym_wrap import make_env


def main():
    for level in ("straight", "corner", "detour"):
        cfg = SimpleNamespace(
            task_specification=SimpleNamespace(
                task=f"maze-medium-{level}",
                max_episode_steps=120,
            )
        )
        env = make_env(cfg)
        try:
            for seed in (1, 2, 3):
                start, goal = env.sample_random_init_goal_states(seed)
                assert start.shape == goal.shape == (4,)
                assert not np.allclose(start[:2], goal[:2])
                # Exercise actual MuJoCo reset, not only the BFS sampler.
                _, goal_info = env.prepare(seed, goal)
                _, start_info = env.prepare(seed, start)
                assert np.allclose(goal_info["state"][:2], goal[:2], atol=1e-5)
                assert np.allclose(start_info["state"][:2], start[:2], atol=1e-5)
                print(f"[PASS] {level} seed={seed}: start={start[:2]} goal={goal[:2]}")
        finally:
            env.close()
    print("Medium Maze start/goal smoke test passed.")


if __name__ == "__main__":
    main()
