import numpy as np
from evals.simu_env_planning.envs.pointmaze_env.maze_model import MazeEnv
from evals.simu_env_planning.envs.pointmaze_gym_wrap import PointMazeWrapper

EPISODES = 10
BASE_SEED = 1
STEPS = 30
THRESHOLD = 0.5

# 与 JEPA-WM 官方 PointMaze 环境配置一致。
# 使用 state 模式省去图像渲染，不影响物理步进。
env = MazeEnv(
    reward_type="sparse",
    reset_target=False,
    return_value="state",
    with_target=False,
    ref_min_score=23.85,
    ref_max_score=161.86,
    dataset_url="http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5",
)
sampler = PointMazeWrapper(env)

initial_successes = 0
zero_action_successes = 0

print("Episode | InitialDist | ZeroActionDist | InitialOK | ZeroActionOK")
print("-" * 70)

try:
    for ep in range(EPISODES):
        # 官方 PlanEvaluator 使用的单 GPU Episode 种子
        seed = (BASE_SEED**2 + ep * BASE_SEED) % (2**32 - 2)

        init_state, goal_state = sampler.sample_random_init_goal_states(seed)

        initial_dist = float(
            np.linalg.norm(init_state[:2] - goal_state[:2])
        )
        initial_ok = initial_dist < THRESHOLD
        initial_successes += int(initial_ok)

        # 在官方 MuJoCo 物理环境中执行 30 步全零动作
        env.seed(seed)
        env.set_init_state(init_state)
        env.reset()

        final_state = None
        for _ in range(STEPS):
            _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
            final_state = info["state"]

        final_dist = float(
            np.linalg.norm(final_state[:2] - goal_state[:2])
        )
        zero_ok = final_dist < THRESHOLD
        zero_action_successes += int(zero_ok)

        print(
            f"{ep:7d} | {initial_dist:11.3f} | "
            f"{final_dist:14.3f} | "
            f"{str(initial_ok):9s} | {zero_ok}"
        )
finally:
    env.close()

print("-" * 70)
print(f"Initial position baseline: {initial_successes}/{EPISODES}")
print(f"Zero-action baseline:      {zero_action_successes}/{EPISODES}")
print("JEPA-WM CEM 2 iterations: 5/10")
print("JEPA-WM CEM 5 iterations: 7/10")
