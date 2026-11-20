# Mini-JEPA Robotics Reproduction — FetchReach-v4 (2026-11-20)

> Reproduction and planning-configuration study using the upstream [WilliamZhang20/Mini-JEPA](https://github.com/WilliamZhang20/Mini-JEPA), run **locally in WSL**. These results were provided from terminal logs; model weights, the upstream repository and evaluation artifacts have **not** been imported into this GitHub repository.

## Setup and scope

- Environment: Gymnasium Robotics `FetchReach-v4`, MuJoCo; WSL, Python 3.11, RTX 5060 8 GB (CUDA).
- Method: low-dimensional state-based action-conditioned JEPA latent predictor + online Gradient MPC. **Not** a vision-input or Meta RoboJEPA/V-JEPA 2-AC reproduction.
- Data: 10,000 simulator interaction steps; training: 2,000 gradient steps, batch size 128.
- Predictor: rollout; latent dimension 64, hidden dimension 256, trained horizons `[1,2,4,8]`; optimizer learning rate 0.0003.
- Evaluation: 50 episodes each, no teacher correction or scripted proposals in JEPA policies (`--teacher-correction-fraction 0.0 --jepa-scripted-proposal-fraction 0.0`).

## Measured results

| Policy / configuration | Success | Mean final distance | Mean action delta | Whole evaluation elapsed |
|---|---:|---:|---:|---:|
| Random baseline (50 episodes) | 0/50 (0%) | 0.237028 | 1.540333 | Not separately timed |
| Scripted baseline (50 episodes) | 50/50 (100%) | 0.001782 | 0.029027 | Not separately timed |
| Gradient MPC: 64 candidates / horizon 8 / 25 grad iters (50 episodes) | 17/50 (34%) | 0.075539 | 1.361999 | Not recorded |
| **Lightweight Gradient MPC, no smoothing:** 16 / 4 / 5 (50 episodes) | **47/50 (94%)** | **0.042857** | **0.808302** | **76.51 s** |
| **Lightweight Smooth MPC:** 16 / 4 / 5 (50 episodes) | **48/50 (96%)** | **0.019417** | **0.189257** | **85.58 s** |

For the lightweight comparison, both policies used `--mpc-method grad --mpc-score state --grad-lr 0.06`. The unsmoothed run set `--action-l2-weight 0 --action-delta-weight 0 --execute-smoothing 0`. The smooth run used weights `0.02`, `0.1`, and `0.2`, respectively.

### Interpretation / limitations

- Changing **both planning compute** (candidates/horizon/iterations) and **smoothing settings** between the initial 34% run and the final 96% run is *not* a one-factor ablation; the 62-percentage-point difference cannot be attributed solely to smoothing.
- With the *same lightweight planning parameters*, the **joint** action-penalty/smoothing change gave 94% → 96% (+2 percentage points, one extra successful episode out of 50), 0.808302 → 0.189257 mean action delta (−76.6%), and 0.042857 → 0.019417 mean final distance (−54.7%). This does **not** isolate individual smoothing coefficients or demonstrate statistically significant success-rate improvement.
- The no-smoothing run took 76.51 s versus 85.58 s with smoothing (+11.9%); do **not** claim smoothing alone speeds inference.
- The initial 34% run lacked a reported elapsed time, so no measured acceleration factor versus the initial configuration is available.
- The upstream simulator `scripted` policy is a target-aware reference, not a learned JEPA policy.
- Five-episode smoke evaluations (including a 5/5 result) are useful for checking execution but are not the primary benchmark.
- Reported numbers are from one model/checkpoint and evaluation run per setting, not multi-seed statistics.

## Reproduction commands

Execute in the *upstream cloned directory*, not the root of this repository:

```bash
conda activate jepa-wm311
cd ~/projects/JEPA-World-Model-Lab/external/mini-jepa

# Lightweight smooth, 50 episodes; CUDA model/MPC, no video flags.
PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MUJOCO_GL=osmesa \
time python -m jepa_robotics.evaluate \
  --task fetch_reach \
  --output-root runs \
  --model-path runs/fetch_reach/checkpoints/reach_gpu_test_model.pt \
  --episodes 50 --mpc-method grad --mpc-score state \
  --mpc-candidates 16 --mpc-horizon 4 \
  --grad-iters 5 --grad-lr 0.06 \
  --action-l2-weight 0.02 --action-delta-weight 0.1 \
  --execute-smoothing 0.2 \
  --teacher-correction-fraction 0.0 \
  --jepa-scripted-proposal-fraction 0.0 \
  --device cuda \
  --out runs/fetch_reach/eval_results/reach_smooth_fast50.jsonl
```

For the lightweight *no-smoothing* joint ablation, use the same command, set the three action coefficients to `0.0`, and write to `reach_ablation_no_smooth50.jsonl`.

### Local output paths (not committed)

- Checkpoint: `runs/fetch_reach/checkpoints/reach_gpu_test_checkpoint.pt`
- Model: `runs/fetch_reach/checkpoints/reach_gpu_test_model.pt`
- Evaluation: `runs/fetch_reach/eval_results/reach_gpu_eval50.jsonl`
- Smooth: `runs/fetch_reach/eval_results/reach_smooth_fast50.jsonl`
- No smoothing: `runs/fetch_reach/eval_results/reach_ablation_no_smooth50.jsonl`

## Related experiment

Earlier Mini-JEPA **PointMaze Medium** experiments were reported separately: self-trained evaluation 45/50 (90%), using a mixture of trained and supplied upstream components. That experiment's checkpoint, exact configuration and logs are **not** part of this report; treat it as a separate track until artifacts are verified and uploaded.

## Resume-ready summary (Chinese)

基于开源 Mini-JEPA 复现状态驱动的 Action-Conditioned JEPA，在 MuJoCo FetchReach-v4 中采集 1 万步交互数据并完成 2,000 步 GPU 训练；针对 Gradient MPC 对规划参数及动作平滑/正则化开展消融，固定轻量规划配置后，50 回合成功率由 94%（47/50）至 96%（48/50），平均动作变化降低 76.6%、最终目标距离降低 54.7%。本实验为仿真复现及配置优化，不代表自主研发官方 RoboJEPA。
