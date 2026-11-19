# Hierarchical JEPA + Flow Matching | Medium Maze

## Actual achieved results (November 19, 2026)

**Stage 1: independently trained HWM and Macro Flow, with author's frozen
State JEPA and Low-level Flow.**

- Own collected medium-maze state/action episodes.
- 30-step HWM macro data: train 5,539, holdout 625.
- Independently trained HWM: 2,000 optimizer steps.
  Last recorded normalized prediction loss 0.0040, holdout 0.0059.
- Independently trained Macro Flow: 1,000 optimizer steps; 20,585
  goal-conditioned macro flow records. Only step-1 training loss printed.
- Closed-loop Medium Maze evaluation with official state JEPA and low-level
  Flow: 10/10 at seed 30000, 45/50 at seed 50000 => **55/60 = 91.7%**
  across two evaluation blocks. Official checkpoint baseline previously
  achieved 10/10 on seed 30000.
- **This is a partial-training reproduction**, not all 4 weights trained.

**Stage 2: state-assisted visual adaptation (already run)**

- Frozen Meta DINO vision encoder; fine-tune late action-conditioned Meta
  Predictor layers and train residual dynamics calibration/readout.
- Held-out future visual latent MSE 0.043234 vs identity 0.044197
  (2.18% improvement); negative-action ablation MSE ~0.04529 at best epoch 11.
- Closed-loop Meta candidate ranking + official Mini state HWM and Flow:
  9/10 at seeds 30000-30009. This is **not RGB-only navigation**: HWM,
  low-level action generator and proprio processing still see simulator state.
- Prior RGB-to-Mini latent bridge: V2 3/10, state/visual residual V4 8/10.
  These are different controller settings and not a controlled predictor-only
  improvement study.

## Completed four-component self-training (evaluated 2026-11-19)

Files:
- `scripts/collect_mini_medium_episodes.py`: native contiguous Episode .npz
- `scripts/run_mini_four_stage.sh`: JEPA -> low Flow -> HWM -> macro Flow -> eval
- `scripts/train_mini_medium_low_flow.py`: independent low walker using
  upstream `build_flow_net`, rectified-flow objective, future achieved-goal
  hindsight relabeling, and checkpoint compatible with upstream `LowLevelFlow`.

The low-level training is **our compatible training recipe**, not a verbatim
replication of the original author's specialized directed-walker data or
hyperparameters. The completed four-component training pipeline reached **50/50 successes (100%)** over 50 `PointMaze_Medium-v3` episodes using seed 60000–60049. This result is distinct from the partial-retraining 55/60 result.

Run:

```bash
cd ~/projects/JEPA-World-Model-Lab
git pull origin visual-hierarchical-jepa
conda activate jepa-wm311
bash scripts/run_mini_four_stage.sh all
```

After failed/partial stages, resume the stage without touching saved
artifacts:

```bash
bash scripts/run_mini_four_stage.sh jepa
bash scripts/run_mini_four_stage.sh low
bash scripts/run_mini_four_stage.sh hwm
bash scripts/run_mini_four_stage.sh macro
bash scripts/run_mini_four_stage.sh eval
```

Fast pipeline sanity check, with limited optimization (NOT a valid final
performance estimate):

```bash
JEPA_STEPS=50 LOW_STEPS=50 HWM_STEPS=50 MACRO_STEPS=50 \
EVAL_EPISODES=2 bash scripts/run_mini_four_stage.sh all
```

Completed workflow used the script defaults: 10k State JEPA / 12k low Flow / 4k HWM / 5k Macro Flow optimizer steps, batch size 128, followed by successful 50-Episode evaluation. These are the project's reproduction settings, not a claim of identical training data or parameters to the upstream author's original directed walker. Model and features are low dimensional, but GPU
memory allocation is not guaranteed; if OOM use `BATCH=64`.

Evaluation uses seeds 60000 onward, separate from previous 30000 and
50000 test blocks; check for collection seeds overlap and maintain
independent trajectory splits for quantitative reports.

## Resume positioning

**Historical partial baseline:** Independently retrained HWM + Macro Flow, 91.7% across 60 evaluated tasks, using official pretrained state encoder and walker. **Current full reproduction:** all four components independently trained, 50/50 on a fresh 50-Episode seed block. Visual dynamics head with held-out action ablations yielded 9/10 state-assisted control, NOT RGB-only.

**Measured completed four-stage result:** trained State JEPA, Low-level Flow, HWM and Macro Flow and evaluated using own checkpoints: **50/50 (100%)**, seed 60000, 50 episodes. In the last visible Macro Flow log: step 5000, train loss 0.0986. Do not apply the separate 55/60 partial-training result to these weights.

**Future RGB-only target**: Remove oracle simulator-state inputs from
both HWM and low Flow conditioning, or replace them with inferred state
from RGB. Current visual version is not yet fully visual.
