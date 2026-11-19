# Visual Hierarchical JEPA — Medium Maze (experimental)

This branch preserves official Meta JEPA-WM and Mini-JEPA and adds a trainable visual-to-hierarchical latent bridge.

**Important:** This is not yet a fully visual world model. Meta's visual encoder is frozen; the new projection learns to feed Mini-JEPA's high-level HWM. Mini-JEPA's lower-level Flow controller still uses privileged state. The Meta action-conditioned predictor is not yet integrated.

## Setup

From ~/projects/JEPA-World-Model-Lab in WSL, with external/mini-jepa already cloned and its four Medium LFS checkpoints downloaded:

~~~bash
conda activate jepa-wm311
export MUJOCO_GL=osmesa
export PYTHONNOUSERSITE=1

# Collect aligned camera frames and teacher state latents
python scripts/visual_medium_bridge.py collect --episodes 30 --device cuda --every 4 --history 4

# Train a small bridge with Meta image encoder frozen
python scripts/visual_medium_bridge.py train --device cuda --epochs 35 --batch-size 2

# Evaluate the new visual HIGH-LEVEL planner on unseen seeds
python scripts/visual_medium_bridge.py eval --device cuda --episodes 10 --seed 30000 --history 4
~~~

Data and model outputs are saved in outputs/visual_medium/. Do not commit large runtime data or checkpoints.

## Mechanism

1. Collect uses Mini-JEPA's official state-based hierarchical flow planner to generate paired RGB frames and teacher state-JEPA latents.
2. Train loads the existing official Meta jepa_wm_pointmaze checkpoint and calls model.encode on the RGB frames; a small trainable projection maps pooled Meta visual representations to Mini's HWM input latent.
3. Eval replaces Mini's high-level input latent with the projected visual representation. Low-level motion still uses Mini's pretrained state policy.

The validation split holds out entire episodes to reduce frame leakage. The evaluation seed 30000 differs from collection seed 20000.

## Interpretation and limitations

- The prior 10/10 Medium Maze result is from unmodified Mini-JEPA, not from this visual bridge.
- This bridge has not yet been evaluated against a real simulator in this chat. No hybrid success rate is claimed.
- Visual domain mismatch may prevent accurate feature alignment. Check training validation MSE and actual navigation success.
- Full visual JEPA-WM planning remains future work: adapt the original Meta action-conditioned predictor and remove privileged low-level state.

Source repos: facebookresearch/jepa-wms and WilliamZhang20/Mini-JEPA.

## V2 after first visual run
The initial global-mean-pooled single-frame bridge scored 1/10 on eval seeds 30000-30009 (best validation MSE 0.6052). V2 preserves Meta spatial tokens as a 4x4 grid and adds a four-step temporal image pair; collect and train must be rerun. V2 is not yet tested and does not use the Meta action-conditioned predictor. No V2 success rate is asserted.


## Shadow planning diagnosis (no training required)

The current V3 visual bridge achieved 2/10 on seeds 30000-30009 versus 3/10 for V2. Before further training, compare each replan with the Mini teacher at the **same exact observed state**.

```bash
python scripts/visual_medium_bridge.py eval \
  --device cuda --episodes 3 --seed 30000 --history 4 \
  --output outputs/visual_medium/visual_bridge_v3.pt \
  --compare-high-level \
  --comparison-out outputs/visual_medium/v3_high_level.jsonl \
  --video-out ''
```

The real controller always uses **visual** subgoals. A separate Mini teacher proposes an alternate subgoal using a snapshot/restoration of Torch/NumPy RNG state to match sampling and avoid perturbing the real rollout. Results log high-latent MSE/cosine, subgoal distance and both subgoal coordinates, plus per-episode summaries. The teacher is evaluated at **the visual policy's encountered states**; this is not a separate teacher rollout and does not itself prove which subgoal is optimal.

Interpretation: if high latents and chosen subgoals diverge immediately, improve representation alignment; if they remain close yet both fail, inspect macro planning, goal semantics and low-level tracking. Neither outcome should be presumed in advance.


## V4: conservative Visual + State residual fusion

This makes **a state-assisted multimodal planner**, not pure visual control.

Fusion rule:

    z_fused = z_state + alpha * clip_by_rms(z_visual - z_state, max_ratio * rms(z_state))

The Mini state JEPA/HWM, macro-flow, low-level flow and the trained V2/V3 visual
bridge remain frozen. No fresh training or data are necessary.

From the repository root:

~~~bash
git pull origin visual-hierarchical-jepa
python -m pytest -q tests/test_visual_fusion.py

# Pure Mini state baseline: exact teacher latent at HWM input
python scripts/visual_medium_bridge.py eval --device cuda --episodes 10 --seed 30000 \
  --fusion-alpha 0 \
  --video-out outputs/visual_medium/v4_state.mp4

# Conservative residual mix using V2 bridge:
python scripts/visual_medium_bridge.py eval --device cuda --episodes 10 --seed 30000 --history 4 \
  --output outputs/visual_medium/visual_bridge.pt \
  --fusion-alpha 0.15 --fusion-max-ratio 0.5 \
  --video-out outputs/visual_medium/v4_fused.mp4
~~~

If your saved V2 checkpoint has a different name, supply that path instead.
To test V3, supply visual_bridge_v3.pt. The default alpha=1 and practically
uncapped max-ratio preserve the historical visual-only path for existing commands.

Fusion is a **fixed user-controlled gate**, not a newly trained network. This
protects the high-performing state baseline and tests whether the pretrained
Meta visual representation can be added without destabilizing the hierarchy.
If alpha is small, a strong score does not prove visual features added benefit.
The original Meta action-conditioned predictor is not involved.

Both eval runs use identical task seeds, but the flow policy and planner
can consume stochastic samples; compare logged success rather than expecting
bit-for-bit identical trajectories. All results require local testing.


## Action-conditioned Medium Maze milestone (Meta Predictor actually used)

The script `scripts/medium_action_predictor.py` adds an **independent**
research pipeline. Unlike V2–V4 VisualBridge experiments, this pipeline
actually calls `Meta VideoWM.forward_pred(visual_context, action_features, proprio_context)` and uses the
predicted visual latent to **rank low-level Flow action candidates**.

Architecture:

1. **High-level:** frozen pretrained Mini state-JEPA -> HWM -> Macro Flow
   outputs an XY subgoal on Medium Maze.
2. **Low-level proposals:** pretrained Mini LowLevelFlow samples multiple
   5-step, 2-D action chunks conditioned on the same subgoal.
3. **Visual world model:** frozen pretrained Meta DINO visual encoder
   encodes the last two RGB frames, 5 simulation steps apart. The **frozen
   action-conditioned Meta AdaLN Predictor** rolls the visual latent forward
   conditioned on each proposed action chunk (flattened 5x2 to 10D, with
   Meta PointMaze mean/std normalization plus trained action adapter).
4. **Learned readout:** a trained 2-D position head projects the predicted
   visual latent to XY. Choose the action chunk whose predicted XY is
   closest to the current HWM subgoal, execute 5 actions, then replan.
5. **Supervision:** paired RGB, true executed 5x2 actions, and observed
   XY labels are from the Mini controller's environment trajectories.
   Only the small input action adapter and latent-to-XY head train.
   Meta Encoder/Predictor and Mini hierarchy remain frozen.

### Commands (run from repository root, WSL + jepa-wm311)

~~~bash
git pull origin visual-hierarchical-jepa
python -m pytest tests/test_medium_action_predictor.py -q

# Collect a limited dataset of 5-step RGB/action transitions.
python scripts/medium_action_predictor.py collect \
  --device cuda --episodes 25 --seed 20000 --max-pairs 600

# Train and validate (frozen Meta Predictor, trainable 10D action adapter).
python scripts/medium_action_predictor.py train \
  --device cuda --head-epochs 4 --epochs 3

# New actual model-based action selection.
python scripts/medium_action_predictor.py eval \
  --device cuda --episodes 10 --seed 30000 \
  --planner meta --candidates 4

# Matched-seed low-level Flow-only baseline.
python scripts/medium_action_predictor.py eval \
  --device cuda --episodes 10 --seed 30000 --planner flow
~~~

The training log includes held-out `valid_future_latent_mse`,
`valid_future_xy_error`, and `valid_negated_action_latent_mse`
(an action-conditioning diagnostic). Evaluation writes per-candidate scores,
predicted XY and the selected index to
`outputs/medium_action/decisions.jsonl`. Do not claim the method improves
Medium-Maze success until both models have been evaluated on the **same
task seeds**; also check that correct action conditioning beats ablated
conditioning on the held-out transitions. The model-based planner may
be slower than the state-only controller due to multiple visual model
rollouts at every five-step replanning interval.

**Scientific boundaries:** The navigation is **state-assisted**: Mini HWM
and low-level proposals consume true simulator states. Meta Predictor is
pretrained; the added action-input adapter and future-visual-state readout are
independently trained on Medium Maze transitions. This is action-conditioned
visual model-based reranking, **not** end-to-end pixel-only navigation and not
end-to-end Meta backbone fine-tuning.

**Implementation caution:** This is a first implementation requiring runtime
validation with the exact WSL dependencies and checkpoints. In particular,
Meta's pretraining used fixed 5-step action packing, a different underlying
PointMaze data distribution, and its specific action normalization statistics.
If the predicted visual transition is not action-sensitive or if XY readout
error is large, do not interpret downstream success as an effect of Meta.

### Suggested resume wording *after evaluation*

"适配 Meta JEPA-WM 预训练视觉编码器和动作条件 AdaLN Predictor，
针对 Medium Maze 构建 5-step 动作序列归一化与轻量适配训练，
以未来视觉潜在状态预测和位置读出筛选 Mini-JEPA 分层控制候选动作；
在相同测试种子上与纯状态 Flow 基线进行对照评估。"

Quantitative performance numbers may be added only once confirmed.

### 2026-11-19: AdaLN dimension fix

An early training run failed at AdaLN LayerNorm: expected 400 channels,
received 384. The pretrained PointMaze predictor concatenates **384 visual
features + 16 encoded proprio features** for each patch. The corrected
`collect` stage saves 4D simulator proprio state (x, y, vx, vy) for each
input frame, and `train/eval` call `Meta.encode({"visual":..., "proprio":...})`
so the official predictor receives its required additional 16 features.

The first adaptation also exposed a potential temporal mismatch in the
upstream `EncPredWM.unroll` API when used with a two-frame visual context
and only one action block: AdaLN operates on matched visual/action time
dimensions. Our explicit one-step `meta.model.forward_pred` uses the
current visual/proprio embedding and a single 10D packed action; the
Meta pretrained AdaLN Predictor remains involved in the differentiable
forward path. The two-frame dataset remains available for future
history-aware rollouts, but only the *current* frame is used for this
one-step predictor.

**Regenerate the previous transition dataset** after pulling this fix;
the original dataset did not save proprio states. The new training script
now runs a small Predictor smoke test *before* the position-head epochs,
so interface failures happen immediately instead of after several epochs.
The earlier four head epochs were not checkpointed prior to the crash.


### 2026-11-19: first full action training run and spatial position readout

The first completed run produced:
- held-out true-action future latent MSE: **4.3036**
- held-out *negated*-action latent MSE: **4.2798** (lower / worse)
- future XY error: **2.4871** simulator units

Thus, despite successful forward and backward computations, action-conditioned
forecasting has NOT shown value over the negated-action control. Do not cite
the model as a successful predictor yet. We found the original position
head averaged away all spatial token layout before its XY regression.

The updated position head preserves pooled **4x4** spatial tokens, and train
now prints:
- `position_validation.mean_xy_error` on held-out episodes,
- `valid_identity_latent_mse` as a no-dynamics baseline, and
- `correct_minus_negated_mse` (negative is preferable).

No new data collection is necessary for this change; the new head architecture
requires training a new checkpoint (old heads are incompatible).

~~~bash
git pull origin visual-hierarchical-jepa
python scripts/medium_action_predictor.py train \
  --device cuda --head-epochs 4 --epochs 3 \
  --checkpoint outputs/medium_action/meta_action_adapter_spatial.pt
~~~

Only evaluate navigation if the visual XY readout and action-conditioned
transition diagnostics look credible. Avoid overclaiming predictor benefits
from raw navigation success if action ablations remain non-discriminative.
