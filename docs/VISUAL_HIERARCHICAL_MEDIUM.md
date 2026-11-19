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
