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
