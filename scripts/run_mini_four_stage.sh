#!/usr/bin/env bash
# Mini-JEPA Medium Maze: train FOUR independent components, preserving all old checkpoints.
# Run from repository root in WSL with the jepa-wm311 environment.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/external/mini-jepa:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

DATA="outputs/mini_scratch/medium_episodes.npz"
OUT="outputs/mini_scratch/full"
mkdir -p "$OUT"
if [[ ! -f "$DATA" ]]; then
  echo "Missing $DATA. First run: python scripts/collect_mini_medium_episodes.py --steps 150000"
  exit 1
fi
STAGE="${1:-all}"
if [[ "$STAGE" == "all" || "$STAGE" == "jepa" ]]; then
  echo "[1/4] Train OWN State JEPA from scratch (no pretrained init)"
  python -m jepa_robotics.train \
    --task point_medium --episodes-npz "$DATA" \
    --model-path "$OUT/point_medium_jepa_model.pt" \
    --save-path "$OUT/point_medium_jepa_checkpoint.pt" \
    --latent-dim 192 --hidden-dim 256 \
    --train-steps "${JEPA_STEPS:-10000}" --batch-size "${BATCH:-128}" \
    --eval-episodes 1 --device cuda
fi
if [[ "$STAGE" == "all" || "$STAGE" == "low" ]]; then
  echo "[2/4] Train OWN goal-conditioned Flow walker"
  python scripts/train_mini_medium_low_flow.py \
    --jepa "$OUT/point_medium_jepa_model.pt" \
    --episodes-npz "$DATA" --out "$OUT/medium_low_flow.pt" \
    --train-steps "${LOW_STEPS:-12000}" --batch-size "${BATCH:-128}" \
    --device cuda
fi
if [[ "$STAGE" == "all" || "$STAGE" == "hwm" ]]; then
  echo "[3/4] Train OWN 30-step high-level JEPA world model"
  python external/mini-jepa/scripts/train/train_hjepa_hwm.py \
    --task point_medium \
    --model-path "$OUT/point_medium_jepa_model.pt" \
    --episodes-npz "$DATA" --out "$OUT/medium_hwm.pt" \
    --stride 30 --steps "${HWM_STEPS:-4000}" \
    --batch-size "${BATCH:-128}" --device cuda
fi
if [[ "$STAGE" == "all" || "$STAGE" == "macro" ]]; then
  echo "[4/4] Train OWN high-level macro Flow"
  python external/mini-jepa/scripts/train/train_hwm_macro_flow.py \
    --task point_medium \
    --model-path "$OUT/point_medium_jepa_model.pt" \
    --hwm "$OUT/medium_hwm.pt" --episodes-npz "$DATA" \
    --out "$OUT/medium_macro_flow.pt" \
    --steps "${MACRO_STEPS:-5000}" --batch-size "${BATCH:-128}" --device cuda
fi
if [[ "$STAGE" == "all" || "$STAGE" == "eval" ]]; then
  echo "[EVAL] FOUR self-trained models; fresh seeds"
  python external/mini-jepa/scripts/eval/eval_hjepa_hwm.py \
    --task point_medium \
    --jepa-model "$OUT/point_medium_jepa_model.pt" \
    --bc-policy "$OUT/medium_low_flow.pt" --low-type flow \
    --hwm "$OUT/medium_hwm.pt" \
    --macro-flow "$OUT/medium_macro_flow.pt" \
    --macro-flow-samples 16 --macro-flow-horizon 1 \
    --episodes "${EVAL_EPISODES:-50}" --seed "${EVAL_SEED:-60000}" \
    --device cuda --video-out "$OUT/medium_four_stage_nav.mp4" \
    --video-episodes 1
fi
