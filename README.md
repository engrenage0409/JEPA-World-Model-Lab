# JEPA World Model Lab — Masked Feature Prediction + Action-Conditioned Dynamics

A **lightweight, educational JEPA-inspired World Model** for experimentation with a modest GPU (e.g. RTX 5060 8GB). Reuses `world-model-lab` CarRacing-v3 episodes. **No pretrained JEPA 2 weights or trained experiment results included.**

## Implemented vs planned

| Track | Current implementation | Not implemented yet |
|---|---|---|
| Masked JEPA | small patch-token ViT, EMA target encoder, masked next-frame target feature predictor | official V-JEPA 2 pretraining, space-time tubelet masking |
| Action JEPA | trainable action-conditioned pooled latent transition, multi-step rollout | official V-JEPA 2-AC, robot state/multi-view |
| Planning | goal-embedding CEM demo for latent terminal matching | real reward-driven control or task success verification |
| Evaluation | held-out episode latent MSE by future horizon, PCA trajectory visualization | linear probes, benchmark planning success, multi-seed ablations |

**Important:** The first stage uses a spatial patch mask to predict selected **next-frame** features from the preceding context frame. It is a pedagogical approximation, not the published V-JEPA objective or architecture. The second stage learns dynamics on mean-pooled latent features; both are small enough to prototype without large video pretraining.

## Architecture

```text
o_t   -> online patch ViT -> spatial tokens --masked predictor---> predicted target features
o_t+1 -> EMA target ViT  -> future patch token targets  ---------> masked feature loss
                             (no gradient through target encoder)

frozen EMA image encoder(o_t) -> pooled z_t
                   z_t + a_t -> ActionDynamics -> predicted z_t+1 -> ... -> z_t+H
frozen EMA image encoder(o_t+1...t+H) ----------------------------> latent MSE
                                                        |
                                          CEM samples action sequences
                                          to match a *supplied latent goal*
```

There is no image decoder. Visualization is a **latent-space plot**, not reconstructed images.

## Project structure

```text
jepa/
  data.py            CarRacing NPZ parsing, episode split, temporal windows
  models.py          Patch ViT, EMA masked predictor, action-conditioned MLP
  planning.py        CEM optimization for a supplied latent goal
scripts/
  inspect_data.py    Verify NPZ arrays
  train_masked.py    Train JEPA feature predictor and EMA teacher
  train_action.py    Train dynamics with a frozen target encoder
  evaluate.py        Per-horizon held-out latent prediction MSE
  visualize.py       PCA latent-trajectory plot
  plan_demo.py       Offline latent goal CEM example
tests/               Data, gradients/shapes, CEM tests
docs/                Original papers and staged research roadmap
```

## Setup on Windows PowerShell

Use the working PyTorch environment and install dependencies:

```powershell
git clone https://github.com/engrenage0409/JEPA-World-Model-Lab.git
cd JEPA-World-Model-Lab
python -m pip install -r requirements.txt
python -m pytest -q
```

## Step 1: reuse CarRacing trajectories

Copy `*.npz` from your local `world-model-lab` into `data/episodes/`. Each NPZ must contain:

```text
observations  (T+1,96,96,3) uint8
actions       (T,3)         float32
rewards       (T,)          float32
dones         (T,)          bool
```

Do not upload private/large episode files to Git. The dataset split is done **by episode**, not frame, to prevent leakage.

```powershell
python -m scripts.inspect_data
```

## Step 2: train JEPA-inspired feature encoder

```powershell
python -m scripts.train_masked --epochs 5 --steps 4 --batch-size 4 --mask-ratio 0.5
```

Output: `checkpoints/masked_jepa.pt`. Online encoder and predictor are optimized; target encoder gets EMA updates. Validation uses different episodes, but mask sampling is stochastic, so use fixed validation masks for rigorous comparisons. Also, guard against feature collapse before interpreting low masked loss as good learning.

## Step 3: train action-conditioned latent dynamics

```powershell
python -m scripts.train_action --epochs 5 --steps 4 --batch-size 4
```

Output: `checkpoints/action_dynamics.pt`. The frozen EMA target encoder produces a latent per image; the dynamics model predicts next latents conditioned on continuous steering/gas/brake, then rolls forward without future image input.

## Step 4: evaluate, visualize and try latent-goal CEM

```powershell
python -m scripts.evaluate
python -m scripts.visualize
python -m scripts.plan_demo
```

Outputs: `results/latent_eval.json`, `results/latent_trajectory.png`, and candidate action sequences from CEM.

**CEM here is offline goal-latent optimization; it is NOT an autonomous CarRacing controller.** The goal feature is extracted from a held-out future observation solely as a demonstration target. Evaluate real control only after adding a reward/cost model, environment interaction and safety checks.

## Research roadmap

See [docs/roadmap.md](docs/roadmap.md) for proper video-time masking, JEPA2-AC token dynamics, strong no-action baselines, planning, and multi-seed ablations. Paper links and official implementations: [docs/reading-list.md](docs/reading-list.md).

**Experiment status:** The code is a starter scaffold and has not yet been trained or benchmarked on the user's GPU. Do not claim official V-JEPA 2 reproduction or real-world planning results.
