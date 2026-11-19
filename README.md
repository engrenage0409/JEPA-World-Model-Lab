# JEPA World Model Lab — Hierarchical JEPA & Flow Matching for Long-Horizon Navigation

> **Validated on PointMaze Medium:** Four independently trained components (State JEPA, HWM, Macro Flow, Low-level Flow) reached **50/50 successful closed-loop episodes (100% on the tested seed block, seed 60000–60049)**. This is a state-based navigation system, **not pure RGB-only control**.

## Highlights

- **Full four-stage training:** state-action trajectory collection, State JEPA world-model learning, goal-conditioned low-level Rectified Flow, 30-step Hierarchical World Model, and feasible goal-conditioned Macro Flow.
- **Closed-loop MuJoCo PointMaze Medium evaluation:** 50/50 successes on an independently selected evaluation seed block for the fully self-trained pipeline. The earlier partial retraining (own HWM+Macro Flow, upstream state JEPA+walker) achieved **55/60 = 91.7%** across two blocks.
- **Experimental visual track:** Meta JEPA-WM/DINO RGB feature adaptation and partial fine-tuning of an action-conditioned AdaLN predictor. Best held-out future latent MSE **0.043234**, compared with **0.044197** for static/identity prediction (2.18% improvement); state-assisted visual candidate ranking achieved **9/10** navigation successes. This **does not** demonstrate a pure-vision agent or an improvement over the stronger state-based baseline.

## Architecture

```text
Simulator state s_t [not RGB-only]
        │
        ▼
State JEPA encoder E(s_t) → z_t
        │
        ▼
High-level encoder ψ(z_t) → abstract zᴴ_t
        │                         goal position
        └──────────┬────────────────────┘
                   ▼
     Goal-conditioned Macro Flow
       feasible 30-step macro proposals
                   │
                   ▼
     HWM g(zᴴ_t, macro) → predicted abstract future
                   │
                   ▼
      Subgoal decoder → reachable (x,y) subgoal
                   │
                   ▼
 Goal-conditioned Low-level Rectified Flow
         + state JEPA / raw state
                   │
                   ▼
       MuJoCo PointMaze actions
```

**Two separate experimental conditions:** The self-trained four-stage result uses the state-based architecture above. The visual experiment adds an RGB encoder and action-conditioned predictor to an existing state-assisted controller; its 9/10 score is **not** the score of the four-stage self-trained model with visual control.

## Verified results

| Condition | Model provenance | Evaluation | Success |
|---|---|---|---|
| Full four-stage reproduction | Own State JEPA + own Low-level Flow + own HWM + own Macro Flow; builds on open-source architecture | 50 episodes, seed 60000 | **50/50 (100%)** |
| Partial high-level reproduction | Own HWM (2,000 steps) + own Macro Flow (1,000 steps); upstream state JEPA + low-level Flow | 10 episodes, seed 30000; 50 episodes, seed 50000 | **55/60 (91.7%)** combined |
| State-assisted visual action reranking | Meta pretrained visual encoder + fine-tuned AdaLN predictor; upstream Mini state planner and controller | 10 episodes, seed 30000 | **9/10 (90%)** |

The 50/50 result is an **observed finite test set**, not a claim of guaranteed success on all tasks. This is an open-source-based reproduction and integration project, not an independently invented JEPA architecture.

## Training & evaluation (WSL / CUDA)

```bash
conda activate jepa-wm311
git clone https://github.com/engrenage0409/JEPA-World-Model-Lab.git
cd JEPA-World-Model-Lab
```

**Upstream dependencies are not vendored into this repository.** Obtain the compatible upstream [Mini-JEPA](https://github.com/WilliamZhang20/Mini-JEPA) source and place it at `external/mini-jepa/`, then install its dependencies in the project environment. The visual experiments separately require Meta's [JEPA-WM](https://github.com/facebookresearch/jepa-wms) source and relevant pretrained model weights at the expected paths. Scripts must not be run until their upstream imports, datasets and model weights are available. For reproducibility, record the upstream commit/version used alongside each run; the repository currently does not pin one.

To collect state-action trajectories:

```bash
python scripts/collect_mini_medium_episodes.py --steps 150000
```

Run four-component independent training and evaluate 50 previously unused environment seeds:

```bash
bash scripts/run_mini_four_stage.sh all
```

Run only the evaluation against the saved four-component weights:

```bash
bash scripts/run_mini_four_stage.sh eval
```

Default four-stage settings: State JEPA 10,000 steps; Low-level Flow 12,000; HWM 4,000; Macro Flow 5,000; Batch Size 128. For 8GB GPUs, try `BATCH=64` if CUDA runs out of memory. Initial training recipes are documented, but reported outcomes derive from the user's executed runs, not from CI reproduction on clean hardware.

**Large files:** `outputs/` and upstream vendor sources are gitignored. GitHub contains scripts and analysis, not saved .pt checkpoints, .npz trajectory data or .mp4 videos. Evaluation requires locally generated checkpoints.

## Documentation

- [Final technical report — Hierarchical JEPA & Flow Matching](docs/HIERARCHICAL_JEPA_FINAL_REPORT.md)
- [Four-stage reproduction and experimental provenance](docs/MINI_FOUR_STAGE_REPRODUCTION.md)
- [Visual action-conditioned predictor design](docs/VISUAL_HIERARCHICAL_MEDIUM.md)
- [Earlier teaching-oriented JEPA roadmap](docs/roadmap.md)

## Legacy educational scaffold

This repository also retains a lightweight CarRacing-v3 JEPA-inspired teaching scaffold (`jepa/`, and scripts such as `train_masked.py`, `train_action.py`, `plan_demo.py`). It is **not** the evaluated PointMaze navigation implementation and should not be confused with official V-JEPA or JEPA-WM pretraining.
