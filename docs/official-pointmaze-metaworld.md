# Official PyTorch JEPA-WMs: PointMaze and MetaWorld

This track is **separate** from the small CarRacing JEPA-inspired implementation
already in this repository. Official Meta models are loaded without changing
the existing training/evaluation scripts.

Official repository: https://github.com/facebookresearch/jepa-wms

## Scope

- PointMaze: goal-directed maze navigation.
- MetaWorld: robot-arm reach task (the first MetaWorld target).
- JEPA-WM and DINO-WM baselines for both tasks.
- Future visualization: environment RGB video, planned/actual trajectories,
  goal distance and success rate. **Not yet implemented here.**

## Step 1 — isolated PyTorch environment

The official project specifies Python 3.10, torch>=2.7, and torchvision==0.22.
Use an isolated conda environment, preferably Linux/WSL2 for complete simulation.
PointMaze's upstream D4RL/mujoco-py requires MuJoCo 2.1 and is particularly
problematic on native Windows. MetaWorld uses modern MuJoCo.

```bash
conda create -n jepa-wms python=3.10 -y
conda activate jepa-wms
git clone https://github.com/facebookresearch/jepa-wms.git
cd jepa-wms
pip install -e .
```

Consult the upstream README for FFmpeg, MuJoCo 2.1, environment variables,
and platform-specific installation. Installing the full package is not
guaranteed on native Windows.

## Step 2 — official checkpoint load smoke tests

From **this** JEPA-World-Model-Lab repository root, in the new environment:

```bash
python -m scripts.load_official_jepa_wms --env pointmaze
python -m scripts.load_official_jepa_wms --env metaworld
python -m scripts.load_official_jepa_wms --env pointmaze --baseline
python -m scripts.load_official_jepa_wms --env metaworld --baseline
```

Results: JSON reports under `results/official/`. These prove loading only;
they are **not** planning evaluations or videos.

## Step 3 — official planning evaluation

Upstream has evaluation templates:
- `configs/online_plan_evals/mz/mz_L1_cem_sourcerandstate_H6_nas6_ctxt2.yaml`
- `configs/online_plan_evals/mw/reach_L1_cem_sourcexp_H6_nas3_ctxt2.yaml`

The templates contain blank `model_kwargs.module_name` and
`model_kwargs.pretrain_kwargs`. A trained checkpoint and matching full
configuration are needed before running. This wrapper detects that and fails
with an actionable error instead of reporting false success.

```bash
python -m scripts.run_official_planning --env pointmaze --official-repo ../jepa-wms --config /path/to/completed_pointmaze.yaml --checkpoint /path/to/jepa_wm_pointmaze.pth.tar
python -m scripts.run_official_planning --env metaworld --official-repo ../jepa-wms --config /path/to/completed_metaworld.yaml --checkpoint /path/to/jepa_wm_metaworld.pth.tar
```

## Next work

1. Resolve the matching official eval configs and run one episode per environment.
2. Capture real simulator frames and planned vs actual trajectories.
3. Add an MP4 dashboard and JSON metrics (success rate, distance, elapsed time).
4. Compare JEPA-WM against DINO-WM with matched seeds and tasks.

Do not claim official planning success or rollout video until these stages run.
