"""Run the upstream JEPA-WMs evaluation with an explicit complete config.

The official template configs have empty model_kwargs fields. This launcher
refuses to silently run an unconfigured template. Configure a checkpoint and
model_kwargs following the official repo first.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

CONFIGS = {
    "pointmaze": "configs/online_plan_evals/mz/mz_L1_cem_sourcerandstate_H6_nas6_ctxt2.yaml",
    "metaworld": "configs/online_plan_evals/mw/reach_L1_cem_sourcexp_H6_nas3_ctxt2.yaml",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=CONFIGS, required=True)
    p.add_argument("--official-repo", type=Path, required=True)
    p.add_argument("--config", type=Path, help="Completed YAML config; defaults to upstream template")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/official_planning"))
    p.add_argument("--allow-template", action="store_true", help="Skip incomplete-config guard only if you know the upstream setup")
    args = p.parse_args()

    repo = args.official_repo.expanduser().resolve()
    config = args.config.expanduser().resolve() if args.config else repo / CONFIGS[args.env]
    checkpoint = args.checkpoint.expanduser().resolve()
    if not (repo / "evals" / "main.py").is_file():
        p.error(f"Official repository missing at {repo}")
    if not config.is_file():
        p.error(f"Config not found: {config}")
    if not checkpoint.is_file():
        p.error(f"Checkpoint not found: {checkpoint}")

    import yaml
    with config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    kwargs = cfg.get("model_kwargs") or {}
    if not args.allow_template and (
        not kwargs.get("module_name") or not kwargs.get("pretrain_kwargs")
    ):
        p.error(
            "Official evaluation template has incomplete model_kwargs. "
            "Supply --config with a complete model configuration from the "
            "matching official training/evaluation setup. "
            "A PyTorch Hub load alone does not fill these fields."
        )

    output = args.output_dir.expanduser().resolve() / args.env
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "evals.main",
        "--fname", str(config),
        "--checkpoint", str(checkpoint),
        "--folder", str(output),
        "--debug",
    ]
    print("Running:", " ".join(command), flush=True)
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "glfw")
    subprocess.run(command, cwd=repo, env=env, check=True)


if __name__ == "__main__":
    main()
