"""Load official Meta JEPA-WM checkpoints for PointMaze or MetaWorld.

This verifies the PyTorch model and produces an environment-specific metadata
report. It does NOT perform planning, simulate an environment, or decode video.
Run from the repository root with Python 3.10 and official dependencies.
"""
import argparse
import json
from pathlib import Path

MODELS = {
    "pointmaze": "jepa_wm_pointmaze",
    "metaworld": "jepa_wm_metaworld",
}
BASELINES = {
    "pointmaze": "dino_wm_pointmaze",
    "metaworld": "dino_wm_metaworld",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=MODELS, required=True)
    parser.add_argument("--baseline", action="store_true", help="Load DINO-WM instead of JEPA-WM")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default="results/official")
    args = parser.parse_args()

    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    model_name = (BASELINES if args.baseline else MODELS)[args.env]
    print(f"Loading {model_name} on {device}. First launch may download checkpoints.")
    model, preprocessor = torch.hub.load(
        "facebookresearch/jepa-wms", model_name, pretrained=True, device=device
    )
    model.eval()
    count = sum(p.numel() for p in model.parameters())
    report = {
        "environment": args.env,
        "model": model_name,
        "device": device,
        "parameters": count,
        "model_class": type(model).__name__,
        "preprocessor_class": type(preprocessor).__name__,
        "status": "pretrained_model_loaded",
        "planning_evaluated": False,
        "note": "A successful load is not evidence of rollout accuracy or task success.",
    }
    out = Path(args.output_dir) / f"{model_name}_load.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved:", out)


if __name__ == "__main__":
    main()
