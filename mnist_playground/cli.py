from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

from .features import export_all, export_conv_features
from .validate import smoke_model, validate_models


CONFIG_DIR = Path(__file__).with_name("configs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mnist_playground")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train", help="Train and package local MNIST models.")
    train.add_argument("--out", default="static/models")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--curated", type=int, default=2000)
    train.add_argument("--config", choices=available_configs(), default="cnn_fast")

    package = sub.add_parser("package", help="Package models using the local training pipeline.")
    package.add_argument("--out", default="static/models")
    package.add_argument("--epochs", type=int, default=3)
    package.add_argument("--seed", type=int, default=0)
    package.add_argument("--config", choices=available_configs(), default="cnn_fast")

    validate = sub.add_parser("validate", help="Validate static model artifacts.")
    validate.add_argument("models_dir", nargs="?", default="static/models")
    validate.add_argument("--min-default-accuracy", type=float, default=0.97)

    smoke = sub.add_parser("smoke", help="Run ONNX Runtime smoke inference for one model folder.")
    smoke.add_argument("model_dir")

    features = sub.add_parser("features", help="Export learned convolution filters from packaged ONNX models.")
    features.add_argument("path", nargs="?", default="static/models")

    configs = sub.add_parser("configs", help="List shipped model configs.")
    configs.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd in {"train", "package"}:
        return run_training_script(args)
    if args.cmd == "validate":
        report = validate_models(Path(args.models_dir), min_default_accuracy=args.min_default_accuracy)
        print(report.to_markdown())
        return 0 if report.ok else 1
    if args.cmd == "smoke":
        result = smoke_model(Path(args.model_dir))
        print(json.dumps(result, indent=2))
        return 0
    if args.cmd == "features":
        path = Path(args.path)
        written = export_all(path) if path.is_dir() and (path / "manifest.json").exists() else ([str(path)] if export_conv_features(path) else [])
        print(json.dumps({"written": written}, indent=2))
        return 0
    if args.cmd == "configs":
        configs_payload = {name: load_config(name) for name in available_configs()}
        print(json.dumps(configs_payload, indent=2) if args.json else "\n".join(configs_payload))
        return 0

    parser.error(f"unknown command {args.cmd}")
    return 2


def available_configs() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.json"))


def load_config(name: str) -> dict:
    return json.loads((CONFIG_DIR / f"{name}.json").read_text())


def run_training_script(args: argparse.Namespace) -> int:
    script = Path("tools/train_and_package.py")
    if not script.exists():
        raise SystemExit("tools/train_and_package.py is missing")
    sys.argv = [
        str(script),
        "--out",
        args.out,
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
        "--config",
        args.config,
    ]
    if hasattr(args, "lr"):
        sys.argv.extend(["--lr", str(args.lr)])
    runpy.run_path(str(script), run_name="__main__")
    return 0
