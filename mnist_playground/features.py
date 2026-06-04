from __future__ import annotations

import json
from pathlib import Path


def export_conv_features(model_dir: Path) -> bool:
    try:
        import onnx
        from onnx import numpy_helper
    except Exception as exc:
        raise RuntimeError(f"onnx is required to export conv features: {exc}") from exc

    model_path = model_dir / "model.onnx"
    if not model_path.exists():
        return False

    model = onnx.load(model_path)
    tensors = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    if "conv1.weight" not in tensors or "conv2.weight" not in tensors:
        return False

    payload = {
        "schemaVersion": 1,
        "kind": "convnet_features",
        "layers": {
            "conv1": {
                "weights": tensors["conv1.weight"].astype("float32").tolist(),
                "bias": tensors["conv1.bias"].astype("float32").tolist(),
                "padding": 2,
                "stride": 1
            },
            "conv2": {
                "weights": tensors["conv2.weight"].astype("float32").tolist(),
                "bias": tensors["conv2.bias"].astype("float32").tolist(),
                "padding": 2,
                "stride": 1
            }
        },
        "samples": {}
    }
    out = model_dir / "featuremaps" / "index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    return True


def export_all(root: Path) -> list[str]:
    written: list[str] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if export_conv_features(model_dir):
            written.append(str(model_dir))
    return written
