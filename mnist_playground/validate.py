from __future__ import annotations

import json
import math
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REQUIRED_ASSET_KEYS = ["metrics", "samplesIndex", "samplesSprite", "embeddings3d", "featuremapsIndex", "modelCard"]


@dataclass
class ValidationReport:
    models_dir: Path
    checked_models: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def to_markdown(self) -> str:
        lines = [
            "# Validation Summary",
            "",
            f"- Models dir: `{self.models_dir}`",
            f"- Checked models: {', '.join(self.checked_models) if self.checked_models else 'none'}",
            f"- Status: {'pass' if self.ok else 'fail'}",
        ]
        if self.errors:
            lines.append("")
            lines.append("## Errors")
            lines.extend(f"- {e}" for e in self.errors)
        if self.warnings:
            lines.append("")
            lines.append("## Warnings")
            lines.extend(f"- {w}" for w in self.warnings)
        return "\n".join(lines)


def validate_models(models_dir: Path, min_default_accuracy: float = 0.97) -> ValidationReport:
    report = ValidationReport(models_dir=models_dir)
    manifest_path = models_dir / "manifest.json"
    manifest = _read_json(manifest_path, report)
    if not isinstance(manifest, dict):
        return report

    if manifest.get("schemaVersion") != 1:
        report.error("manifest schemaVersion must be 1")
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        report.error("manifest models must be a non-empty list")
        return report

    seen_ids: set[str] = set()
    default_id = manifest.get("defaultModelId")
    for model in models:
        if not isinstance(model, dict):
            report.error("manifest contains a non-object model entry")
            continue
        model_id = str(model.get("id", ""))
        report.checked_models.append(model_id or "<missing>")
        if not model_id:
            report.error("model entry missing id")
        if model_id in seen_ids:
            report.error(f"duplicate model id: {model_id}")
        seen_ids.add(model_id)
        _validate_model_entry(models_dir, model, report)

    if default_id not in seen_ids:
        report.error(f"defaultModelId {default_id!r} is not present in models")
    else:
        default = next(m for m in models if m.get("id") == default_id)
        metrics = _read_json(_asset_path(models_dir, default["assets"]["metrics"]), report)
        acc = ((metrics or {}).get("summary") or {}).get("testAcc")
        if not isinstance(acc, (int, float)) or acc < min_default_accuracy:
            report.error(f"default model accuracy {acc!r} is below threshold {min_default_accuracy}")

    external_data = sorted(models_dir.glob("**/*.onnx.data"))
    for path in external_data:
        report.error(f"external ONNX data file is not allowed: {path}")

    return report


def _validate_model_entry(models_dir: Path, model: dict[str, Any], report: ValidationReport) -> None:
    model_id = model.get("id", "<missing>")
    onnx_path = _asset_path(models_dir, str(model.get("onnxPath", "")))
    if not onnx_path.exists():
        report.error(f"{model_id}: missing ONNX file {onnx_path}")
    elif onnx_path.stat().st_size <= 0:
        report.error(f"{model_id}: empty ONNX file {onnx_path}")
    else:
        _validate_onnx_metadata(onnx_path, model, report)

    io = model.get("io")
    if not isinstance(io, dict) or not io.get("input") or not io.get("output"):
        report.error(f"{model_id}: io.input and io.output are required")
    norm = model.get("norm")
    if not isinstance(norm, dict) or not all(isinstance(norm.get(k), (int, float)) for k in ("mean", "std")):
        report.error(f"{model_id}: norm.mean and norm.std are required numbers")

    assets = model.get("assets")
    if not isinstance(assets, dict):
        report.error(f"{model_id}: assets object is required")
        return
    for key in REQUIRED_ASSET_KEYS:
        if key not in assets:
            report.error(f"{model_id}: missing assets.{key}")
            continue
        path = _asset_path(models_dir, assets[key])
        if not path.exists():
            report.error(f"{model_id}: missing asset {key}: {path}")

    metrics = _read_json(_asset_path(models_dir, assets.get("metrics", "")), report)
    if isinstance(metrics, dict):
        _validate_metrics(model_id, metrics, report)
    samples = _read_json(_asset_path(models_dir, assets.get("samplesIndex", "")), report)
    if isinstance(samples, dict):
        _validate_samples(model_id, samples, report)
    embeddings = _read_json(_asset_path(models_dir, assets.get("embeddings3d", "")), report)
    if isinstance(embeddings, dict) and isinstance(samples, dict):
        _validate_embeddings(model_id, embeddings, samples.get("count"), report)
    featuremaps = _read_json(_asset_path(models_dir, assets.get("featuremapsIndex", "")), report)
    if not isinstance(featuremaps, dict) or "samples" not in featuremaps:
        report.error(f"{model_id}: featuremaps index must contain samples")


def _validate_metrics(model_id: str, metrics: dict[str, Any], report: ValidationReport) -> None:
    summary = metrics.get("summary")
    if not isinstance(summary, dict):
        report.error(f"{model_id}: metrics.summary is required")
        return
    acc = summary.get("testAcc")
    loss = summary.get("testLoss")
    if not isinstance(acc, (int, float)) or not (0 <= acc <= 1):
        report.error(f"{model_id}: metrics.summary.testAcc must be in [0,1]")
    if not isinstance(loss, (int, float)) or not math.isfinite(loss):
        report.error(f"{model_id}: metrics.summary.testLoss must be finite")

    confusion = metrics.get("confusion")
    if not isinstance(confusion, dict):
        report.error(f"{model_id}: metrics.confusion is required")
        return
    labels = confusion.get("labels")
    matrix = confusion.get("matrix")
    if labels != list(range(10)):
        report.error(f"{model_id}: confusion labels must be 0..9")
    if not isinstance(matrix, list) or len(matrix) != 10 or any(not isinstance(r, list) or len(r) != 10 for r in matrix):
        report.error(f"{model_id}: confusion matrix must be 10x10")


def _validate_samples(model_id: str, samples: dict[str, Any], report: ValidationReport) -> None:
    count = samples.get("count")
    labels = samples.get("labels")
    sample_ids = samples.get("sampleIds")
    if samples.get("tileSize") != 28:
        report.error(f"{model_id}: samples tileSize must be 28")
    if not isinstance(count, int) or count <= 0:
        report.error(f"{model_id}: samples count must be positive")
        return
    if not isinstance(labels, list) or len(labels) != count:
        report.error(f"{model_id}: samples labels length must match count")
    if not isinstance(sample_ids, list) or len(sample_ids) != count:
        report.error(f"{model_id}: samples sampleIds length must match count")


def _validate_embeddings(model_id: str, embeddings: dict[str, Any], sample_count: Any, report: ValidationReport) -> None:
    coords = embeddings.get("coords3d")
    labels = embeddings.get("label")
    preds = embeddings.get("pred")
    if not isinstance(coords, list) or not isinstance(sample_count, int) or len(coords) != sample_count:
        report.error(f"{model_id}: embeddings coords3d length must match sample count")
    if not isinstance(labels, list) or len(labels) != sample_count:
        report.error(f"{model_id}: embeddings label length must match sample count")
    if not isinstance(preds, list) or len(preds) != sample_count:
        report.error(f"{model_id}: embeddings pred length must match sample count")


def _validate_onnx_metadata(onnx_path: Path, model: dict[str, Any], report: ValidationReport) -> None:
    try:
        import onnx
    except Exception as exc:
        report.warn(f"onnx package unavailable, skipped tensor metadata for {onnx_path}: {exc}")
        return
    graph = onnx.load_model(onnx_path, load_external_data=False).graph
    external = [t.name for t in graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL]
    if external:
        report.error(f"{model.get('id')}: ONNX uses external tensors: {', '.join(external[:5])}")
    input_names = {i.name for i in graph.input}
    output_names = {o.name for o in graph.output}
    io = model.get("io") or {}
    if io.get("input") not in input_names:
        report.error(f"{model.get('id')}: ONNX input {io.get('input')!r} not found in {sorted(input_names)}")
    if io.get("output") not in output_names:
        report.error(f"{model.get('id')}: ONNX output {io.get('output')!r} not found in {sorted(output_names)}")


def smoke_model(model_dir: Path) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    model_path = model_dir / "model.onnx"
    samples = _read_json_or_raise(model_dir / "samples" / "samples.json")
    pixels = _read_sprite_tile(model_dir / "samples" / "samples.png", samples, 0)
    # Static sample sprites are stored in native MNIST polarity:
    # black background, white digit. Use grayscale directly as ink intensity.
    x = pixels.astype("float32") / 255.0
    x = (x - 0.1307) / 0.3081
    x = x.reshape(1, 1, 28, 28)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {session.get_inputs()[0].name: x})
    logits = outputs[0]
    if list(logits.shape) != [1, 10]:
        raise ValueError(f"expected logits shape [1, 10], got {list(logits.shape)}")
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    if not bool(np.isfinite(probs).all()):
        raise ValueError("probabilities contain non-finite values")
    probability_sum = float(probs.sum())
    if abs(probability_sum - 1.0) > 1e-4:
        raise ValueError(f"probabilities sum to {probability_sum}, expected 1.0")
    return {
        "model": str(model_dir),
        "input": {"name": session.get_inputs()[0].name, "shape": list(x.shape)},
        "outputs": [{"name": o.name, "shape": list(v.shape)} for o, v in zip(session.get_outputs(), outputs)],
        "probabilitySum": probability_sum,
        "prediction": int(probs.argmax(axis=1)[0]),
        "finite": bool(np.isfinite(probs).all()),
    }


def _read_sprite_tile(path: Path, samples: dict[str, Any], tile_index: int):
    import numpy as np

    width, height, pixels = _read_png_gray(path)
    tile = int(samples["tileSize"])
    cols = int(samples["cols"])
    sx = (tile_index % cols) * tile
    sy = (tile_index // cols) * tile
    if sx + tile > width or sy + tile > height:
        raise ValueError("sample tile is outside sprite bounds")
    return np.vstack([
        np.frombuffer(row[sx:sx + tile], dtype="uint8")
        for row in pixels[sy:sy + tile]
    ])


def _read_png_gray(path: Path):
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"not a PNG: {path}")
    pos = 8
    width = height = color_type = None
    idat = bytearray()
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", payload)
            if bit_depth != 8 or color_type != 0:
                raise ValueError("only 8-bit grayscale PNG sprites are supported")
        elif tag == b"IDAT":
            idat.extend(payload)
        elif tag == b"IEND":
            break
    if width is None or height is None:
        raise ValueError("PNG is missing IHDR")
    raw = zlib.decompress(bytes(idat))
    stride = width
    rows: list[bytes] = []
    i = 0
    prev = bytearray(stride)
    for _ in range(height):
        filter_type = raw[i]
        i += 1
        row = bytearray(raw[i:i + stride])
        i += stride
        if filter_type == 1:
            for x in range(stride):
                row[x] = (row[x] + (row[x - 1] if x else 0)) & 255
        elif filter_type == 2:
            for x in range(stride):
                row[x] = (row[x] + prev[x]) & 255
        elif filter_type != 0:
            raise ValueError(f"unsupported PNG filter {filter_type}")
        rows.append(bytes(row))
        prev = row
    return width, height, rows


def _asset_path(models_dir: Path, asset: str) -> Path:
    if not asset:
        return models_dir / "<missing>"
    if asset.startswith("/models/"):
        return models_dir / asset.removeprefix("/models/")
    return models_dir / asset


def _read_json(path: Path, report: ValidationReport) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        report.error(f"failed to read JSON {path}: {exc}")
        return None


def _read_json_or_raise(path: Path) -> Any:
    return json.loads(path.read_text())
