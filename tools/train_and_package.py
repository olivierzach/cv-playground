#!/usr/bin/env python3
"""Train a few MNIST model variants and package static assets for the Svelte app.

Outputs under: static/models/<model_id>/
- model.onnx
- metrics.json (confusion matrix + curves + summary)
- samples/samples.png + samples/samples.json  (curated ~2K)
- embeddings/umap3d.json (3D UMAP from penultimate embeddings)

Featuremaps: scaffolded as TODO (can be added next; see notes in code).

Run:
  source .venv/bin/activate
  python tools/train_and_package.py --out static/models --epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
import umap


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


@dataclass
class PackSpec:
    model_id: str
    model_name: str


class MLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def embed(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class SmallCNN(nn.Module):
    def __init__(self, c1=16, c2=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, 5, padding=2)
        self.conv2 = nn.Conv2d(c1, c2, 5, padding=2)
        self.fc1 = nn.Linear(c2 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 14x14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 7x7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def embed(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj):
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))


def make_samples_sprite(images: np.ndarray, out_png: Path, tile=28, cols=50):
    """images: uint8 [N, 28, 28] grayscale"""
    N = images.shape[0]
    rows = math.ceil(N / cols)
    W = cols * tile
    H = rows * tile
    canvas = np.full((H, W), 255, dtype=np.uint8)
    for idx in range(N):
        r = idx // cols
        c = idx % cols
        canvas[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = images[idx]

    # tiny PNG writer (grayscale8)
    import struct, zlib

    raw = b"".join(b"\x00" + canvas[y].tobytes() for y in range(H))
    comp = zlib.compress(raw, 9)

    def chunk(tag, data):
        return struct.pack(">I", len(data)
        ) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", W, H, 8, 0, 0, 0, 0)
    png = sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", comp) + chunk(b"IEND", b"")
    ensure_dir(out_png.parent)
    out_png.write_bytes(png)


@torch.no_grad()
def eval_and_collect(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    all_logits = []
    all_y = []
    all_embed = []
    loss_sum = 0.0
    n = 0
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        loss_sum += float(loss.item())
        n += y.numel()
        all_logits.append(logits.cpu())
        all_y.append(y.cpu())
        if hasattr(model, "embed"):
            all_embed.append(model.embed(x).cpu())
    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)
    emb = torch.cat(all_embed, dim=0) if all_embed else None
    preds = logits.argmax(dim=1)
    acc = float((preds == y).float().mean().item())
    return {
        "logits": logits.numpy(),
        "y": y.numpy(),
        "pred": preds.numpy(),
        "acc": acc,
        "loss": loss_sum / max(1, n),
        "emb": emb.numpy() if emb is not None else None,
    }


def train_one(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: str, epochs: int, lr: float):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curves = {"trainLoss": [], "trainAcc": [], "valLoss": [], "valAcc": []}

    for ep in range(epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for x, y in tqdm(train_loader, desc=f"train ep{ep+1}/{epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            loss_sum += float(loss.item()) * y.size(0)
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.size(0))

        curves["trainLoss"].append(loss_sum / total)
        curves["trainAcc"].append(correct / total)

        ev = eval_and_collect(model, val_loader, device)
        curves["valLoss"].append(ev["loss"])
        curves["valAcc"].append(ev["acc"])

    return curves


def export_onnx(model: nn.Module, out_path: Path):
    ensure_dir(out_path.parent)
    model.eval()
    dummy = torch.zeros(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}},
    )

    # Ensure a single-file ONNX (no external data). Required for onnxruntime-web.
    import onnx
    m = onnx.load_model(out_path, load_external_data=True)
    onnx.save_model(m, out_path, save_as_external_data=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="static/models")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curated", type=int, default=2000)
    args = ap.parse_args()

    seed_all(args.seed)
    device = "cpu"

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    tfm_aug = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])

    ds_train = datasets.MNIST(".data", train=True, download=True, transform=tfm)
    ds_test = datasets.MNIST(".data", train=False, download=True, transform=tfm)

    # Curated: deterministic subset of test set
    curated_idx = list(range(min(args.curated, len(ds_test))))
    curated = Subset(ds_test, curated_idx)

    train_loader = DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=0)
    train_loader_aug = DataLoader(datasets.MNIST(".data", train=True, download=True, transform=tfm_aug), batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(curated, batch_size=256, shuffle=False, num_workers=0)

    packs = [
        (PackSpec("mlp_h256", "MLP (h=256)"), MLP(hidden=256)),
        (PackSpec("cnn_small", "Small CNN (16/32)"), SmallCNN(16, 32)),
        (PackSpec("cnn_good", "Good CNN (32/64, aug)"), SmallCNN(32, 64)),
    ]

    out_root = Path(args.out)
    ensure_dir(out_root)

    manifest_path = Path("static/models/manifest.json")
    manifest = {"models": []}

    for spec, model in packs:
        print(f"\n=== {spec.model_id}: {spec.model_name} ===")
        model.to(device)

        # Use augmentation only for the "good" CNN variant.
        loader = train_loader_aug if spec.model_id == 'cnn_good' else train_loader
        # Slightly longer by default for cnn_good even if caller uses small epochs.
        epochs = args.epochs if spec.model_id != 'cnn_good' else max(args.epochs, 5)
        curves = train_one(model, loader, val_loader, device, epochs, args.lr)
        ev = eval_and_collect(model, val_loader, device)

        # metrics
        cm = confusion_matrix(ev["y"], ev["pred"], labels=list(range(10))).tolist()
        metrics = {
            "summary": {"testAcc": ev["acc"], "testLoss": ev["loss"]},
            "curves": curves,
            "confusion": {"labels": list(range(10)), "matrix": cm},
        }

        model_dir = out_root / spec.model_id
        export_onnx(model, model_dir / "model.onnx")
        write_json(model_dir / "metrics.json", metrics)

        # samples sprite (unnormalized to uint8)
        imgs_u8 = []
        labels = []
        for x, y in tqdm(val_loader, desc="pack samples", leave=False):
            # x is normalized tensor [B,1,28,28]. Convert back to [0,255].
            x = x * MNIST_STD + MNIST_MEAN
            x = torch.clamp(x, 0, 1)
            u8 = (x[:, 0].cpu().numpy() * 255.0).astype(np.uint8)
            imgs_u8.append(u8)
            labels.extend(y.cpu().numpy().tolist())
        imgs_u8 = np.concatenate(imgs_u8, axis=0)

        samples_dir = model_dir / "samples"
        ensure_dir(samples_dir)
        make_samples_sprite(imgs_u8, samples_dir / "samples.png", cols=50)
        write_json(samples_dir / "samples.json", {
            "tileSize": 28,
            "cols": 50,
            "count": int(imgs_u8.shape[0]),
            "labels": labels,
            "sampleIds": curated_idx,
        })

        # embeddings (UMAP 3D) from penultimate layer (embed)
        if ev["emb"] is not None:
            reducer = umap.UMAP(n_components=3, random_state=args.seed, n_neighbors=25, min_dist=0.1)
            coords = reducer.fit_transform(ev["emb"]).astype(np.float32)
            # confidence from softmax max
            logits = ev["logits"]
            ex = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = ex / ex.sum(axis=1, keepdims=True)
            conf = probs.max(axis=1)
            out = {
                "sampleIds": curated_idx,
                "coords3d": coords.tolist(),
                "label": ev["y"].tolist(),
                "pred": ev["pred"].tolist(),
                "correct": (ev["y"] == ev["pred"]).tolist(),
                "confidence": conf.tolist(),
            }
            write_json(model_dir / "embeddings" / "umap3d.json", out)

        # featuremaps placeholder index (real featuremap atlases are next step)
        write_json(model_dir / "featuremaps" / "index.json", {"samples": {}})

        # append manifest entry
        entry = {
            "id": spec.model_id,
            "name": spec.model_name,
            "onnxPath": f"/models/{spec.model_id}/model.onnx",
            "io": {"input": "input", "output": "logits"},
            "norm": {"mean": MNIST_MEAN, "std": MNIST_STD},
            "assets": {
                "metrics": f"/models/{spec.model_id}/metrics.json",
                "samplesIndex": f"/models/{spec.model_id}/samples/samples.json",
                "samplesSprite": f"/models/{spec.model_id}/samples/samples.png",
                "embeddings3d": f"/models/{spec.model_id}/embeddings/umap3d.json",
                "featuremapsIndex": f"/models/{spec.model_id}/featuremaps/index.json",
            },
            "layers": [
                {"id": "conv1", "displayName": "Conv 1", "channels": 16, "tile": [28, 28]},
                {"id": "pool1", "displayName": "Pool 1", "channels": 16, "tile": [14, 14]},
                {"id": "conv2", "displayName": "Conv 2", "channels": 32, "tile": [14, 14]},
                {"id": "pool2", "displayName": "Pool 2", "channels": 32, "tile": [7, 7]},
            ],
        }
        if spec.model_id.startswith("mlp"):
            entry["layers"] = []
        manifest["models"].append(entry)

    write_json(manifest_path, manifest)
    print(f"\nWrote manifest: {manifest_path}")
    print("Done. Re-run dev server to pick up new static assets if needed.")


if __name__ == "__main__":
    main()
