# MNIST Playground

Static SvelteKit MNIST lab for GitHub Pages. Models run in-browser through ONNX Runtime Web; metrics, samples, embeddings, and feature-map indexes are committed as static artifacts under `static/models`.

## App

- `/play`: draw or load a packaged sample, inspect preprocessing, prediction probabilities/logits, learned convolution traces, feature heatmaps, and linked 3D embedding/logit clouds.
- `/explore`: inspect model metrics, confusion matrix filtering, sample gallery, selected-sample probabilities, 3D embeddings, and feature map previews.

## Local Development

```bash
npm install
npm run dev
```

## Static Build

```bash
npm run check
npm run build
```

For GitHub Pages under a repository subpath:

```bash
BASE_PATH=/mnist-playground npm run build
```

## Model Artifacts

Each production model manifest entry points at:

- `model.onnx`
- `metrics.json`
- `samples/samples.json`
- `samples/samples.png`
- `embeddings/umap3d.json`
- `featuremaps/index.json`
- `model-card.md`

Validate the artifact contract:

```bash
python3 -m mnist_playground validate static/models
```

Run ONNX Runtime smoke inference for a packaged model:

```bash
python3 -m mnist_playground smoke static/models/cnn_small
```

If your global `python3` does not have `onnxruntime`, use the project venv or install `numpy onnx onnxruntime`.

## Model Factory

Shipped configs:

```bash
python3 -m mnist_playground configs --json
```

Train/package through the local pipeline:

```bash
python3 -m mnist_playground train --config cnn_fast --out static/models
```

Configs currently include `mlp_baseline`, `cnn_fast`, `cnn_strong`, `cnn_rotation_robust`, and `cnn_freedraw_robust`. The default production manifest model is `cnn_fast`; the strongest free-draw model is `CNN Free Draw Robust`.

`cnn_freedraw_robust` trains with ordinary MNIST plus targeted perturbations for free-draw failure cases: moderate rotations across all digits, mirrored/backwards `3` and `7`, and upside-down-ish `1`.

## Validation

```bash
npm test
npm run test:fuzz
npm run test:screenshots
```

- `npm test`: Playwright smoke coverage for `/play`, `/explore`, and mobile layout.
- `npm run test:fuzz`: seeded canvas, model-switching, prediction, and filtering stress tests.
- `npm run test:screenshots`: screenshot capture for `/play` and `/explore`.

Screenshot output: `artifacts/screenshots/`  
CI artifacts: screenshots, Playwright traces/reports, and `artifacts/validation-summary.md`
