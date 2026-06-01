# MNIST Playground (static)

A GitHub Pages–friendly MNIST interactive playground.

## Goals
- **Snappy play**: draw a digit → classify instantly (ONNX Runtime Web in a **Web Worker**).
- **Deep explore** (static assets): confusion matrix, loss curves, feature-map “layer lens” (conv + pool), and **3D embedding** (UMAP) — all loaded as precomputed artifacts so the site stays static.
- **Extensible models**: add a new model by exporting ONNX + running a local packaging script that emits a folder of assets and an entry in `static/models/manifest.json`.

## Dev
```bash
cd mnist-playground
npm install
npm run dev
```

## Build (static)
```bash
npm run build
```

## Deploy to GitHub Pages
Use adapter-static output and configure `paths.base` if your repo is served under a subpath.

In SvelteKit, set base path in `svelte.config.js` / `src/app.html` depending on your deployment strategy.

A common approach:
- set `kit.paths.base` to `process.env.BASE_PATH ?? ''`
- in GitHub Actions, build with `BASE_PATH=/<repo>`

## Model assets
See `static/models/manifest.json`.

Placeholder assets live in `static/models/lenet_placeholder/`.

## Next build steps (planned)
- Explore tab: sample gallery (sprite-sheet) + linked confusion-cell → examples
- Layer lens: activation summaries + channel tiles (from precomputed featuremap atlases)
- 3D embeddings: three.js point cloud + click-to-select sample
- Packaging tool (`tools/package_model.py`) to export ONNX + metrics + embeddings + featuremaps
