<script lang="ts">
  import { onMount } from 'svelte';
  import { appPath } from '$lib/paths';
  import { loadModelRegistry } from '$lib/models/registry';
  import type { ModelEntry } from '$lib/models/types';
  import { conv1TraceFromInk, learnedConvFeatureMaps, loadConvFeatureSpec, type ConvFeatureSpec } from '$lib/models/convnet';
  import { OrtClient } from '$lib/ort/client';
  import { downsampleTo28x28, imageDataToMnistTensor } from '$lib/ort/preprocess';
  import { renderLogitMultiples, renderProbBars } from '$lib/viz/probBars';
  import { loadImage, drawTileInverted, type SpriteIndex } from '$lib/viz/spriteSheet';
  import { featureMapsFromInk, renderHeatmapSet, type FeatureMapSpec } from '$lib/viz/heatmap';
  import { mountUmapD3, type Umap3D, type UmapHover, type UmapQuery } from '$lib/viz/umapPlot';
  import { buildLogitPcaSpace, type LogitPcaSpace } from '$lib/viz/logitSpace';
  import { renderConvTrace } from '$lib/viz/convTrace';
  import { renderCnnArchitecture } from '$lib/viz/cnnArchitecture';

  type Neighbor = { tile: number; label: number; sampleId: number; reason: string; distance: number };
  type PlotHandle = {
    dispose(): void;
    render(): void;
    setSelected(index: number | null): void;
    setHighlightLabel(label: number | null): void;
    setQuery(query: UmapQuery | null): void;
  };

  let models: ModelEntry[] = $state([]);
  let model: ModelEntry | null = $state(null);
  let client: OrtClient | null = $state(null);
  let status = $state('loading models...');

  let drawCanvas: HTMLCanvasElement;
  let previewCanvas: HTMLCanvasElement;
  let preprocessEl: HTMLDivElement;
  let convTraceEl: HTMLDivElement;
  let architectureEl: HTMLDivElement;
  let heatmapsEl: HTMLDivElement;
  let umapEl: HTMLDivElement;
  let logitUmapEl: HTMLDivElement;
  let isDown = $state(false);
  let lastX = $state(0);
  let lastY = $state(0);

  let pred = $state('-');
  let conf = $state(0);
  let topK: { digit: number; p: number }[] = $state([]);
  let neighbors: Neighbor[] = $state([]);

  let sampleLabel: number | null = $state(null);
  let sampleTile: number | null = $state(null);
  let sampleIdx: SpriteIndex | null = $state(null);
  let sampleSprite: HTMLImageElement | null = $state(null);
  let sampleVectors: Float32Array[] | null = null;
  let convSpec: ConvFeatureSpec | null = null;
  let umap: Umap3D | null = $state(null);
  let umapHandle: PlotHandle | null = null;
  let umapHover: UmapHover = $state(null);
  let queryDetail = $state('Predict a drawn or sampled digit to place it into the embedding.');
  let logitSpace: LogitPcaSpace | null = null;
  let logitHandle: PlotHandle | null = null;
  let logitHover: UmapHover = $state(null);
  let logitQueryDetail = $state('Logit decision space will appear after the model is ready.');
  let logitBuildSeq = 0;
  let latestLogits: Float32Array | null = null;
  let latestProbs: Float32Array | null = null;
  let latestBestLabel: number | null = null;
  let focusedDigit: number | null = $state(null);
  let featureMaps: FeatureMapSpec[] = $state([]);

  let barsEl: HTMLDivElement;
  let logitsEl: HTMLDivElement;

  function resetPrediction() {
    pred = '-';
    conf = 0;
    topK = [];
    neighbors = [];
    queryDetail = 'Predict a drawn or sampled digit to place it into the embedding.';
    logitQueryDetail = logitSpace
      ? 'Predict to place the actual 10-logit vector in decision space.'
      : 'Logit decision space will appear after the model is ready.';
    latestLogits = null;
    latestProbs = null;
    latestBestLabel = null;
    focusedDigit = null;
    umapHandle?.setQuery(null);
    logitHandle?.setQuery(null);
    restorePlotFocus();
    renderProbBars(barsEl, new Float32Array(10));
    renderLogitMultiples(logitsEl, new Float32Array(10));
  }

  function clear() {
    const ctx = drawCanvas.getContext('2d')!;
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    sampleLabel = null;
    sampleTile = null;
    resetPrediction();
    renderPreviews(downsampleTo28x28(drawCanvas));
  }

  async function ensureModelLoaded() {
    if (!model) return;
    client?.dispose();
    client = new OrtClient();
    status = `loading model: ${model.name}...`;
    try {
      await client.load(model.onnxPath, model.io.input, model.io.output);
      status = 'ready';
    } catch (e: any) {
      status = e?.message ?? String(e);
    }
  }

  async function predictNow() {
    if (!model) return;
    if (!client || status !== 'ready') {
      await ensureModelLoaded();
    }
    if (!client) return;

    const img28 = downsampleTo28x28(drawCanvas);
    const ink = renderPreviews(img28);
    const x = imageDataToMnistTensor(img28, model.norm.mean, model.norm.std);
    const { logits, probs } = await client.predict(x, [1, 1, 28, 28]);

    let bestI = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[bestI]) bestI = i;
    pred = String(bestI);
    conf = probs[bestI];
    topK = Array.from(probs, (p, digit) => ({ digit, p })).sort((a, b) => b.p - a.p).slice(0, 3);
    neighbors = await nearestExamples(ink, bestI);
    await placeQueryInEmbedding(ink, probs, bestI);
    placeQueryInLogitSpace(logits, probs, bestI);
    renderProbBars(barsEl, probs);
    renderLogitMultiples(logitsEl, logits);
  }

  async function loadSampleAssets() {
    if (!model) return;
    const r = await fetch(model.assets.samplesIndex);
    if (!r.ok) return;
    sampleIdx = (await r.json()) as SpriteIndex;
    sampleSprite = await loadImage(model.assets.samplesSprite);
    sampleVectors = null;
    convSpec = await loadConvFeatureSpec(model.assets.featuremapsIndex);
    const e = await fetch(model.assets.embeddings3d);
    if (e.ok) umap = (await e.json()) as Umap3D;
    mountPlayUmap();
    void buildLogitCloud();
  }

  async function sampleFromDataset(tile?: number) {
    if (!model) return;
    if (!sampleIdx || !sampleSprite) {
      await loadSampleAssets();
    }
    if (!sampleIdx || !sampleSprite) return;

    const picked = tile ?? Math.floor(Math.random() * sampleIdx.count);
    sampleTile = Math.max(0, Math.min(picked, sampleIdx.count - 1));
    sampleLabel = sampleIdx.labels[sampleTile] ?? null;

    const ctx = drawCanvas.getContext('2d')!;
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);

    const d = Math.floor(drawCanvas.width * 0.70);
    const dx = Math.floor((drawCanvas.width - d) / 2);
    const dy = Math.floor((drawCanvas.height - d) / 2);
    // Packaged MNIST sprites are native polarity: white digit on black background.
    // The drawing canvas is black ink on white paper, so invert samples at render time.
    drawTileInverted(ctx, sampleSprite, sampleIdx, sampleTile, dx, dy, d);

    resetPrediction();
    renderPreviews(downsampleTo28x28(drawCanvas));
    umapHandle?.setSelected(sampleTile);
    umapHandle?.setHighlightLabel(sampleLabel);
    logitHandle?.setSelected(sampleTile);
    logitHandle?.setHighlightLabel(sampleLabel);
    placeSampleInEmbedding(sampleTile);
    placeSampleInLogitSpace(sampleTile);
  }

  async function nearestExamples(ink: Float32Array, digit: number, limit = 8): Promise<Neighbor[]> {
    if (!sampleIdx) return [];
    const vectors = await ensureSampleVectors();
    if (!vectors) return [];
    const out: Neighbor[] = [];
    for (let tile = 0; tile < sampleIdx.count; tile++) {
      const label = sampleIdx.labels[tile];
      if (label !== digit || tile === sampleTile) continue;
      out.push({
        tile,
        label,
        sampleId: sampleIdx.sampleIds[tile] ?? tile,
        reason: `label ${label}`,
        distance: imageDistance(ink, vectors[tile])
      });
    }
    return out.sort((a, b) => a.distance - b.distance).slice(0, limit);
  }

  function renderPreviews(img28: ImageData) {
    const ctx = previewCanvas.getContext('2d')!;
    const tmp = document.createElement('canvas');
    tmp.width = 28;
    tmp.height = 28;
    tmp.getContext('2d')!.putImageData(img28, 0, 0);
    previewCanvas.width = 112;
    previewCanvas.height = 112;
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    ctx.drawImage(tmp, 0, 0, previewCanvas.width, previewCanvas.height);

    const ink = new Float32Array(28 * 28);
    const normalized = new Float32Array(28 * 28);
    const delta = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
      const j = i * 4;
      const gray = (0.299 * img28.data[j] + 0.587 * img28.data[j + 1] + 0.114 * img28.data[j + 2]) / 255;
      ink[i] = 1 - gray;
      normalized[i] = model ? (ink[i] - model.norm.mean) / model.norm.std : ink[i];
      delta[i] = model ? ink[i] - model.norm.mean : ink[i];
    }
    renderHeatmapSet(preprocessEl, [
      { id: 'pre-ink', label: 'input ink', width: 28, height: 28, values: ink },
      { id: 'pre-norm', label: 'normalized tensor', width: 28, height: 28, values: normalized },
      { id: 'pre-delta', label: 'delta from mean', width: 28, height: 28, values: delta }
    ]);
    renderConvTrace(convTraceEl, conv1TraceFromInk(ink, convSpec));
    featureMaps = [
      ...featureMapsFromInk(ink).slice(0, 2),
      ...learnedConvFeatureMaps(ink, convSpec)
    ];
    renderHeatmapSet(heatmapsEl, featureMaps);
    return ink;
  }

  async function ensureSampleVectors() {
    if (sampleVectors) return sampleVectors;
    if (!sampleIdx || !sampleSprite) return null;
    const tmp = document.createElement('canvas');
    tmp.width = 28;
    tmp.height = 28;
    const ctx = tmp.getContext('2d', { willReadFrequently: true });
    if (!ctx) return null;
    sampleVectors = [];
    for (let tile = 0; tile < sampleIdx.count; tile++) {
      ctx.clearRect(0, 0, 28, 28);
      const sx = (tile % sampleIdx.cols) * sampleIdx.tileSize;
      const sy = Math.floor(tile / sampleIdx.cols) * sampleIdx.tileSize;
      ctx.drawImage(sampleSprite, sx, sy, sampleIdx.tileSize, sampleIdx.tileSize, 0, 0, 28, 28);
      const img = ctx.getImageData(0, 0, 28, 28);
      const v = new Float32Array(28 * 28);
      for (let i = 0; i < v.length; i++) {
        const j = i * 4;
        v[i] = (0.299 * img.data[j] + 0.587 * img.data[j + 1] + 0.114 * img.data[j + 2]) / 255;
      }
      sampleVectors.push(v);
    }
    return sampleVectors;
  }

  function imageDistance(a: Float32Array, b: Float32Array) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const d = a[i] - b[i];
      sum += d * d;
    }
    return Math.sqrt(sum / a.length);
  }

  function nearestEmbeddingSamples(tile: number, limit = 8) {
    if (!umap) return [];
    const query = umap.coords3d[tile];
    if (!query) return [];
    const rows: { index: number; distance: number; weight: number }[] = [];
    for (let index = 0; index < umap.coords3d.length; index++) {
      if (index === tile) continue;
      const coord = umap.coords3d[index];
      const dx = coord[0] - query[0];
      const dy = coord[1] - query[1];
      const dz = coord[2] - query[2];
      rows.push({ index, distance: Math.sqrt(dx * dx + dy * dy + dz * dz), weight: 0 });
    }
    const picked = rows.sort((a, b) => a.distance - b.distance).slice(0, limit);
    const scale = Math.max(1e-6, picked[Math.min(picked.length - 1, Math.floor(picked.length / 2))]?.distance ?? 1);
    let total = 0;
    for (const row of picked) {
      row.weight = Math.exp(-row.distance / scale);
      total += row.weight;
    }
    for (const row of picked) row.weight = total ? row.weight / total : 0;
    return picked;
  }

  function placeSampleInEmbedding(tile: number | null) {
    if (tile == null || !umap || !sampleIdx) return;
    const coord = umap.coords3d[tile];
    if (!coord) return;
    const nn = nearestEmbeddingSamples(tile, 8);
    const nnText = nn.slice(0, 3).map((n) => {
      const label = umap?.label[n.index] ?? '?';
      const sampleId = umap?.sampleIds[n.index] ?? n.index;
      return `${label}#${sampleId} ${Math.round(n.weight * 100)}%`;
    }).join(', ');
    queryDetail = `Selected sample #${sampleIdx.sampleIds[tile] ?? tile} linked to nearest embedding neighbors. NN weights: ${nnText}.`;
    umapHandle?.setHighlightLabel(focusedDigit ?? sampleIdx.labels[tile] ?? null);
    umapHandle?.setQuery({
      coord,
      label: sampleIdx.labels[tile] ?? undefined,
      title: `sample #${sampleIdx.sampleIds[tile] ?? tile}`,
      detail: queryDetail,
      neighborWeights: nn
    });
  }

  async function placeQueryInEmbedding(ink: Float32Array, probs: Float32Array, bestLabel: number) {
    if (!umap) return;
    const ranked = Array.from(probs, (p, digit) => ({ digit, p }))
      .sort((a, b) => b.p - a.p)
      .filter((d, index) => index < 4 && d.p >= 0.05);
    if (ranked.length === 0) return;

    let sx = 0, sy = 0, sz = 0, sw = 0;
    const edgeTiles: number[] = [];
    const classSummaries: string[] = [];

    for (const candidate of ranked) {
      const nearest = await nearestExamples(ink, candidate.digit, 6);
      if (nearest.length === 0) continue;
      let cx = 0, cy = 0, cz = 0, cw = 0;
      for (const n of nearest) {
        const coord = umap.coords3d[n.tile];
        if (!coord) continue;
        const localWeight = 1 / Math.max(0.002, n.distance);
        cx += coord[0] * localWeight;
        cy += coord[1] * localWeight;
        cz += coord[2] * localWeight;
        cw += localWeight;
      }
      if (!cw) continue;
      const classWeight = candidate.p;
      sx += (cx / cw) * classWeight;
      sy += (cy / cw) * classWeight;
      sz += (cz / cw) * classWeight;
      sw += classWeight;
      edgeTiles.push(...nearest.slice(0, 3).map((n) => n.tile));
      classSummaries.push(`${candidate.digit} ${(candidate.p * 100).toFixed(0)}%`);
    }
    if (!sw) return;
    const coord: [number, number, number] = [sx / sw, sy / sw, sz / sw];
    const mixed = ranked.length > 1 && ranked[1].p >= 0.12;
    queryDetail = mixed
      ? `Probability-weighted embedding fit across ${classSummaries.join(', ')}.`
      : `Estimated embedding fit near packaged ${bestLabel}s.`;
    umapHandle?.setHighlightLabel(mixed ? null : bestLabel);
    umapHandle?.setQuery({
      coord,
      label: bestLabel,
      title: mixed ? 'probability-weighted query' : `query -> ${bestLabel}`,
      detail: queryDetail,
      neighborIndices: Array.from(new Set(edgeTiles)).slice(0, 12),
      mixed
    });
  }

  function placeQueryInLogitSpace(logits: Float32Array, probs: Float32Array, bestLabel: number) {
    latestLogits = new Float32Array(logits);
    latestProbs = new Float32Array(probs);
    latestBestLabel = bestLabel;

    if (!logitSpace) {
      logitQueryDetail = 'Building logit decision space; this prediction will be placed when it is ready.';
      return;
    }

    const top = Array.from(probs, (p, digit) => ({ digit, p }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 3)
      .map((d) => `${d.digit} ${(d.p * 100).toFixed(0)}%`)
      .join(', ');
    const coord = logitSpace.project(logits);
    const nn = logitSpace.nearest(logits, 8);
    const nnText = nn.slice(0, 3).map((n) => `#${logitSpace?.data.sampleIds[n.index] ?? n.index} ${Math.round(n.weight * 100)}%`).join(', ');
    logitQueryDetail = `Actual raw-logit vector projected into PCA decision space. Top evidence: ${top}. NN logit weights: ${nnText}.`;
    logitHandle?.setHighlightLabel(focusedDigit);
    logitHandle?.setQuery({
      coord,
      label: bestLabel,
      title: `logits -> ${bestLabel}`,
      detail: logitQueryDetail,
      neighborWeights: nn,
      mixed: true
    });
  }

  function placeSampleInLogitSpace(tile: number | null) {
    if (tile == null || !logitSpace || !sampleIdx) return;
    const offset = tile * 10;
    const logits = logitSpace.logits.subarray(offset, offset + 10);
    const coord = logitSpace.data.coords3d[tile];
    if (!coord) return;
    const nn = logitSpace.nearest(logits, 8, tile);
    const nnText = nn.slice(0, 3).map((n) => {
      const label = logitSpace?.data.label[n.index] ?? '?';
      const sampleId = logitSpace?.data.sampleIds[n.index] ?? n.index;
      return `${label}#${sampleId} ${Math.round(n.weight * 100)}%`;
    }).join(', ');
    logitQueryDetail = `Selected sample #${sampleIdx.sampleIds[tile] ?? tile} linked to nearest logit-space neighbors. NN weights: ${nnText}.`;
    logitHandle?.setHighlightLabel(focusedDigit ?? sampleIdx.labels[tile] ?? null);
    logitHandle?.setQuery({
      coord,
      label: sampleIdx.labels[tile] ?? undefined,
      title: `sample #${sampleIdx.sampleIds[tile] ?? tile}`,
      detail: logitQueryDetail,
      neighborWeights: nn
    });
  }

  function pointerPos(ev: PointerEvent) {
    const r = drawCanvas.getBoundingClientRect();
    const x = (ev.clientX - r.left) * (drawCanvas.width / r.width);
    const y = (ev.clientY - r.top) * (drawCanvas.height / r.height);
    return { x, y };
  }

  function onDown(ev: PointerEvent) {
    isDown = true;
    const p = pointerPos(ev);
    lastX = p.x;
    lastY = p.y;
    drawCanvas.setPointerCapture?.(ev.pointerId);
  }

  function onMove(ev: PointerEvent) {
    if (!isDown) return;
    const ctx = drawCanvas.getContext('2d')!;
    const p = pointerPos(ev);

    ctx.strokeStyle = 'black';
    ctx.lineWidth = 22;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();

    lastX = p.x;
    lastY = p.y;
  }

  function onUp() {
    isDown = false;
  }

  async function changeModel() {
    logitBuildSeq += 1;
    client?.dispose();
    client = null;
    sampleIdx = null;
    sampleSprite = null;
    sampleVectors = null;
    convSpec = null;
    umap = null;
    umapHandle?.dispose();
    umapHandle = null;
    logitSpace = null;
    logitHover = null;
    logitHandle?.dispose();
    logitHandle = null;
    logitQueryDetail = 'Logit decision space will appear after the model is ready.';
    latestLogits = null;
    latestProbs = null;
    latestBestLabel = null;
    focusedDigit = null;
    await ensureModelLoaded();
    await loadSampleAssets();
    clear();
  }

  function mountPlayUmap() {
    if (!umap || !umapEl) return;
    umapHandle?.dispose();
    umapHandle = mountUmapD3({
      el: umapEl,
      data: umap,
      highlightLabel: sampleLabel,
      selectedIndex: sampleTile,
      onPick: (sampleIndex) => sampleFromDataset(sampleIndex),
      onHover: (hover) => handleLinkedHover('embedding', hover)
    });
  }

  async function buildLogitCloud() {
    const seq = ++logitBuildSeq;
    logitSpace = null;
    logitHover = null;
    logitHandle?.dispose();
    logitHandle = null;

    if (!model || !client || status !== 'ready') {
      logitQueryDetail = 'Logit decision space waits for the model runtime.';
      return;
    }
    if (!sampleIdx || !sampleSprite) {
      logitQueryDetail = 'Logit decision space waits for packaged samples.';
      return;
    }

    logitQueryDetail = 'Building logit decision space from packaged samples...';
    const vectors = await ensureSampleVectors();
    if (seq !== logitBuildSeq || !vectors || !client || !model || !sampleIdx) return;

    const count = Math.min(sampleIdx.count, vectors.length);
    const batchSize = 160;
    const logits = new Float32Array(count * 10);
    const probs = new Float32Array(count * 10);

    for (let start = 0; start < count; start += batchSize) {
      if (seq !== logitBuildSeq || !client || !model) return;
      const end = Math.min(count, start + batchSize);
      const batch = new Float32Array((end - start) * 28 * 28);
      for (let tile = start; tile < end; tile++) {
        const source = vectors[tile];
        const offset = (tile - start) * 28 * 28;
        for (let i = 0; i < source.length; i++) {
          batch[offset + i] = (source[i] - model.norm.mean) / model.norm.std;
        }
      }
      const out = await client.predictBatch(batch, end - start);
      logits.set(out.logits, start * 10);
      probs.set(out.probs, start * 10);
      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    if (seq !== logitBuildSeq || !sampleIdx) return;
    logitSpace = buildLogitPcaSpace(logits, probs, sampleIdx.labels, sampleIdx.sampleIds, count);
    mountLogitCloud();
    logitQueryDetail = 'Predict to place the actual 10-logit vector in decision space. Toolbar supports zoom, reset, rotate, pan, and box zoom.';
    if (latestLogits && latestProbs && latestBestLabel != null) {
      placeQueryInLogitSpace(latestLogits, latestProbs, latestBestLabel);
    } else if (sampleTile != null) {
      placeSampleInLogitSpace(sampleTile);
    }
  }

  function mountLogitCloud() {
    if (!logitSpace || !logitUmapEl) return;
    logitHandle?.dispose();
    logitHandle = mountUmapD3({
      el: logitUmapEl,
      data: logitSpace.data,
      selectedIndex: sampleTile,
      highlightLabel: null,
      onPick: (sampleIndex) => sampleFromDataset(sampleIndex),
      onHover: (hover) => handleLinkedHover('logit', hover)
    });
    restorePlotFocus();
  }

  function fmtCoord(coord: [number, number, number]) {
    return coord.map((value) => value.toFixed(3)).join(', ');
  }

  function handleLinkedHover(source: 'embedding' | 'logit', hover: UmapHover) {
    if (source === 'embedding') {
      umapHover = hover;
    } else {
      logitHover = hover;
    }
    if (!hover) {
      restorePlotFocus();
      return;
    }
    umapHandle?.setSelected(hover.sampleIndex);
    logitHandle?.setSelected(hover.sampleIndex);
    umapHandle?.setHighlightLabel(hover.label);
    logitHandle?.setHighlightLabel(hover.label);
  }

  function toggleDigitFocus(digit: number) {
    focusedDigit = focusedDigit === digit ? null : digit;
    restorePlotFocus();
  }

  function restorePlotFocus() {
    const label = focusedDigit ?? sampleLabel;
    umapHandle?.setSelected(sampleTile);
    logitHandle?.setSelected(sampleTile);
    umapHandle?.setHighlightLabel(label);
    logitHandle?.setHighlightLabel(label);
  }

  onMount(async () => {
    const registry = await loadModelRegistry();
    models = registry.models;
    model = models.find((m) => m.id === registry.defaultModelId) ?? models[0] ?? null;
    renderCnnArchitecture(architectureEl, models.find((m) => m.id === 'cnn_strong') ?? model);
    clear();

    await ensureModelLoaded();
    await loadSampleAssets();
    renderProbBars(barsEl, new Float32Array(10));
    renderLogitMultiples(logitsEl, new Float32Array(10));
  });
</script>

<div class="container">
  <div class="topbar">
    <div class="title">MNIST Playground</div>
    <nav class="nav">
      <a href={appPath('/')}>Home</a>
      <a href={appPath('/explore')}>Explore</a>
      <a class="active" href={appPath('/play')}>Play</a>
    </nav>
  </div>

  <div class="grid">
    <div class="panel">
      <div class="h">Controls</div>
      <div class="row">
        <select bind:value={model} onchange={changeModel}>
          {#each models as m}
            <option value={m as any}>{m.name}</option>
          {/each}
        </select>
      </div>
      <hr />
      <div class="kv">
        <div>Status</div><b>{status}</b>
        <div>Sample label</div><b>{sampleLabel ?? '-'}</b>
        <div>Prediction</div><b style="font-size: 18px">{pred}</b>
        <div>Confidence</div><b>{(conf * 100).toFixed(1)}%</b>
      </div>
      <hr />
      <div class="h">Convolution trace</div>
      <div bind:this={convTraceEl}></div>
      <hr />
      <div class="h">Preprocessing</div>
      <canvas bind:this={previewCanvas} class="preview-canvas"></canvas>
      <div class="mini-label">28x28 normalized input</div>
      <div class="preprocess-heatmaps heatmaps" bind:this={preprocessEl}></div>
      <hr />
      <div class="h">CNN Strong architecture</div>
      <div bind:this={architectureEl}></div>
    </div>

    <div class="panel center-workbench">
      <div class="h">Draw</div>
      <div class="canvasBox">
        <canvas
          bind:this={drawCanvas}
          width="280"
          height="280"
          class="draw-canvas"
          onpointerdown={onDown}
          onpointermove={onMove}
          onpointerup={onUp}
          onpointercancel={onUp}
          onpointerleave={onUp}
        ></canvas>
      </div>
      <div class="row" style="margin-top: 10px; justify-content:center;">
        <button class="primary" onclick={predictNow}>Predict</button>
        <button onclick={() => sampleFromDataset()}>Sample</button>
        <button onclick={clear}>Clear</button>
      </div>
      <hr />
      <div class="h">Top predictions</div>
      <div class="topk">
        {#if topK.length === 0}
          <span class="muted">No prediction yet.</span>
        {:else}
          {#each topK as item}
            <button
              class:active={focusedDigit === item.digit}
              class="topk-chip"
              onclick={() => toggleDigitFocus(item.digit)}
              onmouseenter={() => {
                umapHandle?.setHighlightLabel(item.digit);
                logitHandle?.setHighlightLabel(item.digit);
              }}
              onmouseleave={restorePlotFocus}
            >{item.digit}: {(item.p * 100).toFixed(1)}%</button>
          {/each}
        {/if}
      </div>
      <hr />
      <div class="h">Embedding cloud</div>
      <div class="umap-frame umap-frame-large">
        <div bind:this={umapEl} class="umap-svg-host"></div>
        {#if umapHover}
          <div class="umap-tip" style={`left:${umapHover.x + 12}px; top:${umapHover.y + 12}px;`}>
            <b>sample #{umapHover.sampleId}</b>
            <span>{umapHover.label} -> {umapHover.pred} · {(umapHover.confidence * 100).toFixed(1)}%</span>
            <span>umap ({fmtCoord(umapHover.coord)})</span>
          </div>
        {/if}
      </div>
      <div class="mini-label">{queryDetail} Toolbar supports zoom, reset, rotate, pan, and box zoom.</div>
      <hr />
      <div class="h">Logit decision space</div>
      <div class="umap-frame umap-frame-large">
        <div bind:this={logitUmapEl} class="umap-svg-host"></div>
        {#if logitHover}
          <div class="umap-tip" style={`left:${logitHover.x + 12}px; top:${logitHover.y + 12}px;`}>
            <b>sample #{logitHover.sampleId}</b>
            <span>{logitHover.label} -> {logitHover.pred} · {(logitHover.confidence * 100).toFixed(1)}%</span>
            <span>logit PCA ({fmtCoord(logitHover.coord)})</span>
          </div>
        {/if}
      </div>
      <div class="mini-label">{logitQueryDetail}</div>
    </div>

    <div class="panel">
      <div class="h">Probabilities</div>
      <div bind:this={barsEl}></div>
      <hr />
      <div class="h">Raw logits</div>
      <div bind:this={logitsEl}></div>
      <hr />
      <div class="h">Feature heatmaps</div>
      <div class="heatmaps" bind:this={heatmapsEl}></div>
    </div>
  </div>
</div>
