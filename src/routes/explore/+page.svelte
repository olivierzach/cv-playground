<script lang="ts">
  import { onMount } from 'svelte';
  import { appPath } from '$lib/paths';
  import { loadModelRegistry } from '$lib/models/registry';
  import type { ModelEntry } from '$lib/models/types';
  import { learnedConvFeatureMaps, loadConvFeatureSpec, type ConvFeatureSpec } from '$lib/models/convnet';
  import { renderConfusionMatrix } from '$lib/viz/confusionMatrix';
  import { renderLogitMultiples, renderProbBars } from '$lib/viz/probBars';
  import { OrtClient } from '$lib/ort/client';
  import { loadImage, drawTile, type SpriteIndex } from '$lib/viz/spriteSheet';
  import { mountUmapD3, type Umap3D, type UmapHover } from '$lib/viz/umapPlot';
  import { featureMapsFromInk, renderHeatmapSet, type FeatureMapSpec } from '$lib/viz/heatmap';

  type Metrics = {
    summary?: { testAcc?: number; testLoss?: number };
    curves?: { trainLoss?: number[]; valLoss?: number[]; trainAcc?: number[]; valAcc?: number[] };
    confusion?: { labels: number[]; matrix: number[][] };
  };

  let models: ModelEntry[] = $state([]);
  let model: ModelEntry | null = $state(null);
  let status = $state('loading…');

  let metrics: Metrics | null = $state(null);
  let confEl: HTMLDivElement;

  // samples
  let spriteIdx: SpriteIndex | null = $state(null);
  let spriteImg: HTMLImageElement | null = $state(null);
  let gridCanvas: HTMLCanvasElement;
  let sampleCanvas: HTMLCanvasElement;
  let barsEl: HTMLDivElement;
  let logitsEl: HTMLDivElement;
  let selectedTile = $state(0);
  let selectedLabel: number | null = $state(null);
  let selectedPred: number | null = $state(null);
  let confusionFilter: { trueLabel: number; predLabel: number } | null = $state(null);
  let featuremapStatus = $state('not loaded');
  let heatmapsEl: HTMLDivElement;
  let featureMaps: FeatureMapSpec[] = $state([]);

  let client: OrtClient | null = $state(null);
  let convSpec: ConvFeatureSpec | null = null;

  // embeddings
  let umap: Umap3D | null = $state(null);
  let umapEl: HTMLDivElement;
  let umapHover: UmapHover = $state(null);
  let labelFocus: number | null = $state(null);
  let umapHandle: {
    dispose(): void;
    render(): void;
    setSelected(index: number | null): void;
    setHighlightLabel(label: number | null): void;
    setQuery(query: null): void;
  } | null = null;

  const GRID_COLS = 25;
  const GRID_ROWS = 10;
  const GRID_COUNT = GRID_COLS * GRID_ROWS;

  async function ensureModelLoaded() {
    if (!model) return;
    client?.dispose();
    client = new OrtClient();
    try {
      await client.load(model.onnxPath, model.io.input, model.io.output);
    } catch (e: any) {
      status = e?.message ?? String(e);
      throw e;
    }
  }

  async function loadAssets() {
    if (!model) return;

    status = 'loading metrics…';
    const r = await fetch(model.assets.metrics);
    metrics = r.ok ? ((await r.json()) as Metrics) : null;

    status = 'loading samples…';
    const si = await fetch(model.assets.samplesIndex);
    spriteIdx = (await si.json()) as SpriteIndex;
    spriteImg = await loadImage(model.assets.samplesSprite);

    status = 'loading embeddings…';
    const ei = await fetch(model.assets.embeddings3d);
    umap = (await ei.json()) as Umap3D;

    const fi = await fetch(model.assets.featuremapsIndex);
    featuremapStatus = fi.ok ? 'available' : 'missing';
    convSpec = await loadConvFeatureSpec(model.assets.featuremapsIndex);

    if (metrics?.confusion) {
      renderConfusionMatrix(confEl, metrics.confusion, {
        onSelect: (t, p) => {
          confusionFilter = { trueLabel: t, predLabel: p };
          const first = visibleTiles()[0];
          if (first != null) void selectTile(first);
          void renderGrid();
        }
      });
    }

    await ensureModelLoaded();

    // mount 3D embedding
    umapHandle?.dispose();
    umapHandle = mountUmapD3({
      el: umapEl,
      data: umap,
      selectedIndex: selectedTile,
      highlightLabel: labelFocus,
      onPick: (sampleIndex) => {
        // sampleIndex indexes into umap arrays, which correspond to our curated ordering.
        void selectTile(sampleIndex);
      },
      onHover: (hover) => (umapHover = hover)
    });

    status = 'ready';

    confusionFilter = null;
    selectedTile = 0;
    await renderGrid();
    await selectTile(0);
  }

  function visibleTiles() {
    if (!spriteIdx) return [];
    const count = Math.min(spriteIdx.count, GRID_COUNT);
    const all = Array.from({ length: count }, (_, i) => i);
    if (!confusionFilter || !umap) return all;
    return all.filter((i) => umap?.label[i] === confusionFilter?.trueLabel && umap?.pred[i] === confusionFilter?.predLabel);
  }

  async function renderGrid() {
    if (!spriteIdx || !spriteImg) return;
    const ctx = gridCanvas.getContext('2d')!;
    const dSize = 18;
    gridCanvas.width = GRID_COLS * dSize;
    gridCanvas.height = GRID_ROWS * dSize;

    ctx.clearRect(0, 0, gridCanvas.width, gridCanvas.height);

    const tiles = visibleTiles();
    for (let viewIndex = 0; viewIndex < tiles.length; viewIndex++) {
      const i = tiles[viewIndex];
      const x = (viewIndex % GRID_COLS) * dSize;
      const y = Math.floor(viewIndex / GRID_COLS) * dSize;
      drawTile(ctx, spriteImg, spriteIdx, i, x, y, dSize);

      if (i === selectedTile) {
        ctx.strokeStyle = 'rgba(125,211,252,0.95)';
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 1, y + 1, dSize - 2, dSize - 2);
      }
    }
  }

  async function selectTile(tileIndex: number) {
    if (!spriteIdx || !spriteImg || !model || !client) return;
    selectedTile = Math.max(0, Math.min(tileIndex, spriteIdx.count - 1));
    selectedLabel = spriteIdx.labels[selectedTile] ?? null;

    // render selected image big
    const ctx = sampleCanvas.getContext('2d')!;
    sampleCanvas.width = 28;
    sampleCanvas.height = 28;
    ctx.clearRect(0, 0, 28, 28);
    drawTile(ctx, spriteImg, spriteIdx, selectedTile, 0, 0, 28);

    // Packaged sample sprites are already MNIST polarity: black background, white digit.
    // Use grayscale directly as ink intensity; do not apply the draw-canvas inversion here.
    const img = ctx.getImageData(0, 0, 28, 28);
    const x = new Float32Array(1 * 1 * 28 * 28);
    const ink = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
      const j = i * 4;
      const r = img.data[j] / 255;
      const g = img.data[j + 1] / 255;
      const b = img.data[j + 2] / 255;
      const gray = 0.299 * r + 0.587 * g + 0.114 * b;
      const inkValue = gray;
      ink[i] = inkValue;
      x[i] = (inkValue - model.norm.mean) / model.norm.std;
    }

    const { logits, probs } = await client.predict(x, [1, 1, 28, 28]);
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    selectedPred = best;
    renderProbBars(barsEl, probs);
    renderLogitMultiples(logitsEl, logits);
    featureMaps = [
      ...featureMapsFromInk(ink).slice(0, 2),
      ...learnedConvFeatureMaps(ink, convSpec)
    ];
    renderHeatmapSet(heatmapsEl, featureMaps);
    umapHandle?.setSelected(selectedTile);

    await renderGrid();
  }

  function setLabelFocus(label: number | null) {
    labelFocus = label;
    umapHandle?.setHighlightLabel(label);
  }

  function gridClick(ev: MouseEvent) {
    if (!spriteIdx) return;
    const rect = gridCanvas.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    const dSize = rect.width / GRID_COLS;
    const c = Math.floor(x / dSize);
    const r = Math.floor(y / dSize);
    const idx = visibleTiles()[r * GRID_COLS + c];
    if (idx == null) return;
    void selectTile(idx);
  }

  function fmtCoord(coord: [number, number, number]) {
    return coord.map((value) => value.toFixed(3)).join(', ');
  }

  onMount(async () => {
    const registry = await loadModelRegistry();
    models = registry.models;
    model = models.find((m) => m.id === registry.defaultModelId) ?? models[0] ?? null;
    renderProbBars(barsEl, new Float32Array(10));
    renderLogitMultiples(logitsEl, new Float32Array(10));
    await loadAssets();
  });
</script>

<div class="container">
  <div class="topbar">
    <div class="title">MNIST Playground</div>
    <nav class="nav">
      <a href={appPath('/')}>Home</a>
      <a class="active" href={appPath('/explore')}>Explore</a>
      <a href={appPath('/play')}>Play</a>
    </nav>
  </div>

  <div class="grid">
    <div class="panel">
      <div class="h">Model</div>
      <div class="row">
        <select bind:value={model} onchange={loadAssets}>
          {#each models as m}
            <option value={m as any}>{m.name}</option>
          {/each}
        </select>
      </div>
      <hr />
      <div class="kv">
        <div>Status</div><b>{status}</b>
        <div>Test accuracy</div><b>{metrics?.summary?.testAcc != null ? (metrics.summary.testAcc * 100).toFixed(2) + '%' : '—'}</b>
        <div>Test loss</div><b>{metrics?.summary?.testLoss != null ? metrics.summary.testLoss.toFixed(4) : '—'}</b>
        <div>Selected</div><b>{selectedLabel ?? '—'} → {selectedPred ?? '—'}</b>
        <div>Filter</div><b>{confusionFilter ? `${confusionFilter.trueLabel} -> ${confusionFilter.predLabel}` : 'all'}</b>
      </div>
      {#if confusionFilter}
        <div class="row" style="margin-top:10px;">
          <button onclick={() => { confusionFilter = null; void renderGrid(); }}>Clear filter</button>
        </div>
      {/if}
      <hr />
      <div class="h">Curated samples (click)</div>
      <canvas bind:this={gridCanvas} onclick={gridClick as any} style="width:100%; image-rendering: pixelated; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08);"></canvas>
    </div>

    <div class="panel">
      <div class="h">Confusion matrix</div>
      <div bind:this={confEl}></div>
      <hr />
      <div class="h">Selected sample</div>
      <div class="row" style="align-items:flex-start;">
        <canvas bind:this={sampleCanvas} style="width:140px; height:140px; background:white; border-radius: 14px; border: 1px solid rgba(255,255,255,0.10); image-rendering: pixelated;"></canvas>
        <div style="flex:1; min-width: 220px" bind:this={barsEl}></div>
      </div>
      <hr />
      <div class="h">Raw logits</div>
      <div bind:this={logitsEl}></div>
    </div>

    <div class="panel">
      <div class="h">3D embedding (UMAP)</div>
      <div class="digit-filter">
        <button class:active={labelFocus === null} onclick={() => setLabelFocus(null)}>all</button>
        {#each Array.from({ length: 10 }, (_, i) => i) as digit}
          <button class:active={labelFocus === digit} onclick={() => setLabelFocus(digit)}>{digit}</button>
        {/each}
      </div>
      <div class="umap-frame">
        <div bind:this={umapEl} class="umap-svg-host"></div>
        {#if umapHover}
          <div class="umap-tip" style={`left:${umapHover.x + 12}px; top:${umapHover.y + 12}px;`}>
            <b>sample #{umapHover.sampleId}</b>
            <span>{umapHover.label} -> {umapHover.pred} · {(umapHover.confidence * 100).toFixed(1)}%</span>
            <span>umap ({fmtCoord(umapHover.coord)})</span>
          </div>
        {/if}
      </div>
      <div style="margin-top:10px; color: var(--muted); font-size: 12px; line-height: 1.55;">
        Toolbar: zoom, reset, rotate, pan, box zoom. Hover for details. Click a point to select that sample.
      </div>
      <hr />
      <div class="h">Selected feature maps</div>
      <div class="mini-label">Feature map artifacts: {featuremapStatus}</div>
      <div class="heatmaps" bind:this={heatmapsEl}></div>
    </div>
  </div>
</div>
