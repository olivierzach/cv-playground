<script lang="ts">
  import { onMount } from 'svelte';
  import { loadModels } from '$lib/models/registry';
  import type { ModelEntry } from '$lib/models/types';
  import { renderConfusionMatrix } from '$lib/viz/confusionMatrix';
  import { renderProbBars } from '$lib/viz/probBars';
  import { OrtClient } from '$lib/ort/client';
  import { loadImage, drawTile, type SpriteIndex } from '$lib/viz/spriteSheet';

  type Metrics = {
    summary?: { testAcc?: number; testLoss?: number };
    curves?: { trainLoss?: number[]; valLoss?: number[]; trainAcc?: number[]; valAcc?: number[] };
    confusion?: { labels: number[]; matrix: number[][] };
  };

  let models: ModelEntry[] = [];
  let model: ModelEntry | null = null;
  let status = 'loading…';

  let metrics: Metrics | null = null;
  let confEl: HTMLDivElement;

  // samples
  let spriteIdx: SpriteIndex | null = null;
  let spriteImg: HTMLImageElement | null = null;
  let gridCanvas: HTMLCanvasElement;
  let sampleCanvas: HTMLCanvasElement;
  let barsEl: HTMLDivElement;
  let selectedTile = 0;
  let selectedLabel: number | null = null;
  let selectedPred: number | null = null;

  let client: OrtClient | null = null;

  const GRID_COLS = 25;
  const GRID_ROWS = 10;
  const GRID_COUNT = GRID_COLS * GRID_ROWS;

  async function ensureModelLoaded() {
    if (!model) return;
    client = new OrtClient();
    await client.load(model.onnxPath, model.io.input, model.io.output);
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

    if (metrics?.confusion) {
      renderConfusionMatrix(confEl, metrics.confusion, {
        onSelect: (t, p) => console.log('confusion select', { t, p })
      });
    }

    await ensureModelLoaded();
    status = 'ready';

    selectedTile = 0;
    await renderGrid();
    await selectTile(0);
  }

  async function renderGrid() {
    if (!spriteIdx || !spriteImg) return;
    const ctx = gridCanvas.getContext('2d')!;
    const dSize = 18;
    gridCanvas.width = GRID_COLS * dSize;
    gridCanvas.height = GRID_ROWS * dSize;

    ctx.clearRect(0, 0, gridCanvas.width, gridCanvas.height);

    const count = Math.min(spriteIdx.count, GRID_COUNT);
    for (let i = 0; i < count; i++) {
      const x = (i % GRID_COLS) * dSize;
      const y = Math.floor(i / GRID_COLS) * dSize;
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

    // get pixels, convert to normalized tensor like play-mode (ink=1-bg)
    const img = ctx.getImageData(0, 0, 28, 28);
    const x = new Float32Array(1 * 1 * 28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
      const j = i * 4;
      const r = img.data[j] / 255;
      const g = img.data[j + 1] / 255;
      const b = img.data[j + 2] / 255;
      const gray = 0.299 * r + 0.587 * g + 0.114 * b;
      const ink = 1 - gray;
      x[i] = (ink - model.norm.mean) / model.norm.std;
    }

    const { probs } = await client.predict(x, [1, 1, 28, 28]);
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    selectedPred = best;
    renderProbBars(barsEl, probs);

    await renderGrid();
  }

  function gridClick(ev: MouseEvent) {
    if (!spriteIdx) return;
    const rect = gridCanvas.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    const dSize = rect.width / GRID_COLS;
    const c = Math.floor(x / dSize);
    const r = Math.floor(y / dSize);
    const idx = r * GRID_COLS + c;
    void selectTile(idx);
  }

  onMount(async () => {
    models = await loadModels();
    model = models.find((m) => m.id === 'cnn_small') ?? models[0] ?? null;
    renderProbBars(barsEl, new Float32Array(10));
    await loadAssets();
  });
</script>

<div class="container">
  <div class="topbar">
    <div class="title">MNIST Playground</div>
    <nav class="nav">
      <a href="/">Home</a>
      <a class="active" href="/explore">Explore</a>
      <a href="/play">Play</a>
    </nav>
  </div>

  <div class="grid">
    <div class="panel">
      <div class="h">Model</div>
      <div class="row">
        <select bind:value={model} on:change={loadAssets}>
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
      </div>
      <hr />
      <div class="h">Curated samples (click)</div>
      <canvas bind:this={gridCanvas} on:click={gridClick} style="width:100%; image-rendering: pixelated; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08);" />
    </div>

    <div class="panel">
      <div class="h">Confusion matrix</div>
      <div bind:this={confEl} />
      <hr />
      <div class="h">Selected sample</div>
      <div class="row" style="align-items:flex-start;">
        <canvas bind:this={sampleCanvas} style="width:140px; height:140px; background:white; border-radius: 14px; border: 1px solid rgba(255,255,255,0.10); image-rendering: pixelated;" />
        <div style="flex:1; min-width: 220px" bind:this={barsEl} />
      </div>
    </div>

    <div class="panel">
      <div class="h">Embeddings + Layer lens (next)</div>
      <div style="color: var(--muted); font-size: 13px; line-height: 1.6;">
        Assets are already generated:
        <ul>
          <li>3D UMAP: <code>{model?.assets.embeddings3d}</code></li>
          <li>Featuremaps index (currently empty): <code>{model?.assets.featuremapsIndex}</code></li>
        </ul>
        Next wiring:
        <ul>
          <li>three.js point cloud from <code>umap3d.json</code> (click point → select sample)</li>
          <li>feature map atlases for conv/pool per sample</li>
        </ul>
      </div>
    </div>
  </div>
</div>
