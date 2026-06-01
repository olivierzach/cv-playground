<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { loadModels } from '$lib/models/registry';
  import type { ModelEntry } from '$lib/models/types';
  import { OrtClient } from '$lib/ort/client';
  import { downsampleTo28x28, imageDataToMnistTensor } from '$lib/ort/preprocess';
  import { renderProbBars } from '$lib/viz/probBars';
  import { loadImage, drawTile, type SpriteIndex } from '$lib/viz/spriteSheet';

  let models: ModelEntry[] = $state([]);
  let model: ModelEntry | null = $state(null);
  let client: OrtClient | null = $state(null);
  let status = $state('loading models…');

  let drawCanvas: HTMLCanvasElement;
  let isDown = $state(false);
  let lastX = $state(0);
  let lastY = $state(0);

  let pred = $state('-');
  let conf = $state(0);

  // Optional: show a known MNIST sample
  let sampleLabel: number | null = $state(null);
  let sampleTile: number | null = $state(null);
  let sampleIdx: SpriteIndex | null = $state(null);
  let sampleSprite: HTMLImageElement | null = $state(null);

  let barsEl: HTMLDivElement;

  function setActiveNav() {
    // noop – layout uses pathname
  }

  function clear() {
    const ctx = drawCanvas.getContext('2d')!;
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    sampleLabel = null;
    sampleTile = null;
    pred = '-';
    conf = 0;
    renderProbBars(barsEl, new Float32Array(10));
  }

  async function ensureModelLoaded() {
    if (!model) return;
    client = new OrtClient();
    status = `loading model: ${model.name}…`;
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
    const x = imageDataToMnistTensor(img28, model.norm.mean, model.norm.std);
    const { probs } = await client.predict(x, [1, 1, 28, 28]);

    let bestI = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[bestI]) bestI = i;
    pred = String(bestI);
    conf = probs[bestI];
    renderProbBars(barsEl, probs);
  }

  async function loadSampleAssets() {
    if (!model) return;
    const r = await fetch(model.assets.samplesIndex);
    if (!r.ok) return;
    sampleIdx = (await r.json()) as SpriteIndex;
    sampleSprite = await loadImage(model.assets.samplesSprite);
  }

  async function sampleFromDataset() {
    if (!model) return;
    if (!sampleIdx || !sampleSprite) {
      await loadSampleAssets();
    }
    if (!sampleIdx || !sampleSprite) return;

    const tile = Math.floor(Math.random() * sampleIdx.count);
    sampleTile = tile;
    sampleLabel = sampleIdx.labels[tile] ?? null;

    const ctx = drawCanvas.getContext('2d')!;
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);

    // Draw the 28x28 tile scaled up into the center.
    const d = Math.floor(drawCanvas.width * 0.70);
    const dx = Math.floor((drawCanvas.width - d) / 2);
    const dy = Math.floor((drawCanvas.height - d) / 2);
    drawTile(ctx, sampleSprite, sampleIdx, tile, dx, dy, d);

    // do not auto-predict; user presses Predict
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
    // No auto-predict; use the Predict button.
  }

  onMount(async () => {
    models = await loadModels();
    model = models[0] ?? null;
    clear();

    await ensureModelLoaded();
    await loadSampleAssets();
    renderProbBars(barsEl, new Float32Array(10));
  });
</script>

<div class="container">
  <div class="topbar">
    <div class="title">MNIST Playground</div>
    <nav class="nav">
      <a href="/">Home</a>
      <a href="/explore">Explore</a>
      <a class="active" href="/play">Play</a>
    </nav>
  </div>

  <div class="grid">
    <div class="panel">
      <div class="h">Controls</div>
      <div class="row">
        <select bind:value={model} onchange={async () => { await ensureModelLoaded(); await loadSampleAssets(); }}>
          {#each models as m}
            <option value={m as any}>{m.name}</option>
          {/each}
        </select>
      </div>
      <div style="margin-top:10px; color: var(--muted); font-size: 13px; line-height: 1.5;">
        Draw a digit, then press <b>Predict</b>.
      </div>
      <hr />
      <div class="kv">
        <div>Status</div><b>{status}</b>
        <div>Sample label</div><b>{sampleLabel ?? '—'}</b>
        <div>Prediction</div><b style="font-size: 18px">{pred}</b>
        <div>Confidence</div><b>{(conf * 100).toFixed(1)}%</b>
      </div>
    </div>

    <div class="panel">
      <div class="h">Draw</div>
      <div class="canvasBox">
        <canvas
          bind:this={drawCanvas}
          width="280"
          height="280"
          style="border-radius: 14px; border: 1px solid rgba(255,255,255,0.10); touch-action:none; background:white"
          onpointerdown={onDown}
          onpointermove={onMove}
          onpointerup={onUp}
          onpointercancel={onUp}
          onpointerleave={onUp}
        ></canvas>
      </div>
      <div class="row" style="margin-top: 10px; justify-content:center;">
        <button class="primary" onclick={predictNow}>Predict</button>
        <button onclick={sampleFromDataset}>Sample</button>
        <button onclick={clear}>Clear</button>
      </div>
      <div style="margin-top:10px; color: var(--muted); font-size: 12px;">
        Tip: draw big and centered; MNIST preprocessing will be improved in V2.
      </div>
    </div>

    <div class="panel">
      <div class="h">Probabilities</div>
      <div bind:this={barsEl}></div>
    </div>
  </div>
</div>
