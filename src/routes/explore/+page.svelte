<script lang="ts">
  import { onMount } from 'svelte';
  import { loadModels } from '$lib/models/registry';
  import type { ModelEntry } from '$lib/models/types';
  import { renderConfusionMatrix } from '$lib/viz/confusionMatrix';

  type Metrics = {
    summary?: { testAcc?: number; testLoss?: number };
    confusion?: { labels: number[]; matrix: number[][] };
  };

  let models: ModelEntry[] = [];
  let model: ModelEntry | null = null;
  let status = 'loading…';

  let metrics: Metrics | null = null;

  let confEl: HTMLDivElement;

  async function loadMetrics() {
    if (!model) return;
    status = 'loading metrics…';
    const r = await fetch(model.assets.metrics);
    if (!r.ok) {
      status = `missing metrics: ${r.status}`;
      metrics = null;
      return;
    }
    metrics = (await r.json()) as Metrics;
    status = 'ready';

    if (metrics?.confusion) {
      renderConfusionMatrix(confEl, metrics.confusion, {
        onSelect: (t, p) => {
          // MVP: just log; next: update error gallery + sample selection.
          console.log('confusion select', { t, p });
        }
      });
    }
  }

  onMount(async () => {
    models = await loadModels();
    model = models[0] ?? null;
    await loadMetrics();
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
        <select bind:value={model} on:change={loadMetrics}>
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
      </div>
      <div style="margin-top:10px; color: var(--muted); font-size: 12px; line-height: 1.55;">
        MVP: confusion matrix + model selector.
        <br />
        Next: linked error gallery, layer lens (conv/pool), and 3D UMAP embeddings.
      </div>
    </div>

    <div class="panel">
      <div class="h">Confusion matrix</div>
      <div bind:this={confEl} />
    </div>

    <div class="panel">
      <div class="h">Layer lens (stub)</div>
      <div style="color: var(--muted); font-size: 13px; line-height: 1.6;">
        This panel will show per-layer activation summaries and channel grids.
        <ul>
          <li>conv/pool summary maps</li>
          <li>channel tiles (sprite sheets)</li>
          <li>selected-sample filmstrip across layers</li>
        </ul>
      </div>
    </div>
  </div>
</div>
