<script lang="ts">
  import { page } from '$app/stores';
</script>

<div class="container">
  <div class="topbar">
    <div class="title">MNIST Playground</div>
    <nav class="nav">
      <a class:active={$page.url.pathname === '/'} href="/">Home</a>
      <a class:active={$page.url.pathname.startsWith('/explore')} href="/explore">Explore</a>
      <a class:active={$page.url.pathname.startsWith('/play')} href="/play">Play</a>
    </nav>
  </div>

  <div class="panel">
    <p style="margin:0; color: var(--muted); line-height: 1.55; max-width: 78ch;">
      A static, GitHub-Pages-friendly MNIST playground.
      <br />
      Models run in-browser (ONNX Runtime in a Web Worker). Deep exploration uses precomputed assets (metrics, confusion, feature maps, 3D embeddings).
    </p>
    <div style="margin-top: 10px" class="row">
      <a href="/explore"><button class="primary">Explore</button></a>
      <a href="/play"><button>Play</button></a>
    </div>
  </div>
</div>
