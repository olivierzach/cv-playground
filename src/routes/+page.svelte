<script lang="ts">
  import { page } from '$app/stores';
  import { appPath } from '$lib/paths';
</script>

<div class="container">
  <div class="topbar">
    <div class="title">Computer Vision Playground</div>
    <nav class="nav">
      <a class:active={$page.url.pathname === appPath('/')} href={appPath('/')}>Home</a>
      <a class:active={$page.url.pathname.startsWith(appPath('/explore'))} href={appPath('/explore')}>Explore</a>
      <a class:active={$page.url.pathname.startsWith(appPath('/play'))} href={appPath('/play')}>Play</a>
    </nav>
  </div>

  <div class="panel">
    <p style="margin:0; color: var(--muted); line-height: 1.55; max-width: 78ch;">
      A static, GitHub-Pages-friendly MNIST playground.
      <br />
      Models run in-browser (ONNX Runtime in a Web Worker). Deep exploration uses precomputed assets (metrics, confusion, feature maps, 3D embeddings).
    </p>
    <div style="margin-top: 10px" class="row">
      <a href={appPath('/explore')}><button class="primary">Explore</button></a>
      <a href={appPath('/play')}><button>Play</button></a>
    </div>
  </div>
</div>
