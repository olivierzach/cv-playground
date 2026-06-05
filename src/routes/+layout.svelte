<script lang="ts">
  import '../app.css';
  let { children } = $props();
</script>

<svelte:head>
  <title>Computer Vision Playground</title>
</svelte:head>

{@render children()}
