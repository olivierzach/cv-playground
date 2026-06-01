<script lang="ts">
  import '../app.css';
  // Svelte 5 runes: layout receives a `children` render function.
  let { children } = $props<{ children: any }>();
</script>

<svelte:head>
  <title>MNIST Playground</title>
</svelte:head>

{@render children()}
