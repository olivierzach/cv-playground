import type { ModelEntry, ModelManifest } from './types';

export async function loadManifest(path = '/models/manifest.json'): Promise<ModelManifest> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`Failed to load manifest: ${r.status} ${r.statusText}`);
  return (await r.json()) as ModelManifest;
}

export async function loadModels(): Promise<ModelEntry[]> {
  const m = await loadManifest();
  return m.models;
}
