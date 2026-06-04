import type { ModelEntry, ModelManifest } from './types';
import { assetPath } from '$lib/paths';

export async function loadManifest(path = '/models/manifest.json'): Promise<ModelManifest> {
  const r = await fetch(assetPath(path));
  if (!r.ok) throw new Error(`Failed to load manifest: ${r.status} ${r.statusText}`);
  return (await r.json()) as ModelManifest;
}

export async function loadModels(): Promise<ModelEntry[]> {
  const m = await loadManifest();
  return m.models.map(resolveModelPaths);
}

export async function loadModelRegistry(): Promise<ModelManifest> {
  const m = await loadManifest();
  return {
    ...m,
    models: m.models.map(resolveModelPaths)
  };
}

function resolveModelPaths(model: ModelEntry): ModelEntry {
  return {
    ...model,
    onnxPath: assetPath(model.onnxPath),
    assets: {
      ...model.assets,
      metrics: assetPath(model.assets.metrics),
      samplesIndex: assetPath(model.assets.samplesIndex),
      samplesSprite: assetPath(model.assets.samplesSprite),
      embeddings3d: assetPath(model.assets.embeddings3d),
      featuremapsIndex: assetPath(model.assets.featuremapsIndex),
      modelCard: model.assets.modelCard ? assetPath(model.assets.modelCard) : undefined
    }
  };
}
