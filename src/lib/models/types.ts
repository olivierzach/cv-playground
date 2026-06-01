export type ModelLayer = {
  id: string;
  displayName: string;
  channels: number;
  tile: [number, number];
};

export type ModelEntry = {
  id: string;
  name: string;
  onnxPath: string;
  io: { input: string; output: string };
  norm: { mean: number; std: number };
  assets: {
    metrics: string;
    samplesIndex: string;
    samplesSprite: string;
    embeddings3d: string;
    featuremapsIndex: string;
  };
  layers: ModelLayer[];
};

export type ModelManifest = { models: ModelEntry[] };
