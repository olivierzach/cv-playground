import type { Umap3D } from './umapPlot';

export type LogitPcaSpace = {
  data: Umap3D;
  mean: number[];
  components: number[][];
  logits: Float32Array;
  probs: Float32Array;
  count: number;
  project(logits: ArrayLike<number>): [number, number, number];
  nearest(logits: ArrayLike<number>, limit?: number, excludeIndex?: number | null): { index: number; distance: number; weight: number }[];
};

const DIGITS = 10;

export function buildLogitPcaSpace(
  logits: Float32Array,
  probs: Float32Array,
  labels: number[],
  sampleIds: number[],
  count: number
): LogitPcaSpace {
  if (logits.length < count * DIGITS || probs.length < count * DIGITS) {
    throw new Error('Logit/probability buffers are shorter than the requested sample count');
  }

  const mean = makeMean(logits, count);
  const components = principalComponents(logits, count, mean, 3);
  const coords3d: [number, number, number][] = [];
  const pred: number[] = [];
  const confidence: number[] = [];
  const correct: boolean[] = [];

  for (let row = 0; row < count; row++) {
    coords3d.push(projectRow(logits, row * DIGITS, mean, components));
    const best = argmax(probs, row * DIGITS, DIGITS);
    pred.push(best.index);
    confidence.push(best.value);
    correct.push(best.index === labels[row]);
  }

  return {
    data: {
      sampleIds: sampleIds.slice(0, count),
      coords3d,
      label: labels.slice(0, count),
      pred,
      confidence,
      correct
    },
    mean,
    components,
    logits: new Float32Array(logits.slice(0, count * DIGITS)),
    probs: new Float32Array(probs.slice(0, count * DIGITS)),
    count,
    project(values: ArrayLike<number>) {
      return projectVector(values, mean, components);
    },
    nearest(values: ArrayLike<number>, limit = 8, excludeIndex: number | null = null) {
      return nearestLogits(logits, values, count, limit, excludeIndex);
    }
  };
}

function makeMean(logits: Float32Array, count: number) {
  const mean = Array(DIGITS).fill(0);
  for (let row = 0; row < count; row++) {
    const offset = row * DIGITS;
    for (let dim = 0; dim < DIGITS; dim++) mean[dim] += logits[offset + dim];
  }
  for (let dim = 0; dim < DIGITS; dim++) mean[dim] /= Math.max(1, count);
  return mean;
}

function principalComponents(logits: Float32Array, count: number, mean: number[], componentCount: number) {
  const cov = Array.from({ length: DIGITS }, () => Array(DIGITS).fill(0));
  for (let row = 0; row < count; row++) {
    const offset = row * DIGITS;
    for (let i = 0; i < DIGITS; i++) {
      const ai = logits[offset + i] - mean[i];
      for (let j = i; j < DIGITS; j++) {
        cov[i][j] += ai * (logits[offset + j] - mean[j]);
      }
    }
  }

  const denom = Math.max(1, count - 1);
  for (let i = 0; i < DIGITS; i++) {
    for (let j = i; j < DIGITS; j++) {
      cov[i][j] /= denom;
      cov[j][i] = cov[i][j];
    }
  }

  const components: number[][] = [];
  for (let k = 0; k < componentCount; k++) {
    let v = seededVector(k);
    for (let iter = 0; iter < 90; iter++) {
      v = normalize(matVec(cov, v));
    }
    const eigenvalue = dot(v, matVec(cov, v));
    if (!Number.isFinite(eigenvalue) || Math.abs(eigenvalue) < 1e-9) {
      v = seededVector(k + 11);
    }
    components.push(v);
    for (let i = 0; i < DIGITS; i++) {
      for (let j = 0; j < DIGITS; j++) {
        cov[i][j] -= eigenvalue * v[i] * v[j];
      }
    }
  }
  return components;
}

function projectRow(logits: Float32Array, offset: number, mean: number[], components: number[][]): [number, number, number] {
  const out = components.map((component) => {
    let sum = 0;
    for (let dim = 0; dim < DIGITS; dim++) sum += (logits[offset + dim] - mean[dim]) * component[dim];
    return sum;
  });
  return [out[0] ?? 0, out[1] ?? 0, out[2] ?? 0];
}

function projectVector(values: ArrayLike<number>, mean: number[], components: number[][]): [number, number, number] {
  const out = components.map((component) => {
    let sum = 0;
    for (let dim = 0; dim < DIGITS; dim++) sum += (values[dim] - mean[dim]) * component[dim];
    return sum;
  });
  return [out[0] ?? 0, out[1] ?? 0, out[2] ?? 0];
}

function matVec(matrix: number[][], vector: number[]) {
  return matrix.map((row) => dot(row, vector));
}

function dot(a: number[], b: number[]) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function normalize(vector: number[]): number[] {
  const norm = Math.hypot(...vector);
  if (!Number.isFinite(norm) || norm < 1e-12) return seededVector(0);
  return vector.map((v) => v / norm);
}

function seededVector(seed: number): number[] {
  const vector = Array.from({ length: DIGITS }, (_, index) => {
    const x = Math.sin((seed + 1) * 12.9898 + (index + 1) * 78.233) * 43758.5453;
    return (x - Math.floor(x)) * 2 - 1;
  });
  return normalize(vector);
}

function argmax(values: Float32Array, offset: number, length: number) {
  let index = 0;
  let value = values[offset];
  for (let i = 1; i < length; i++) {
    if (values[offset + i] > value) {
      index = i;
      value = values[offset + i];
    }
  }
  return { index, value };
}

function nearestLogits(
  sampleLogits: Float32Array,
  query: ArrayLike<number>,
  count: number,
  limit: number,
  excludeIndex: number | null
) {
  const rows: { index: number; distance: number; weight: number }[] = [];
  for (let row = 0; row < count; row++) {
    if (row === excludeIndex) continue;
    let sum = 0;
    const offset = row * DIGITS;
    for (let dim = 0; dim < DIGITS; dim++) {
      const d = sampleLogits[offset + dim] - query[dim];
      sum += d * d;
    }
    rows.push({ index: row, distance: Math.sqrt(sum / DIGITS), weight: 0 });
  }

  const picked = rows.sort((a, b) => a.distance - b.distance).slice(0, limit);
  const scale = Math.max(1e-6, picked[Math.min(picked.length - 1, Math.floor(picked.length / 2))]?.distance ?? 1);
  let total = 0;
  for (const row of picked) {
    row.weight = Math.exp(-row.distance / scale);
    total += row.weight;
  }
  for (const row of picked) row.weight = total ? row.weight / total : 0;
  return picked;
}
