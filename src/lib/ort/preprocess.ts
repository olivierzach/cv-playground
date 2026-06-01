// Canvas/ImageData -> Float32Array shaped [1,1,28,28] with MNIST-like normalization.
// This is deliberately simple for MVP; you can upgrade to center-of-mass centering, bounding-box scaling, etc.

export function imageDataToMnistTensor(
  img: ImageData,
  mean: number,
  std: number
): Float32Array {
  // Expect 28x28 RGBA
  const { width, height, data } = img;
  if (width !== 28 || height !== 28) {
    throw new Error(`Expected 28x28 ImageData, got ${width}x${height}`);
  }

  const out = new Float32Array(1 * 1 * 28 * 28);
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const i = (y * 28 + x);
      const j = i * 4;
      // Convert to grayscale in [0,1]. Assume white background; drawing should be dark.
      const r = data[j] / 255;
      const g = data[j + 1] / 255;
      const b = data[j + 2] / 255;
      const gray = (0.299 * r + 0.587 * g + 0.114 * b);

      // Invert so ink=1, bg=0 if using black ink on white background.
      const ink = 1 - gray;

      out[i] = (ink - mean) / std;
    }
  }
  return out;
}

export function downsampleTo28x28(source: HTMLCanvasElement): ImageData {
  const tmp = document.createElement('canvas');
  tmp.width = 28;
  tmp.height = 28;
  const ctx = tmp.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('2D context unavailable');

  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, 28, 28);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, 28, 28);
  return ctx.getImageData(0, 0, 28, 28);
}
