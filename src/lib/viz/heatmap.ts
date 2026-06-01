// Simple heatmap overlay renderer for a 28x28 float/int array.
// Produces an ImageData RGBA suitable for ctx.putImageData or drawImage.

export function heatmap28ToImageData(values: Float32Array | Uint8Array, alpha = 0.55): ImageData {
  const w = 28, h = 28;
  const out = new Uint8ClampedArray(w * h * 4);

  // normalize
  let mn = +Infinity, mx = -Infinity;
  for (let i = 0; i < w * h; i++) {
    const v = values[i];
    mn = Math.min(mn, v);
    mx = Math.max(mx, v);
  }
  const denom = (mx - mn) || 1;

  for (let i = 0; i < w * h; i++) {
    const t = (values[i] - mn) / denom; // 0..1
    // magma-ish ramp (cheap approximation)
    const r = Math.round(255 * Math.min(1, Math.max(0, 1.8 * t)));
    const g = Math.round(255 * Math.min(1, Math.max(0, 1.2 * (t - 0.2))));
    const b = Math.round(255 * Math.min(1, Math.max(0, 1.0 * (t - 0.45))));

    const j = i * 4;
    out[j + 0] = r;
    out[j + 1] = g;
    out[j + 2] = b;
    out[j + 3] = Math.round(255 * alpha * t);
  }

  return new ImageData(out, w, h);
}
