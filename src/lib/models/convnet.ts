import type { FeatureMapSpec } from '$lib/viz/heatmap';

export type ConvLayerSpec = {
  weights: number[][][][];
  bias: number[];
  padding: number;
  stride: number;
};

export type ConvFeatureSpec = {
  schemaVersion: number;
  kind: string;
  layers?: {
    conv1?: ConvLayerSpec;
    conv2?: ConvLayerSpec;
  };
};

export type ConvTrace = {
  layer: 'conv1';
  channel: number;
  outX: number;
  outY: number;
  kernelSize: number;
  padding: number;
  stride: number;
  bias: number;
  preActivation: number;
  activation: number;
  patch: number[];
  kernel: number[];
  products: number[];
  output: FeatureMapSpec;
};

export async function loadConvFeatureSpec(url: string): Promise<ConvFeatureSpec | null> {
  const response = await fetch(url);
  if (!response.ok) return null;
  const spec = (await response.json()) as ConvFeatureSpec;
  if (spec.kind !== 'convnet_features' || !spec.layers?.conv1) return null;
  return spec;
}

export function learnedConvFeatureMaps(ink: Float32Array, spec: ConvFeatureSpec | null): FeatureMapSpec[] {
  if (!spec?.layers?.conv1) return [];

  const input = { channels: 1, width: 28, height: 28, data: [ink] };
  const conv1 = relu(conv2d(input, spec.layers.conv1));
  const pool1 = maxPool2d(conv1, 2);
  const maps: FeatureMapSpec[] = [
    summarizeChannels('conv1 mean activation', conv1, 'conv1-mean', 'mean'),
    summarizeChannels('conv1 max activation', conv1, 'conv1-max', 'max'),
    ...topChannelMaps('conv1', conv1, 6)
  ];

  if (spec.layers.conv2) {
    const conv2 = relu(conv2d(pool1, spec.layers.conv2));
    const pool2 = maxPool2d(conv2, 2);
    maps.push(
      summarizeChannels('conv2 mean activation', conv2, 'conv2-mean', 'mean'),
      summarizeChannels('conv2 max activation', conv2, 'conv2-max', 'max'),
      ...topChannelMaps('conv2', conv2, 10),
      summarizeChannels('pool2 compressed response', pool2, 'pool2-mean', 'mean')
    );
  }

  return maps;
}

export function conv1TraceFromInk(ink: Float32Array, spec: ConvFeatureSpec | null): ConvTrace | null {
  const layer = spec?.layers?.conv1;
  if (!layer) return null;

  const input = { channels: 1, width: 28, height: 28, data: [ink] };
  const conv1Linear = conv2d(input, layer);
  const conv1 = relu(conv1Linear);
  let best = { channel: 0, x: 0, y: 0, value: -Infinity };

  for (let channel = 0; channel < conv1.channels; channel++) {
    const map = conv1.data[channel];
    for (let i = 0; i < map.length; i++) {
      if (map[i] > best.value) {
        best = {
          channel,
          x: i % conv1.width,
          y: Math.floor(i / conv1.width),
          value: map[i]
        };
      }
    }
  }

  const kernel = layer.weights[best.channel][0];
  const kernelSize = kernel.length;
  const padding = layer.padding ?? 0;
  const stride = layer.stride ?? 1;
  const patch: number[] = [];
  const flatKernel: number[] = [];
  const products: number[] = [];

  for (let ky = 0; ky < kernelSize; ky++) {
    for (let kx = 0; kx < kernelSize; kx++) {
      const sx = best.x * stride + kx - padding;
      const sy = best.y * stride + ky - padding;
      const inputValue = sx < 0 || sx >= input.width || sy < 0 || sy >= input.height
        ? 0
        : ink[sy * input.width + sx];
      const weight = kernel[ky][kx];
      patch.push(inputValue);
      flatKernel.push(weight);
      products.push(inputValue * weight);
    }
  }

  const bias = layer.bias[best.channel] ?? 0;
  const productSum = products.reduce((sum, value) => sum + value, 0);
  const preActivation = bias + productSum;

  return {
    layer: 'conv1',
    channel: best.channel,
    outX: best.x,
    outY: best.y,
    kernelSize,
    padding,
    stride,
    bias,
    preActivation,
    activation: Math.max(0, preActivation),
    patch,
    kernel: flatKernel,
    products,
    output: {
      id: `conv1-trace-ch-${best.channel}`,
      label: `conv1 channel #${best.channel}`,
      width: conv1.width,
      height: conv1.height,
      values: conv1.data[best.channel]
    }
  };
}

type Tensor3 = { channels: number; width: number; height: number; data: Float32Array[] };

function conv2d(input: Tensor3, layer: ConvLayerSpec): Tensor3 {
  const outChannels = layer.weights.length;
  const kernel = layer.weights[0][0].length;
  const pad = layer.padding ?? 0;
  const stride = layer.stride ?? 1;
  const outW = Math.floor((input.width + 2 * pad - kernel) / stride) + 1;
  const outH = Math.floor((input.height + 2 * pad - kernel) / stride) + 1;
  const out: Float32Array[] = [];

  for (let oc = 0; oc < outChannels; oc++) {
    const target = new Float32Array(outW * outH);
    for (let oy = 0; oy < outH; oy++) {
      for (let ox = 0; ox < outW; ox++) {
        let sum = layer.bias[oc] ?? 0;
        for (let ic = 0; ic < input.channels; ic++) {
          const weights = layer.weights[oc][ic];
          for (let ky = 0; ky < kernel; ky++) {
            for (let kx = 0; kx < kernel; kx++) {
              const sx = ox * stride + kx - pad;
              const sy = oy * stride + ky - pad;
              if (sx < 0 || sx >= input.width || sy < 0 || sy >= input.height) continue;
              sum += input.data[ic][sy * input.width + sx] * weights[ky][kx];
            }
          }
        }
        target[oy * outW + ox] = sum;
      }
    }
    out.push(target);
  }

  return { channels: outChannels, width: outW, height: outH, data: out };
}

function relu(input: Tensor3): Tensor3 {
  return {
    ...input,
    data: input.data.map((channel) => {
      const out = new Float32Array(channel.length);
      for (let i = 0; i < channel.length; i++) out[i] = Math.max(0, channel[i]);
      return out;
    })
  };
}

function maxPool2d(input: Tensor3, size: number): Tensor3 {
  const outW = Math.floor(input.width / size);
  const outH = Math.floor(input.height / size);
  return {
    channels: input.channels,
    width: outW,
    height: outH,
    data: input.data.map((channel) => {
      const out = new Float32Array(outW * outH);
      for (let oy = 0; oy < outH; oy++) {
        for (let ox = 0; ox < outW; ox++) {
          let mx = -Infinity;
          for (let ky = 0; ky < size; ky++) {
            for (let kx = 0; kx < size; kx++) {
              mx = Math.max(mx, channel[(oy * size + ky) * input.width + (ox * size + kx)]);
            }
          }
          out[oy * outW + ox] = mx;
        }
      }
      return out;
    })
  };
}

function summarizeChannels(label: string, tensor: Tensor3, id: string, mode: 'mean' | 'max'): FeatureMapSpec {
  const out = new Float32Array(tensor.width * tensor.height);
  for (let i = 0; i < out.length; i++) {
    if (mode === 'max') {
      let mx = -Infinity;
      for (const channel of tensor.data) mx = Math.max(mx, channel[i]);
      out[i] = mx;
    } else {
      let sum = 0;
      for (const channel of tensor.data) sum += channel[i];
      out[i] = sum / tensor.channels;
    }
  }
  return { id, label, width: tensor.width, height: tensor.height, values: out };
}

function topChannelMaps(layer: string, tensor: Tensor3, count: number): FeatureMapSpec[] {
  return channelMeans(tensor)
    .sort((a, b) => b.mean - a.mean)
    .slice(0, Math.min(count, tensor.channels))
    .map(({ index }) => channelMap(`${layer} channel`, tensor, index, `${layer}-ch-${index}`));
}

function channelMeans(tensor: Tensor3) {
  return tensor.data.map((channel, index) => {
    let sum = 0;
    for (const value of channel) sum += value;
    return { index, mean: sum / channel.length };
  });
}

function channelMap(label: string, tensor: Tensor3, channel: number, id: string): FeatureMapSpec {
  return {
    id,
    label: `${label} #${channel}`,
    width: tensor.width,
    height: tensor.height,
    values: tensor.data[channel]
  };
}
