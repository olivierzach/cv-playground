/// <reference lib="webworker" />
import * as ort from 'onnxruntime-web';

type LoadMsg = {
  type: 'load';
  modelUrl: string;
  inputName: string;
  outputName: string;
};

type PredictMsg = {
  type: 'predict';
  id: number;
  // Float32Array serialized as ArrayBuffer
  data: ArrayBuffer;
  shape: [number, number, number, number];
};

type Msg = LoadMsg | PredictMsg;

type ReadyResp = { type: 'ready' };

type PredictResp = {
  type: 'pred';
  id: number;
  logits: Float32Array;
  probs: Float32Array;
};

type ErrorResp = { type: 'error'; message: string };

let session: ort.InferenceSession | null = null;
let inputName = 'input';
let outputName = 'logits';

function softmax(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
  let sum = 0;
  const exps = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    const v = Math.exp(logits[i] - max);
    exps[i] = v;
    sum += v;
  }
  for (let i = 0; i < exps.length; i++) exps[i] /= sum;
  return exps;
}

self.onmessage = async (ev: MessageEvent<Msg>) => {
  const msg = ev.data;
  try {
    if (msg.type === 'load') {
      inputName = msg.inputName;
      outputName = msg.outputName;

      // Prefer WebGPU when available; WASM fallback otherwise.
      // Note: On GitHub Pages, cross-origin isolation may not be enabled; avoid threads.
      ort.env.wasm.numThreads = 1;

      session = await ort.InferenceSession.create(msg.modelUrl, {
        executionProviders: ['webgpu', 'wasm']
      });
      const resp: ReadyResp = { type: 'ready' };
      self.postMessage(resp);
      return;
    }

    if (msg.type === 'predict') {
      if (!session) throw new Error('Model not loaded');
      const x = new Float32Array(msg.data);
      const tensor = new ort.Tensor('float32', x, msg.shape);

      const feeds: Record<string, ort.Tensor> = { [inputName]: tensor };
      const results = await session.run(feeds);
      const out = results[outputName];
      if (!out) throw new Error(`Missing output '${outputName}'. Got keys: ${Object.keys(results).join(', ')}`);

      // MNIST logits expected shape [1,10]
      const logits = out.data as Float32Array;
      const probs = softmax(logits);

      const resp: PredictResp = { type: 'pred', id: msg.id, logits, probs };
      self.postMessage(resp, { transfer: [resp.logits.buffer, resp.probs.buffer] });
      return;
    }
  } catch (e: any) {
    const resp: ErrorResp = { type: 'error', message: e?.message ?? String(e) };
    self.postMessage(resp);
  }
};
