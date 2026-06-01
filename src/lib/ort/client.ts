type WorkerMsg =
  | { type: 'ready' }
  | { type: 'pred'; id: number; logits: Float32Array; probs: Float32Array }
  | { type: 'error'; message: string };

export class OrtClient {
  private w: Worker;
  private ready = false;
  private seq = 1;
  private pending = new Map<number, (m: WorkerMsg) => void>();

  constructor() {
    // Vite/SvelteKit-friendly worker import
    this.w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    this.w.onmessage = (ev: MessageEvent<WorkerMsg>) => {
      const msg = ev.data;
      if (msg.type === 'ready') {
        this.ready = true;
        return;
      }
      if (msg.type === 'pred') {
        const cb = this.pending.get(msg.id);
        if (cb) {
          this.pending.delete(msg.id);
          cb(msg);
        }
        return;
      }
      if (msg.type === 'error') {
        // Fail all pending
        for (const [, cb] of this.pending) cb(msg);
        this.pending.clear();
      }
    };
  }

  async load(modelUrl: string, inputName: string, outputName: string): Promise<void> {
    this.ready = false;
    this.w.postMessage({ type: 'load', modelUrl, inputName, outputName });
    await this.waitReady(8000);
  }

  private async waitReady(timeoutMs: number): Promise<void> {
    const start = performance.now();
    while (!this.ready) {
      if (performance.now() - start > timeoutMs) throw new Error('Timed out loading model');
      await new Promise((r) => setTimeout(r, 25));
    }
  }

  async predict(x: Float32Array, shape: [number, number, number, number]): Promise<{ logits: Float32Array; probs: Float32Array }> {
    if (!this.ready) throw new Error('Model not ready');
    const id = this.seq++;
    const buf = x.buffer.slice(0);

    const p = new Promise<WorkerMsg>((resolve) => this.pending.set(id, resolve));
    this.w.postMessage({ type: 'predict', id, data: buf, shape }, [buf]);
    const msg = await p;
    if (msg.type === 'error') throw new Error(msg.message);
    if (msg.type !== 'pred') throw new Error('Unexpected worker message');
    return { logits: msg.logits, probs: msg.probs };
  }
}
