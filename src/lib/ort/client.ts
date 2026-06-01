type WorkerMsg =
  | { type: 'ready' }
  | { type: 'pred'; id: number; logits: Float32Array; probs: Float32Array }
  | { type: 'error'; message: string };

export class OrtClient {
  private w: Worker;
  private ready = false;
  private readyResolve: (() => void) | null = null;
  private readyReject: ((err: Error) => void) | null = null;
  private seq = 1;
  private pending = new Map<number, (m: WorkerMsg) => void>();

  constructor() {
    // Vite/SvelteKit-friendly worker import
    this.w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });

    this.w.onmessage = (ev: MessageEvent<WorkerMsg>) => {
      const msg = ev.data;

      if (msg.type === 'ready') {
        this.ready = true;
        this.readyResolve?.();
        this.readyResolve = null;
        this.readyReject = null;
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
        // Reject model load if we're currently waiting for it
        this.readyReject?.(new Error(msg.message));
        this.readyResolve = null;
        this.readyReject = null;

        // Fail all pending predictions
        for (const [, cb] of this.pending) cb(msg);
        this.pending.clear();
      }
    };

    this.w.onerror = (e) => {
      this.readyReject?.(new Error(e.message || 'Worker error'));
      this.readyResolve = null;
      this.readyReject = null;
    };
  }

  async load(modelUrl: string, inputName: string, outputName: string): Promise<void> {
    this.ready = false;

    const p = new Promise<void>((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });

    this.w.postMessage({ type: 'load', modelUrl, inputName, outputName });

    // Timeout wrapper so we don't hang forever.
    await Promise.race([
      p,
      new Promise<void>((_, reject) => setTimeout(() => reject(new Error('Timed out loading model')), 15000))
    ]);
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
