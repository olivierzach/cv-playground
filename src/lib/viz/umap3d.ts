import * as THREE from 'three';

export type Umap3D = {
  sampleIds: number[];
  coords3d: [number, number, number][];
  label: number[];
  pred: number[];
  correct: boolean[];
  confidence: number[];
};

export function mountUmap3D(opts: {
  canvas: HTMLCanvasElement;
  data: Umap3D;
  onPick?: (sampleIndex: number) => void;
}) {
  const { canvas, data } = opts;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
  camera.position.set(0, 0, 3);

  // Fit / normalize coords to unit-ish cube
  const pts = data.coords3d;
  const min = new THREE.Vector3(+Infinity, +Infinity, +Infinity);
  const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
  for (const [x, y, z] of pts) {
    min.min(new THREE.Vector3(x, y, z));
    max.max(new THREE.Vector3(x, y, z));
  }
  const center = min.clone().add(max).multiplyScalar(0.5);
  const span = max.clone().sub(min);
  const scale = 1 / Math.max(span.x, span.y, span.z, 1e-6);

  const N = pts.length;
  const positions = new Float32Array(N * 3);
  const colors = new Float32Array(N * 3);

  const palette = [
    0x60a5fa, 0xf87171, 0x34d399, 0xfbbf24, 0xa78bfa,
    0x22d3ee, 0xfb7185, 0x4ade80, 0xf97316, 0xe879f9
  ];

  for (let i = 0; i < N; i++) {
    const [x, y, z] = pts[i];
    const v = new THREE.Vector3(x, y, z).sub(center).multiplyScalar(scale);
    positions[i * 3 + 0] = v.x;
    positions[i * 3 + 1] = v.y;
    positions[i * 3 + 2] = v.z;

    const c = new THREE.Color(palette[data.label[i] % palette.length]);
    colors[i * 3 + 0] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({ size: 0.02, vertexColors: true, transparent: true, opacity: 0.95 });
  const points = new THREE.Points(geom, mat);
  scene.add(points);

  // Simple orbit-ish controls
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  let rotX = 0;
  let rotY = 0;
  let zoom = 1;

  function resize() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function render() {
    resize();
    points.rotation.x = rotY;
    points.rotation.y = rotX;
    camera.position.set(0, 0, 3 / zoom);
    renderer.render(scene, camera);
  }

  const raycaster = new THREE.Raycaster();
  raycaster.params.Points!.threshold = 0.03;
  const mouse = new THREE.Vector2();

  function pick(ev: PointerEvent) {
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -(((ev.clientY - rect.top) / rect.height) * 2 - 1);
    raycaster.setFromCamera(mouse, camera);
    const hit = raycaster.intersectObject(points);
    if (hit.length > 0 && hit[0].index != null) {
      opts.onPick?.(hit[0].index);
    }
  }

  function onDown(ev: PointerEvent) {
    dragging = true;
    lastX = ev.clientX;
    lastY = ev.clientY;
    (ev.target as Element).setPointerCapture?.(ev.pointerId);
  }

  function onMove(ev: PointerEvent) {
    if (!dragging) return;
    const dx = ev.clientX - lastX;
    const dy = ev.clientY - lastY;
    lastX = ev.clientX;
    lastY = ev.clientY;
    rotX += dx * 0.005;
    rotY += dy * 0.005;
    render();
  }

  function onUp() {
    dragging = false;
  }

  function onWheel(ev: WheelEvent) {
    ev.preventDefault();
    zoom *= Math.exp(-ev.deltaY * 0.001);
    zoom = Math.max(0.5, Math.min(5, zoom));
    render();
  }

  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointerleave', onUp);
  canvas.addEventListener('dblclick', (ev) => pick(ev as any));
  canvas.addEventListener('wheel', onWheel, { passive: false });

  render();

  return {
    dispose() {
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerup', onUp);
      canvas.removeEventListener('pointerleave', onUp);
      canvas.removeEventListener('wheel', onWheel);
      geom.dispose();
      mat.dispose();
      renderer.dispose();
    },
    render
  };
}
