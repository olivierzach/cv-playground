import * as d3 from 'd3';

export type Umap3D = {
  sampleIds: number[];
  coords3d: [number, number, number][];
  label: number[];
  pred: number[];
  correct: boolean[];
  confidence: number[];
};

export type UmapHover = {
  sampleIndex: number;
  sampleId: number;
  label: number;
  pred: number;
  confidence: number;
  correct: boolean;
  coord: [number, number, number];
  x: number;
  y: number;
} | null;

export type UmapQuery = {
  coord: [number, number, number];
  label?: number;
  title: string;
  detail?: string;
  neighborIndices?: number[];
  neighborWeights?: { index: number; weight: number }[];
  mixed?: boolean;
};

const classicPalette = [
  '#60a5fa', '#f87171', '#34d399', '#fbbf24', '#a78bfa',
  '#22d3ee', '#fb7185', '#4ade80', '#f97316', '#e879f9'
];

const scientificPalette = [
  '#5eead4', '#fb7185', '#a3e635', '#fbbf24', '#c084fc',
  '#38bdf8', '#f97316', '#22c55e', '#e879f9', '#94a3b8'
];

const darkLabPalette = [
  '#38bdf8', '#f97316', '#22c55e', '#c084fc', '#facc15',
  '#fb7185', '#14b8a6', '#818cf8', '#a3e635', '#e5e7eb'
];

const palette = darkLabPalette;

type Point = {
  index: number;
  sampleId: number;
  label: number;
  pred: number;
  correct: boolean;
  confidence: number;
  raw: [number, number, number];
  x: number;
  y: number;
  z: number;
};

type Projected = Point & {
  sx: number;
  sy: number;
  depth: number;
  radius: number;
};

type PlotMode = 'rotate' | 'pan' | 'box';

export function mountUmapD3(opts: {
  el: HTMLElement;
  data: Umap3D;
  selectedIndex?: number | null;
  highlightLabel?: number | null;
  query?: UmapQuery | null;
  onPick?: (sampleIndex: number) => void;
  onHover?: (hover: UmapHover) => void;
}) {
  const { el, data } = opts;
  let selectedIndex: number | null = opts.selectedIndex ?? null;
  let highlightLabel: number | null = opts.highlightLabel ?? null;
  let query: UmapQuery | null = opts.query ?? null;
  let yaw = -0.52;
  let pitch = 0.42;
  let zoom = 1;
  let panX = 0;
  let panY = 0;
  let mode: PlotMode = 'rotate';

  const width = Math.max(420, el.clientWidth || 640);
  const height = Math.max(320, el.clientHeight || 440);
  const center = { x: width / 2, y: height / 2 };
  const scale = Math.min(width, height) * 0.31;
  const distance = 5.2;
  const standardizers = makeStandardizers(data.coords3d);
  const overlap = makeOverlapTransform(data, standardizers);
  const points = normalizePoints(data, standardizers, overlap);

  el.innerHTML = '';
  const svg = d3.select(el)
    .append('svg')
    .attr('class', 'umap-svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('width', '100%')
    .attr('height', '100%')
    .attr('role', 'img');

  svg.append('rect')
    .attr('width', width)
    .attr('height', height)
    .attr('rx', 8)
    .attr('fill', '#07101d');

  const grid = svg.append('g').attr('class', 'umap-grid');
  for (let x = 60; x < width; x += 60) {
    grid.append('line').attr('x1', x).attr('x2', x).attr('y1', 24).attr('y2', height - 24);
  }
  for (let y = 60; y < height; y += 60) {
    grid.append('line').attr('x1', 24).attr('x2', width - 24).attr('y1', y).attr('y2', y);
  }

  const axisLayer = svg.append('g').attr('class', 'umap-axes');
  const densityLayer = svg.append('g').attr('class', 'umap-density');
  const edgeLayer = svg.append('g').attr('class', 'umap-edges');
  const pointLayer = svg.append('g').attr('class', 'umap-points');
  const markerLayer = svg.append('g').attr('class', 'umap-markers');
  const brushLayer = svg.append('g').attr('class', 'umap-brush');
  const toolbar = svg.append('g').attr('class', 'umap-toolbar').attr('transform', 'translate(14,14)');
  const legend = svg.append('g').attr('class', 'umap-legend').attr('transform', `translate(${width - 150},18)`);

  legend.append('rect').attr('width', 132).attr('height', 74).attr('rx', 8).attr('fill', 'rgba(3,7,18,0.58)');
  d3.range(10).forEach((digit) => {
    const x = (digit % 5) * 25 + 12;
    const y = Math.floor(digit / 5) * 27 + 17;
    legend.append('circle').attr('cx', x).attr('cy', y).attr('r', 5).attr('fill', palette[digit]);
    legend.append('text')
      .attr('x', x + 8)
      .attr('y', y + 3.5)
      .attr('fill', 'rgba(230,238,247,0.78)')
      .attr('font-size', 10)
      .text(digit);
  });

  const circles = pointLayer.selectAll('circle')
    .data(points, (d: any) => d.index)
    .enter()
    .append('circle')
    .attr('fill', (d) => palette[d.label])
    .attr('stroke', (d) => d.correct ? 'rgba(255,255,255,0.16)' : 'rgba(255,255,255,0.8)')
    .attr('stroke-width', (d) => d.correct ? 0.6 : 1.5)
    .style('cursor', 'pointer')
    .on('mouseenter', (event, d) => emitHover(event, d))
    .on('mousemove', (event, d) => emitHover(event, d))
    .on('mouseleave', () => opts.onHover?.(null))
    .on('click', (_, d) => opts.onPick?.(d.index));

  const selected = markerLayer.append('circle')
    .attr('class', 'umap-selected')
    .attr('fill', 'none')
    .attr('stroke', 'white')
    .attr('stroke-width', 2.1);

  const queryMarker = markerLayer.append('g').attr('class', 'umap-query');
  queryMarker.append('circle').attr('r', 10).attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 2.6);
  queryMarker.append('circle').attr('r', 5).attr('fill', 'white');

  const drag = d3.drag<SVGSVGElement, unknown>()
    .on('drag', (event) => {
      if (mode === 'box') return;
      if (mode === 'pan' || event.sourceEvent?.shiftKey) {
        panX += event.dx;
        panY += event.dy;
      } else {
        yaw += event.dx * 0.008;
        pitch = Math.max(-1.35, Math.min(1.35, pitch + event.dy * 0.008));
      }
      renderState();
    });
  svg.call(drag as any);
  svg.on('wheel', (event: WheelEvent) => {
    event.preventDefault();
    zoomAt(event.offsetX, event.offsetY, Math.exp(-event.deltaY * 0.001));
    renderState();
  });
  svg.on('dblclick', () => {
    resetView();
    renderState();
  });

  const brush = d3.brush<unknown>()
    .extent([[0, 0], [width, height]])
    .on('end', (event) => {
      const selection = event.selection as [[number, number], [number, number]] | null;
      if (!selection) return;
      const [[x0, y0], [x1, y1]] = selection;
      const boxW = Math.abs(x1 - x0);
      const boxH = Math.abs(y1 - y0);
      brushLayer.call(brush.move as any, null);
      if (boxW < 12 || boxH < 12) return;
      const factor = Math.min(width / boxW, height / boxH, 4);
      const cx = (x0 + x1) / 2;
      const cy = (y0 + y1) / 2;
      zoomAt(cx, cy, factor);
      panX += width / 2 - cx;
      panY += height / 2 - cy;
      mode = 'pan';
      updateToolbar();
      updateBrushMode();
      renderState();
    });
  brushLayer.call(brush as any);
  updateBrushMode();
  renderToolbar();

  function emitHover(event: MouseEvent, d: Point) {
    opts.onHover?.({
      sampleIndex: d.index,
      sampleId: d.sampleId,
      label: d.label,
      pred: d.pred,
      confidence: d.confidence,
      correct: d.correct,
      coord: d.raw,
      x: event.offsetX,
      y: event.offsetY
    });
  }

  function focusOpacity(d: Point, depth: number) {
    const base = 0.28 + depth * 0.68;
    const focus = queryFocusSet();
    if (focus) {
      if (d.index === selectedIndex || focus.has(d.index)) return Math.min(1, base + 0.08);
      if (highlightLabel != null && d.label === highlightLabel) return Math.min(base, 0.34);
      return 0.11;
    }
    if (highlightLabel == null || d.label === highlightLabel || d.index === selectedIndex) return base;
    return 0.08;
  }

  function queryFocusSet() {
    if (!query) return null;
    const indices = query.neighborWeights?.length
      ? query.neighborWeights.map((edge) => edge.index)
      : (query.neighborIndices ?? []);
    if (!indices.length && selectedIndex == null) return null;
    const out = new Set(indices);
    if (selectedIndex != null) out.add(selectedIndex);
    return out;
  }

  function projectPoint(d: Point): Projected {
    const rotated = rotate(d.x, d.y, d.z, yaw, pitch);
    const perspective = distance / (distance - rotated.z);
    return {
      ...d,
      sx: center.x + panX + rotated.x * scale * zoom * perspective,
      sy: center.y + panY - rotated.y * scale * zoom * perspective,
      depth: (rotated.z + 2.8) / 5.6,
      radius: Math.max(1.7, Math.min(6.8, 3.2 * zoom * perspective))
    };
  }

  function renderAxes() {
    axisLayer.selectAll('*').remove();
    const axes = [
      { label: 'UMAP-x', color: '#7dd3fc', end: rotate(1.35, 0, 0, yaw, pitch) },
      { label: 'UMAP-y', color: '#fbbf24', end: rotate(0, 1.35, 0, yaw, pitch) },
      { label: 'UMAP-z', color: '#fb7185', end: rotate(0, 0, 1.35, yaw, pitch) }
    ];
    const origin = { x: 48, y: height - 48 };
    axes.forEach((axis) => {
      const ex = origin.x + axis.end.x * 34;
      const ey = origin.y - axis.end.y * 34;
      axisLayer.append('line')
        .attr('x1', origin.x)
        .attr('y1', origin.y)
        .attr('x2', ex)
        .attr('y2', ey)
        .attr('stroke', axis.color)
        .attr('stroke-width', 2);
      axisLayer.append('text')
        .attr('x', ex + 4)
        .attr('y', ey + 3)
        .attr('fill', axis.color)
        .attr('font-size', 10)
        .text(axis.label);
    });
  }

  function renderState() {
    const projected = points.map(projectPoint).sort((a, b) => a.depth - b.depth);
    const projectedByIndex = new Map(projected.map((d) => [d.index, d]));
    renderDensity(projected);

    circles
      .data(projected, (d: any) => d.index)
      .sort((a, b) => a.depth - b.depth)
      .attr('cx', (d) => d.sx)
      .attr('cy', (d) => d.sy)
      .attr('r', (d) => d.index === selectedIndex ? d.radius + 2 : d.radius)
      .attr('opacity', (d) => focusOpacity(d, d.depth));

    edgeLayer.selectAll('*').remove();
    let queryProjected: { sx: number; sy: number; label?: number } | null = null;
    if (query) {
      const q = projectCoord(query.coord, query.mixed ? null : (query.label ?? null));
      queryProjected = { sx: q.sx, sy: q.sy, label: query.label };
      const weighted = query.neighborWeights?.length
        ? query.neighborWeights
        : (query.neighborIndices ?? []).map((index) => ({ index, weight: 0.24 }));
      const edgeData = weighted
        .map((edge) => {
          const point = projectedByIndex.get(edge.index);
          return point ? { ...point, edgeWeight: edge.weight } : null;
        })
        .filter(Boolean) as (Projected & { edgeWeight: number })[];
      edgeLayer.selectAll('line')
        .data(edgeData)
        .enter()
        .append('line')
        .attr('x1', q.sx)
        .attr('y1', q.sy)
        .attr('x2', (d) => d.sx)
        .attr('y2', (d) => d.sy)
        .attr('stroke', (d) => `rgba(255,255,255,${0.22 + Math.min(0.68, d.edgeWeight * 1.6)})`)
        .attr('stroke-width', (d) => 0.8 + Math.min(4.4, d.edgeWeight * 8))
        .attr('stroke-dasharray', (d) => d.edgeWeight > 0.18 ? null : '4 4');

      edgeLayer.selectAll('text')
        .data(edgeData.filter((d) => d.edgeWeight >= 0.08).slice(0, 6))
        .enter()
        .append('text')
        .attr('x', (d) => q.sx * 0.58 + d.sx * 0.42)
        .attr('y', (d) => q.sy * 0.58 + d.sy * 0.42)
        .attr('fill', 'rgba(255,255,255,0.76)')
        .attr('font-size', 10)
        .attr('paint-order', 'stroke')
        .attr('stroke', 'rgba(3,7,18,0.92)')
        .attr('stroke-width', 3)
        .text((d) => `${Math.round(d.edgeWeight * 100)}%`);
    }

    if (selectedIndex != null && projectedByIndex.has(selectedIndex)) {
      const d = projectedByIndex.get(selectedIndex)!;
      selected
        .attr('display', null)
        .attr('cx', d.sx)
        .attr('cy', d.sy)
        .attr('r', d.radius + 5);
    } else {
      selected.attr('display', 'none');
    }

    if (queryProjected) {
      queryMarker
        .attr('display', null)
        .attr('transform', `translate(${queryProjected.sx},${queryProjected.sy})`);
      queryMarker.select('circle:nth-child(2)').attr('fill', palette[(queryProjected.label ?? 0) % palette.length]);
    } else {
      queryMarker.attr('display', 'none');
    }

    renderAxes();
  }

  function renderDensity(projected: Projected[]) {
    const densityPoints = highlightLabel == null
      ? projected
      : projected.filter((d) => d.label === highlightLabel);
    densityLayer.selectAll('*').remove();
    if (densityPoints.length < 8) return;

    const contours = d3.contourDensity<Projected>()
      .x((d) => d.sx)
      .y((d) => d.sy)
      .size([width, height])
      .cellSize(4)
      .bandwidth(18)
      .thresholds(9)(densityPoints);

    const color = highlightLabel == null
      ? 'rgba(125,211,252,'
      : hexToRgbaPrefix(palette[highlightLabel]);

    densityLayer.selectAll('path')
      .data(contours)
      .enter()
      .append('path')
      .attr('d', d3.geoPath() as any)
      .attr('fill', (_, i) => `${color}${0.025 + i * 0.012})`)
      .attr('stroke', (_, i) => `${color}${0.16 + i * 0.035})`)
      .attr('stroke-width', 0.85)
      .attr('stroke-linejoin', 'round');
  }

  function projectCoord(coord: [number, number, number], label: number | null) {
    const p = overlap.apply(normalizeCoord(coord, standardizers), label, undefined);
    const rotated = rotate(p.x, p.y, p.z, yaw, pitch);
    const perspective = distance / (distance - rotated.z);
    return {
      sx: center.x + panX + rotated.x * scale * zoom * perspective,
      sy: center.y + panY - rotated.y * scale * zoom * perspective,
      depth: (rotated.z + 2.8) / 5.6
    };
  }

  function zoomAt(x: number, y: number, factor: number) {
    const next = Math.max(0.45, Math.min(10, zoom * factor));
    const actual = next / zoom;
    panX = x - center.x - (x - center.x - panX) * actual;
    panY = y - center.y - (y - center.y - panY) * actual;
    zoom = next;
  }

  function resetView() {
    yaw = -0.52;
    pitch = 0.42;
    zoom = 1;
    panX = 0;
    panY = 0;
    mode = 'rotate';
    updateToolbar();
    updateBrushMode();
  }

  function renderToolbar() {
    const buttons: { id: string; label: string; title: string; action: () => void }[] = [
      { id: 'zoom-in', label: '+', title: 'Zoom in', action: () => { zoomAt(width / 2, height / 2, 1.32); renderState(); } },
      { id: 'zoom-out', label: '-', title: 'Zoom out', action: () => { zoomAt(width / 2, height / 2, 1 / 1.32); renderState(); } },
      { id: 'reset', label: '1:1', title: 'Reset view', action: () => { resetView(); renderState(); } },
      { id: 'rotate', label: 'rot', title: 'Rotate mode', action: () => { mode = 'rotate'; updateToolbar(); updateBrushMode(); } },
      { id: 'pan', label: 'pan', title: 'Pan mode', action: () => { mode = 'pan'; updateToolbar(); updateBrushMode(); } },
      { id: 'box', label: 'box', title: 'Box zoom mode', action: () => { mode = 'box'; updateToolbar(); updateBrushMode(); } }
    ];

    const items = toolbar.selectAll<SVGGElement, (typeof buttons)[number]>('g.umap-tool')
      .data(buttons)
      .enter()
      .append('g')
      .attr('class', 'umap-tool')
      .attr('transform', (_, i) => `translate(${i * 37},0)`)
      .style('cursor', 'pointer')
      .on('click', (_, d) => d.action());

    items.append('title').text((d) => d.title);
    items.append('rect').attr('width', 33).attr('height', 24).attr('rx', 6);
    items.append('text')
      .attr('x', 16.5)
      .attr('y', 15.5)
      .attr('text-anchor', 'middle')
      .attr('font-size', 10)
      .attr('font-weight', 700)
      .text((d) => d.label);

    updateToolbar();
  }

  function updateToolbar() {
    toolbar.selectAll<SVGGElement, { id: string }>('g.umap-tool')
      .classed('active', (d) => d.id === mode);
  }

  function updateBrushMode() {
    brushLayer
      .attr('display', mode === 'box' ? null : 'none')
      .style('pointer-events', mode === 'box' ? 'all' : 'none');
    brushLayer.selectAll('.overlay').attr('cursor', 'crosshair');
  }

  renderState();

  return {
    setSelected(index: number | null) {
      selectedIndex = index;
      renderState();
    },
    setHighlightLabel(label: number | null) {
      highlightLabel = label;
      renderState();
    },
    setQuery(next: UmapQuery | null) {
      query = next;
      renderState();
    },
    dispose() {
      el.innerHTML = '';
    },
    render() {
      renderState();
    }
  };
}

function normalizePoints(
  data: Umap3D,
  standardizers: ReturnType<typeof makeStandardizers>,
  overlap: ReturnType<typeof makeOverlapTransform>
): Point[] {
  return data.coords3d.map((coord, index) => {
    const normalized = normalizeCoord(coord, standardizers);
    const softened = overlap.apply(normalized, data.label[index] ?? null, index);
    return {
      index,
      sampleId: data.sampleIds[index] ?? index,
      label: data.label[index],
      pred: data.pred[index],
      correct: data.correct[index],
      confidence: data.confidence[index],
      raw: coord,
      ...softened
    };
  });
}

function makeOverlapTransform(data: Umap3D, standardizers: ReturnType<typeof makeStandardizers>) {
  const normalized = data.coords3d.map((coord) => normalizeCoord(coord, standardizers));
  const global = centroid(normalized);
  const byLabel = new Map<number, { x: number; y: number; z: number }>();
  for (let label = 0; label < 10; label++) {
    const points = normalized.filter((_, index) => data.label[index] === label);
    byLabel.set(label, points.length ? centroid(points) : global);
  }

  return {
    apply(coord: { x: number; y: number; z: number }, label: number | null, index: number | undefined) {
      const c = label == null ? global : (byLabel.get(label) ?? global);
      const jitter = index == null ? { x: 0, y: 0, z: 0 } : deterministicJitter(index, label ?? 0);
      const pull = 0.42;
      return {
        x: coord.x - (c.x - global.x) * pull + jitter.x,
        y: coord.y - (c.y - global.y) * pull + jitter.y,
        z: coord.z - (c.z - global.z) * pull + jitter.z
      };
    }
  };
}

function centroid(points: { x: number; y: number; z: number }[]) {
  const sum = points.reduce((acc, p) => ({
    x: acc.x + p.x,
    y: acc.y + p.y,
    z: acc.z + p.z
  }), { x: 0, y: 0, z: 0 });
  return {
    x: sum.x / points.length,
    y: sum.y / points.length,
    z: sum.z / points.length
  };
}

function deterministicJitter(index: number, label: number) {
  const a = seededUnit(index * 73856093 + label * 19349663);
  const b = seededUnit(index * 83492791 + label * 2971215073);
  const c = seededUnit(index * 2654435761 + label * 97531);
  const amount = 0.026;
  return {
    x: (a - 0.5) * amount,
    y: (b - 0.5) * amount,
    z: (c - 0.5) * amount
  };
}

function seededUnit(seed: number) {
  let x = seed >>> 0;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  return (x >>> 0) / 0xffffffff;
}

function makeStandardizers(coords: [number, number, number][]) {
  const xs = coords.map((d) => d[0]);
  const ys = coords.map((d) => d[1]);
  const zs = coords.map((d) => d[2]);
  return {
    x: standardizer(xs),
    y: standardizer(ys),
    z: standardizer(zs)
  };
}

function normalizeCoord(coord: [number, number, number], standardizers: ReturnType<typeof makeStandardizers>) {
  return {
    x: standardizers.x(coord[0]),
    y: standardizers.y(coord[1]),
    z: standardizers.z(coord[2])
  };
}

function standardizer(values: number[]) {
  const mean = d3.mean(values) ?? 0;
  const deviation = d3.deviation(values) || 1;
  return (value: number) => (value - mean) / deviation;
}

function rotate(x: number, y: number, z: number, yaw: number, pitch: number) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = x * cy + z * sy;
  const z1 = -x * sy + z * cy;
  const y2 = y * cp - z1 * sp;
  const z2 = y * sp + z1 * cp;
  return { x: x1, y: y2, z: z2 };
}

function hexToRgbaPrefix(hex: string) {
  const value = hex.replace('#', '');
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return `rgba(${r},${g},${b},`;
}
