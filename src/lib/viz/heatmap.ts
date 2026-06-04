import * as d3 from 'd3';

export type FeatureMapSpec = {
  id: string;
  label: string;
  width: number;
  height: number;
  values: Float32Array;
};

export function renderHeatmapSet(el: HTMLElement, maps: FeatureMapSpec[]) {
  el.innerHTML = '';
  const grid = document.createElement('div');
  grid.className = 'heatmap-small-multiples';
  el.appendChild(grid);
  for (const map of maps) {
    const holder = document.createElement('div');
    holder.className = 'heatmap-tile';
    grid.appendChild(holder);
    renderHeatmapTile(holder, map);
  }
}

function renderHeatmapTile(el: HTMLElement, spec: FeatureMapSpec) {
  const width = 132;
  const mapSize = 96;
  const height = 132;
  const pad = { top: 22, left: 18 };
  const cellW = mapSize / spec.width;
  const cellH = mapSize / spec.height;
  const values = Array.from(spec.values, (v) => Number.isFinite(v) ? v : 0);
  const sorted = [...values].sort((a, b) => a - b);
  const min = d3.quantile(sorted, 0.02) ?? d3.min(values) ?? 0;
  const max = d3.quantile(sorted, 0.98) ?? d3.max(values) ?? 1;
  const domain: [number, number] = min === max ? [min, min + 1] : [min, max];
  const color = d3.scaleSequential(d3.interpolateMagma).domain(domain);

  el.innerHTML = '';
  const svg = d3.select(el).append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('width', '100%')
    .attr('height', height)
    .attr('class', 'heatmap-svg heatmap-mini-svg')
    .attr('role', 'img');

  svg.append('rect')
    .attr('width', width)
    .attr('height', height)
    .attr('rx', 8)
    .attr('fill', '#070d18');

  svg.append('text')
    .attr('x', 8)
    .attr('y', 14)
    .attr('fill', 'rgba(230,238,247,0.86)')
    .attr('font-size', 9.5)
    .attr('font-weight', 650)
    .text(shortLabel(spec.label));

  const g = svg.append('g').attr('transform', `translate(${pad.left},${pad.top})`);
  const cells = values.map((value, index) => ({
    value,
    x: index % spec.width,
    y: Math.floor(index / spec.width)
  }));

  g.selectAll('rect.hm-cell')
    .data(cells)
    .enter()
    .append('rect')
    .attr('class', 'hm-cell')
    .attr('x', (d) => d.x * cellW)
    .attr('y', (d) => d.y * cellH)
    .attr('width', Math.max(1, cellW + 0.35))
    .attr('height', Math.max(1, cellH + 0.35))
    .attr('fill', (d) => color(Math.max(domain[0], Math.min(domain[1], d.value))));

  const thresholds = d3.ticks(domain[0], domain[1], 5).slice(1, -1);
  const contours = d3.contours()
    .size([spec.width, spec.height])
    .thresholds(thresholds)(values);
  const projection = d3.geoTransform({
    point(x, y) {
      this.stream.point(x * cellW, y * cellH);
    }
  });
  const path = d3.geoPath(projection);

  g.append('g')
    .selectAll('path')
    .data(contours)
    .enter()
    .append('path')
    .attr('d', path as any)
    .attr('fill', 'none')
    .attr('stroke', 'rgba(255,255,255,0.38)')
    .attr('stroke-width', 0.55)
    .attr('stroke-linejoin', 'round');

  const extentText = `${formatTiny(domain[0])}..${formatTiny(domain[1])}`;
  svg.append('text')
    .attr('x', 8)
    .attr('y', height - 8)
    .attr('fill', 'rgba(138,160,184,0.82)')
    .attr('font-size', 9)
    .text(extentText);

  const hover = svg.append('g').style('display', 'none');
  hover.append('rect').attr('rx', 5).attr('height', 20).attr('fill', 'rgba(3,7,18,0.9)').attr('stroke', 'rgba(255,255,255,0.16)');
  const tip = hover.append('text').attr('x', 6).attr('y', 13).attr('fill', 'white').attr('font-size', 9.5);

  svg.append('rect')
    .attr('x', pad.left)
    .attr('y', pad.top)
    .attr('width', mapSize)
    .attr('height', mapSize)
    .attr('fill', 'transparent')
    .style('cursor', 'crosshair')
    .on('mousemove', (ev) => {
      const [mx, my] = d3.pointer(ev, svg.node());
      const px = Math.max(0, Math.min(spec.width - 1, Math.floor((mx - pad.left) / cellW)));
      const py = Math.max(0, Math.min(spec.height - 1, Math.floor((my - pad.top) / cellH)));
      const value = spec.values[py * spec.width + px] ?? 0;
      const text = `${px},${py} ${value.toFixed(3)}`;
      tip.text(text);
      const boxW = Math.max(62, text.length * 5.8 + 12);
      hover.select('rect').attr('width', boxW);
      hover
        .attr('transform', `translate(${Math.min(width - boxW - 4, mx + 6)},${Math.max(4, my - 24)})`)
        .style('display', null);
    })
    .on('mouseleave', () => hover.style('display', 'none'));
}

export function renderHeatmapPlot(el: HTMLElement, spec: FeatureMapSpec) {
  const width = Math.max(260, el.clientWidth || 320);
  const height = 178;
  const margin = { top: 20, right: 52, bottom: 28, left: 30 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const cellW = plotW / spec.width;
  const cellH = plotH / spec.height;
  const values = Array.from(spec.values);
  const extent = d3.extent(values) as [number, number];
  const min = extent[0] ?? 0;
  const max = extent[1] ?? 1;
  const color = d3.scaleSequential(d3.interpolateTurbo).domain([min, max || 1]);

  el.innerHTML = '';
  const svg = d3.select(el).append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('width', '100%')
    .attr('height', height)
    .attr('class', 'heatmap-svg');

  svg.append('rect')
    .attr('x', 0)
    .attr('y', 0)
    .attr('width', width)
    .attr('height', height)
    .attr('rx', 8)
    .attr('fill', '#0b1220');

  svg.append('text')
    .attr('x', margin.left)
    .attr('y', 15)
    .attr('fill', 'rgba(230,238,247,0.9)')
    .attr('font-size', 12)
    .attr('font-weight', 650)
    .text(spec.label);

  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const cells = values.map((value, index) => ({
    value,
    x: index % spec.width,
    y: Math.floor(index / spec.width)
  }));

  g.selectAll('rect.hm-cell')
    .data(cells)
    .enter()
    .append('rect')
    .attr('class', 'hm-cell')
    .attr('x', (d) => d.x * cellW)
    .attr('y', (d) => d.y * cellH)
    .attr('width', Math.max(1, cellW + 0.4))
    .attr('height', Math.max(1, cellH + 0.4))
    .attr('fill', (d) => color(d.value));

  const contourValues = values.map((v) => (Number.isFinite(v) ? v : 0));
  const thresholds = d3.ticks(min, max, 7).slice(1, -1);
  const contours = d3.contours()
    .size([spec.width, spec.height])
    .thresholds(thresholds)(contourValues);
  const projection = d3.geoTransform({
    point(x, y) {
      this.stream.point(x * cellW, y * cellH);
    }
  });
  const path = d3.geoPath(projection);

  g.append('g')
    .selectAll('path')
    .data(contours)
    .enter()
    .append('path')
    .attr('d', path as any)
    .attr('fill', 'none')
    .attr('stroke', 'rgba(255,255,255,0.48)')
    .attr('stroke-width', 0.8)
    .attr('stroke-linejoin', 'round');

  const x = d3.scaleLinear().domain([0, spec.width]).range([0, plotW]);
  const y = d3.scaleLinear().domain([0, spec.height]).range([0, plotH]);
  g.append('g')
    .attr('transform', `translate(0,${plotH})`)
    .call(d3.axisBottom(x).ticks(4).tickSizeOuter(0))
    .call(styleAxis);
  g.append('g')
    .call(d3.axisLeft(y).ticks(4).tickSizeOuter(0))
    .call(styleAxis);

  g.append('text')
    .attr('x', plotW)
    .attr('y', plotH + 28)
    .attr('text-anchor', 'end')
    .attr('fill', 'rgba(138,160,184,0.85)')
    .attr('font-size', 10)
    .text('x');

  g.append('text')
    .attr('x', -26)
    .attr('y', 2)
    .attr('fill', 'rgba(138,160,184,0.85)')
    .attr('font-size', 10)
    .text('y');

  const legendX = width - margin.right + 22;
  const legendY = margin.top;
  const legendH = plotH;
  const defs = svg.append('defs');
  const grad = defs.append('linearGradient')
    .attr('id', `grad-${spec.id}-${Math.random().toString(36).slice(2)}`)
    .attr('x1', '0%')
    .attr('x2', '0%')
    .attr('y1', '100%')
    .attr('y2', '0%');
  d3.range(0, 1.001, 0.1).forEach((t) => {
    grad.append('stop')
      .attr('offset', `${t * 100}%`)
      .attr('stop-color', d3.interpolateTurbo(t));
  });
  svg.append('rect')
    .attr('x', legendX)
    .attr('y', legendY)
    .attr('width', 12)
    .attr('height', legendH)
    .attr('rx', 6)
    .attr('fill', `url(#${grad.attr('id')})`);

  const l = d3.scaleLinear().domain([min, max || 1]).range([legendY + legendH, legendY]);
  svg.append('g')
    .attr('transform', `translate(${legendX + 15},0)`)
    .call(d3.axisRight(l).ticks(5).tickSize(3))
    .call(styleAxis);

  const hover = svg.append('g').style('display', 'none');
  hover.append('line').attr('y1', margin.top).attr('y2', margin.top + plotH).attr('stroke', 'rgba(255,255,255,0.45)');
  hover.append('line').attr('x1', margin.left).attr('x2', margin.left + plotW).attr('stroke', 'rgba(255,255,255,0.45)');
  const tooltip = hover.append('text')
    .attr('fill', 'white')
    .attr('font-size', 11)
    .attr('paint-order', 'stroke')
    .attr('stroke', 'rgba(0,0,0,0.72)')
    .attr('stroke-width', 3);

  svg.append('rect')
    .attr('x', margin.left)
    .attr('y', margin.top)
    .attr('width', plotW)
    .attr('height', plotH)
    .attr('fill', 'transparent')
    .on('mousemove', (ev) => {
      const [mx, my] = d3.pointer(ev, svg.node());
      const px = Math.max(0, Math.min(spec.width - 1, Math.floor((mx - margin.left) / cellW)));
      const py = Math.max(0, Math.min(spec.height - 1, Math.floor((my - margin.top) / cellH)));
      const value = spec.values[py * spec.width + px];
      hover.style('display', null);
      hover.select('line:nth-child(1)').attr('x1', mx).attr('x2', mx);
      hover.select('line:nth-child(2)').attr('y1', my).attr('y2', my);
      tooltip
        .attr('x', Math.min(width - 110, mx + 8))
        .attr('y', Math.max(18, my - 8))
        .text(`(${px}, ${py}) ${value.toFixed(3)}`);
    })
    .on('mouseleave', () => hover.style('display', 'none'));
}

export function heatmap28ToImageData(values: Float32Array | Uint8Array, alpha = 1): ImageData {
  return heatmapToImageData(values, 28, 28, alpha);
}

export function renderHeatmapCanvas(
  canvas: HTMLCanvasElement,
  values: Float32Array | Uint8Array,
  width = 28,
  height = 28
) {
  const tmp = document.createElement('canvas');
  tmp.width = width;
  tmp.height = height;
  const tmpCtx = tmp.getContext('2d');
  const ctx = canvas.getContext('2d');
  if (!tmpCtx || !ctx) return;

  tmpCtx.putImageData(heatmapToImageData(values, width, height), 0, 0);
  canvas.width = 160;
  canvas.height = 132;

  const grad = ctx.createLinearGradient(0, 0, 0, canvas.height);
  grad.addColorStop(0, '#101827');
  grad.addColorStop(1, '#070a10');
  ctx.fillStyle = grad;
  roundRect(ctx, 0, 0, canvas.width, canvas.height, 10);
  ctx.fill();

  const pad = 14;
  const size = 96;
  ctx.imageSmoothingEnabled = false;
  ctx.shadowColor = 'rgba(0,0,0,0.45)';
  ctx.shadowBlur = 14;
  ctx.drawImage(tmp, 0, 0, width, height, pad, 12, size, size);
  ctx.shadowBlur = 0;

  drawContourLines(ctx, values, width, height, pad, 12, size);
  drawLegend(ctx, pad + size + 12, 18, 12, 78);
}

export function featureMapsFromInk(ink: Float32Array): FeatureMapSpec[] {
  const blur = blur3x3(ink, 28, 28);
  const edge = sobelMagnitude(blur, 28, 28);
  const pooled = poolGrid(blur, 28, 28, 14, 14);
  const coarse = poolGrid(edge, 28, 28, 7, 7);
  return [
    { id: 'stroke', label: 'stroke field', width: 28, height: 28, values: blur },
    { id: 'edge', label: 'edge energy', width: 28, height: 28, values: edge },
    { id: 'pooled', label: 'pooled stroke', width: 14, height: 14, values: pooled },
    { id: 'coarse', label: 'coarse edge', width: 7, height: 7, values: coarse }
  ];
}

export function poolGrid(values: Float32Array, sourceWidth: number, sourceHeight: number, outWidth: number, outHeight: number) {
  const out = new Float32Array(outWidth * outHeight);
  const cellW = sourceWidth / outWidth;
  const cellH = sourceHeight / outHeight;
  for (let oy = 0; oy < outHeight; oy++) {
    for (let ox = 0; ox < outWidth; ox++) {
      let sum = 0;
      let count = 0;
      const x0 = Math.floor(ox * cellW);
      const x1 = Math.min(sourceWidth, Math.ceil((ox + 1) * cellW));
      const y0 = Math.floor(oy * cellH);
      const y1 = Math.min(sourceHeight, Math.ceil((oy + 1) * cellH));
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          sum += values[y * sourceWidth + x];
          count++;
        }
      }
      out[oy * outWidth + ox] = count ? sum / count : 0;
    }
  }
  return out;
}

function heatmapToImageData(values: Float32Array | Uint8Array, width: number, height: number, alpha = 1): ImageData {
  const out = new Uint8ClampedArray(width * height * 4);
  let mn = +Infinity, mx = -Infinity;
  for (let i = 0; i < width * height; i++) {
    const v = values[i];
    mn = Math.min(mn, v);
    mx = Math.max(mx, v);
  }
  const denom = (mx - mn) || 1;
  for (let i = 0; i < width * height; i++) {
    const t = smoothstep((values[i] - mn) / denom);
    const [r, g, b] = turbo(t);
    const j = i * 4;
    out[j] = r;
    out[j + 1] = g;
    out[j + 2] = b;
    out[j + 3] = Math.round(255 * alpha);
  }
  return new ImageData(out, width, height);
}

function blur3x3(values: Float32Array, width: number, height: number) {
  const out = new Float32Array(width * height);
  const k = [1, 2, 1, 2, 4, 2, 1, 2, 1];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      let weight = 0;
      let n = 0;
      for (let yy = -1; yy <= 1; yy++) {
        for (let xx = -1; xx <= 1; xx++) {
          const sx = Math.max(0, Math.min(width - 1, x + xx));
          const sy = Math.max(0, Math.min(height - 1, y + yy));
          sum += values[sy * width + sx] * k[n];
          weight += k[n];
          n++;
        }
      }
      out[y * width + x] = sum / weight;
    }
  }
  return out;
}

function sobelMagnitude(values: Float32Array, width: number, height: number) {
  const out = new Float32Array(width * height);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const gx =
        -values[i - width - 1] + values[i - width + 1] -
        2 * values[i - 1] + 2 * values[i + 1] -
        values[i + width - 1] + values[i + width + 1];
      const gy =
        -values[i - width - 1] - 2 * values[i - width] - values[i - width + 1] +
        values[i + width - 1] + 2 * values[i + width] + values[i + width + 1];
      out[i] = Math.hypot(gx, gy);
    }
  }
  return out;
}

function drawContourLines(
  ctx: CanvasRenderingContext2D,
  values: Float32Array | Uint8Array,
  width: number,
  height: number,
  x0: number,
  y0: number,
  size: number
) {
  let mx = 0;
  for (const value of values) mx = Math.max(mx, value);
  if (mx <= 0) return;

  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.34)';
  ctx.lineWidth = 0.8;
  const stepX = size / width;
  const stepY = size / height;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const t = values[y * width + x] / mx;
      if (t < 0.38 || t > 0.44) continue;
      ctx.strokeRect(x0 + x * stepX, y0 + y * stepY, stepX, stepY);
    }
  }
  ctx.restore();
}

function drawLegend(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number) {
  const grad = ctx.createLinearGradient(0, y + h, 0, y);
  for (let i = 0; i <= 10; i++) {
    const t = i / 10;
    const [r, g, b] = turbo(t);
    grad.addColorStop(t, `rgb(${r},${g},${b})`);
  }
  ctx.fillStyle = grad;
  roundRect(ctx, x, y, w, h, 5);
  ctx.fill();
  ctx.strokeStyle = 'rgba(255,255,255,0.18)';
  ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,255,0.62)';
  ctx.font = '10px system-ui, sans-serif';
  ctx.fillText('hi', x + w + 5, y + 9);
  ctx.fillText('lo', x + w + 5, y + h);
}

function turbo(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t));
  const r = 34.61 + x * (1172.33 + x * (-10793.56 + x * (33300.12 + x * (-38394.49 + x * 14825.05))));
  const g = 23.31 + x * (557.33 + x * (1225.33 + x * (-3574.96 + x * (1073.77 + x * 707.56))));
  const b = 27.2 + x * (3211.1 + x * (-15327.97 + x * (27814.0 + x * (-22569.18 + x * 6838.66))));
  return [clamp255(r), clamp255(g), clamp255(b)];
}

function smoothstep(t: number) {
  return t * t * (3 - 2 * t);
}

function clamp255(v: number) {
  return Math.max(0, Math.min(255, Math.round(v)));
}

function shortLabel(label: string) {
  return label
    .replace('activation', 'act.')
    .replace('channel', 'ch')
    .replace('compressed response', 'compressed')
    .slice(0, 24);
}

function formatTiny(value: number) {
  const abs = Math.abs(value);
  if (abs >= 10) return value.toFixed(1);
  if (abs >= 1) return value.toFixed(2);
  return value.toFixed(3);
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function styleAxis(selection: d3.Selection<SVGGElement, unknown, null, undefined>) {
  selection.selectAll('path,line').attr('stroke', 'rgba(255,255,255,0.16)');
  selection.selectAll('text').attr('fill', 'rgba(138,160,184,0.85)').attr('font-size', 10);
}
