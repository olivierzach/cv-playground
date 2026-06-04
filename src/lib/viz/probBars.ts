import * as d3 from 'd3';

type ProbPoint = { digit: number; p: number; rank: number };
type LogitPoint = { digit: number; v: number; rank: number };

export function renderProbBars(el: HTMLElement, probs: Float32Array) {
  const w = el.clientWidth || 280;
  const h = 188;
  el.innerHTML = '';

  const svg = d3.select(el)
    .append('svg')
    .attr('class', 'prob-svg')
    .attr('width', w)
    .attr('height', h)
    .attr('viewBox', `0 0 ${w} ${h}`)
    .attr('role', 'img');

  const data: ProbPoint[] = Array.from({ length: 10 }, (_, digit) => ({ digit, p: probs[digit] ?? 0 }))
    .map((d) => ({ ...d, rank: 1 + Array.from(probs).filter((p) => p > d.p).length }));
  const margin = { top: 20, right: 16, bottom: 28, left: 34 };
  const iw = w - margin.left - margin.right;
  const ih = h - margin.top - margin.bottom;
  const maxP = Math.max(0.04, d3.max(data, (d) => d.p) ?? 0);
  const density = probabilityDensity(data, 0.58);
  const maxDensity = Math.max(0.001, d3.max(density, (d) => d.y) ?? 0);
  const scaledDensity = density.map((d) => ({ ...d, y: d.y * (maxP / maxDensity) }));

  const x = d3.scaleLinear().domain([0, 9]).range([0, iw]);
  const y = d3.scaleLinear()
    .domain([0, Math.min(1, Math.max(0.12, maxP * 1.16))])
    .range([ih, 0])
    .nice();

  const line = d3.line<{ x: number; y: number }>()
    .x((d) => x(d.x))
    .y((d) => y(d.y))
    .curve(d3.curveCatmullRom.alpha(0.65));
  const area = d3.area<{ x: number; y: number }>()
    .x((d) => x(d.x))
    .y0(ih)
    .y1((d) => y(d.y))
    .curve(d3.curveCatmullRom.alpha(0.65));

  const g = baseChart(svg, w, h, margin, iw, ih);
  drawProbabilityAxes(g, x, y, iw, ih);

  const defs = svg.append('defs');
  const grad = defs.append('linearGradient')
    .attr('id', `prob-kde-${Math.random().toString(36).slice(2)}`)
    .attr('x1', '0%')
    .attr('x2', '0%')
    .attr('y1', '100%')
    .attr('y2', '0%');
  grad.append('stop').attr('offset', '0%').attr('stop-color', 'rgba(34,211,238,0.02)');
  grad.append('stop').attr('offset', '55%').attr('stop-color', 'rgba(34,211,238,0.18)');
  grad.append('stop').attr('offset', '100%').attr('stop-color', 'rgba(251,191,36,0.36)');

  g.append('path')
    .datum(scaledDensity)
    .attr('d', area)
    .attr('fill', `url(#${grad.attr('id')})`);

  g.append('path')
    .datum(scaledDensity)
    .attr('d', line)
    .attr('fill', 'none')
    .attr('stroke', '#fbbf24')
    .attr('stroke-width', 2.1)
    .attr('stroke-linejoin', 'round');

  g.selectAll('line.prob-stem')
    .data(data.filter((d) => d.p > 0))
    .enter()
    .append('line')
    .attr('class', 'prob-stem')
    .attr('x1', (d) => x(d.digit))
    .attr('x2', (d) => x(d.digit))
    .attr('y1', ih)
    .attr('y2', (d) => y(d.p))
    .attr('stroke', (d) => d.rank <= 3 ? 'rgba(251,191,36,0.48)' : 'rgba(125,211,252,0.18)')
    .attr('stroke-width', 1);

  g.selectAll('circle.prob-dot')
    .data(data)
    .enter()
    .append('circle')
    .attr('class', 'prob-dot')
    .attr('cx', (d) => x(d.digit))
    .attr('cy', (d) => y(d.p))
    .attr('r', (d) => d.rank <= 3 ? 5.4 : 3.5)
    .attr('fill', (d) => d.rank <= 3 ? '#fff7ed' : '#bfdbfe')
    .attr('stroke', (d) => d.rank <= 3 ? '#f59e0b' : '#38bdf8')
    .attr('stroke-width', 1.5);

  g.selectAll('text.prob-rank')
    .data(data.filter((d) => d.rank <= 3 && d.p > 0))
    .enter()
    .append('text')
    .attr('class', 'prob-rank')
    .attr('x', (d) => x(d.digit))
    .attr('y', (d) => Math.max(9, y(d.p) - 10))
    .attr('text-anchor', 'middle')
    .attr('fill', '#fef3c7')
    .attr('font-size', 10)
    .attr('font-weight', 700)
    .text((d) => `#${d.rank}`);

  attachClassHover(svg, g, data, w, ih, margin, x, (d) => `${d.digit}: ${(d.p * 100).toFixed(2)}%`);
}

export function renderLogitMultiples(el: HTMLElement, logits: Float32Array) {
  const w = el.clientWidth || 280;
  const h = 168;
  el.innerHTML = '';

  const svg = d3.select(el)
    .append('svg')
    .attr('class', 'logit-svg')
    .attr('viewBox', `0 0 ${w} ${h}`)
    .attr('width', '100%')
    .attr('height', h)
    .attr('role', 'img');

  const data: LogitPoint[] = Array.from({ length: 10 }, (_, digit) => ({ digit, v: logits[digit] ?? 0 }))
    .map((d) => ({ ...d, rank: 1 + Array.from(logits).filter((v) => v > d.v).length }));
  const margin = { top: 18, right: 16, bottom: 28, left: 34 };
  const iw = w - margin.left - margin.right;
  const ih = h - margin.top - margin.bottom;
  const density = signedDensity(data, 0.64);
  const maxAbs = Math.max(
    1,
    ...data.map((d) => Math.abs(d.v)),
    ...density.map((d) => Math.abs(d.y))
  );

  const x = d3.scaleLinear().domain([0, 9]).range([0, iw]);
  const y = d3.scaleLinear().domain([-maxAbs * 1.12, maxAbs * 1.12]).range([ih, 0]).nice();
  const zeroY = y(0);
  const line = d3.line<{ x: number; y: number }>()
    .x((d) => x(d.x))
    .y((d) => y(d.y))
    .curve(d3.curveCatmullRom.alpha(0.65));
  const area = d3.area<{ x: number; y: number }>()
    .x((d) => x(d.x))
    .y0(zeroY)
    .y1((d) => y(d.y))
    .curve(d3.curveCatmullRom.alpha(0.65));

  const g = baseChart(svg, w, h, margin, iw, ih);
  drawLogitAxes(g, x, y, iw, ih);

  const positive = splitBySign(density, 1);
  const negative = splitBySign(density, -1);
  g.selectAll('path.logit-positive')
    .data(positive)
    .enter()
    .append('path')
    .attr('class', 'logit-positive')
    .attr('d', area)
    .attr('fill', 'rgba(34,211,238,0.20)');
  g.selectAll('path.logit-negative')
    .data(negative)
    .enter()
    .append('path')
    .attr('class', 'logit-negative')
    .attr('d', area)
    .attr('fill', 'rgba(251,113,133,0.18)');

  g.append('path')
    .datum(density)
    .attr('d', line)
    .attr('fill', 'none')
    .attr('stroke', '#7dd3fc')
    .attr('stroke-width', 2.1)
    .attr('stroke-linejoin', 'round');

  g.selectAll('line.logit-stem')
    .data(data)
    .enter()
    .append('line')
    .attr('class', 'logit-stem')
    .attr('x1', (d) => x(d.digit))
    .attr('x2', (d) => x(d.digit))
    .attr('y1', zeroY)
    .attr('y2', (d) => y(d.v))
    .attr('stroke', (d) => d.v >= 0 ? 'rgba(34,211,238,0.34)' : 'rgba(251,113,133,0.34)')
    .attr('stroke-width', 1);

  g.selectAll('circle.logit-dot')
    .data(data)
    .enter()
    .append('circle')
    .attr('class', 'logit-dot')
    .attr('cx', (d) => x(d.digit))
    .attr('cy', (d) => y(d.v))
    .attr('r', (d) => d.rank <= 3 ? 5 : 3.5)
    .attr('fill', (d) => d.v >= 0 ? '#cffafe' : '#ffe4e6')
    .attr('stroke', (d) => d.v >= 0 ? '#06b6d4' : '#fb7185')
    .attr('stroke-width', 1.5);

  g.selectAll('text.logit-value')
    .data(data.filter((d) => d.rank <= 3))
    .enter()
    .append('text')
    .attr('class', 'logit-value')
    .attr('x', (d) => x(d.digit))
    .attr('y', (d) => d.v >= 0 ? y(d.v) - 9 : y(d.v) + 16)
    .attr('text-anchor', 'middle')
    .attr('fill', 'rgba(230,238,247,0.86)')
    .attr('font-size', 9.5)
    .attr('font-weight', 700)
    .text((d) => d.v.toFixed(2));

  attachClassHover(svg, g, data, w, ih, margin, x, (d) => `${d.digit}: logit ${d.v.toFixed(3)}`);
}

function baseChart(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  w: number,
  h: number,
  margin: { top: number; right: number; bottom: number; left: number },
  iw: number,
  ih: number
) {
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
  g.append('rect')
    .attr('x', -margin.left + 1)
    .attr('y', -margin.top + 1)
    .attr('width', w - 2)
    .attr('height', h - 2)
    .attr('rx', 8)
    .attr('fill', '#07101d');
  g.append('g')
    .attr('class', 'prob-grid')
    .call(d3.axisLeft(d3.scaleLinear().domain([0, 1]).range([ih, 0])).ticks(4).tickSize(-iw).tickFormat(() => ''))
    .call((s) => s.selectAll('path').remove())
    .call((s) => s.selectAll('line').attr('stroke', 'rgba(255,255,255,0.065)'));
  return g;
}

function drawProbabilityAxes(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  x: d3.ScaleLinear<number, number>,
  y: d3.ScaleLinear<number, number>,
  iw: number,
  ih: number
) {
  g.append('g')
    .attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(x).ticks(10).tickFormat((d) => `${d}`).tickSizeOuter(0))
    .call(styleAxis);
  g.append('g')
    .call(d3.axisLeft(y).ticks(3).tickFormat((d) => `${Math.round(Number(d) * 100)}%`).tickSizeOuter(0))
    .call(styleAxis);
  g.append('text')
    .attr('x', iw)
    .attr('y', ih + 24)
    .attr('text-anchor', 'end')
    .attr('fill', 'rgba(138,160,184,0.85)')
    .attr('font-size', 10)
    .text('digit');
}

function drawLogitAxes(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  x: d3.ScaleLinear<number, number>,
  y: d3.ScaleLinear<number, number>,
  iw: number,
  ih: number
) {
  g.append('line')
    .attr('x1', 0)
    .attr('x2', iw)
    .attr('y1', y(0))
    .attr('y2', y(0))
    .attr('stroke', 'rgba(255,255,255,0.22)');
  g.append('g')
    .attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(x).ticks(10).tickFormat((d) => `${d}`).tickSizeOuter(0))
    .call(styleAxis);
  g.append('g')
    .call(d3.axisLeft(y).ticks(4).tickSizeOuter(0))
    .call(styleAxis);
}

function attachClassHover<T extends { digit: number }>(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  data: T[],
  w: number,
  ih: number,
  margin: { top: number; right: number; bottom: number; left: number },
  x: d3.ScaleLinear<number, number>,
  label: (d: T) => string
) {
  const tooltip = svg.append('g').attr('class', 'prob-tip').style('display', 'none');
  tooltip.append('rect').attr('rx', 6).attr('height', 25).attr('fill', 'rgba(3,7,18,0.92)').attr('stroke', 'rgba(255,255,255,0.18)');
  const tipText = tooltip.append('text').attr('x', 8).attr('y', 16).attr('fill', 'white').attr('font-size', 11);

  const half = (x(1) - x(0)) / 2;
  g.selectAll('rect.hover-zone')
    .data(data)
    .enter()
    .append('rect')
    .attr('class', 'hover-zone')
    .attr('x', (d) => Math.max(0, x(d.digit) - half))
    .attr('y', 0)
    .attr('width', Math.max(1, half * 2))
    .attr('height', ih)
    .attr('fill', 'transparent')
    .style('cursor', 'crosshair')
    .on('mouseenter mousemove', (ev, d) => {
      const [mx, my] = d3.pointer(ev, svg.node());
      const text = label(d);
      tipText.text(text);
      const boxW = Math.max(82, text.length * 6.5 + 16);
      tooltip.select('rect').attr('width', boxW);
      tooltip
        .attr('transform', `translate(${Math.min(w - boxW - 4, mx + 10)},${Math.max(4, my - 30)})`)
        .style('display', null);
    })
    .on('mouseleave', () => tooltip.style('display', 'none'));
}

function probabilityDensity(data: ProbPoint[], bandwidth: number) {
  return d3.range(0, 9.0001, 0.06).map((x) => ({
    x,
    y: data.reduce((sum, d) => sum + d.p * gaussian((x - d.digit) / bandwidth), 0)
  }));
}

function signedDensity(data: LogitPoint[], bandwidth: number) {
  return d3.range(0, 9.0001, 0.06).map((x) => {
    let weight = 0;
    let value = 0;
    for (const d of data) {
      const k = gaussian((x - d.digit) / bandwidth);
      weight += k;
      value += d.v * k;
    }
    return { x, y: weight ? value / weight : 0 };
  });
}

function splitBySign(series: { x: number; y: number }[], sign: 1 | -1) {
  const groups: { x: number; y: number }[][] = [];
  let current: { x: number; y: number }[] = [];
  for (const point of series) {
    const matches = sign > 0 ? point.y >= 0 : point.y <= 0;
    if (matches) {
      current.push(point);
    } else if (current.length) {
      groups.push(current);
      current = [];
    }
  }
  if (current.length) groups.push(current);
  return groups;
}

function gaussian(x: number) {
  return Math.exp(-0.5 * x * x);
}

function styleAxis(selection: d3.Selection<SVGGElement, unknown, null, undefined>) {
  selection.selectAll('path,line').attr('stroke', 'rgba(255,255,255,0.12)');
  selection.selectAll('text').attr('fill', 'rgba(255,255,255,0.65)').attr('font-size', 11);
}
