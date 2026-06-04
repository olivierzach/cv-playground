import * as d3 from 'd3';
import type { ConvTrace } from '$lib/models/convnet';

export function renderConvTrace(el: HTMLElement, trace: ConvTrace | null) {
  const width = Math.max(250, el.clientWidth || 252);
  const height = 360;
  el.innerHTML = '';

  const svg = d3.select(el)
    .append('svg')
    .attr('class', 'conv-trace-svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('width', '100%')
    .attr('height', height)
    .attr('role', 'img');

  svg.append('rect')
    .attr('width', width)
    .attr('height', height)
    .attr('rx', 8)
    .attr('fill', '#07101d')
    .attr('stroke', 'rgba(255,255,255,0.08)');

  if (!trace) {
    svg.append('text')
      .attr('x', 14)
      .attr('y', 30)
      .attr('fill', 'rgba(138,160,184,0.9)')
      .attr('font-size', 12)
      .text('Conv trace available for CNN models.');
    return;
  }

  svg.append('text')
    .attr('x', 12)
    .attr('y', 18)
    .attr('fill', 'rgba(230,238,247,0.9)')
    .attr('font-size', 11)
    .attr('font-weight', 750)
    .text(`strongest conv1 response: ch ${trace.channel} @ (${trace.outX}, ${trace.outY})`);

  drawOutputMap(svg, trace, width);

  const matrixY = 166;
  drawMatrix(svg, {
    title: 'input patch',
    values: trace.patch,
    size: trace.kernelSize,
    x: 10,
    y: matrixY,
    color: d3.scaleSequential(d3.interpolateMagma).domain([0, 1]),
    signed: false
  });
  drawOperator(svg, 80, matrixY + 42, '×');
  const maxWeight = maxAbs(trace.kernel);
  drawMatrix(svg, {
    title: 'kernel',
    values: trace.kernel,
    size: trace.kernelSize,
    x: 96,
    y: matrixY,
    color: d3.scaleDiverging<string>().domain([-maxWeight, 0, maxWeight]).interpolator(d3.interpolateRdBu),
    signed: true
  });
  drawOperator(svg, 166, matrixY + 42, '=');
  const maxProduct = maxAbs(trace.products);
  drawMatrix(svg, {
    title: 'products',
    values: trace.products,
    size: trace.kernelSize,
    x: 182,
    y: matrixY,
    color: d3.scaleDiverging<string>().domain([-maxProduct, 0, maxProduct]).interpolator(d3.interpolateRdBu),
    signed: true
  });

  const equationY = 300;
  const sumProducts = trace.products.reduce((sum, value) => sum + value, 0);
  const rows = [
    ['Σ patch × kernel', sumProducts],
    ['bias', trace.bias],
    ['pre-ReLU', trace.preActivation],
    ['ReLU output', trace.activation]
  ] as const;

  const row = svg.append('g').attr('transform', `translate(12,${equationY})`);
  rows.forEach(([label, value], index) => {
    const y = index * 14;
    row.append('text')
      .attr('x', 0)
      .attr('y', y)
      .attr('fill', index === rows.length - 1 ? '#fef3c7' : 'rgba(138,160,184,0.9)')
      .attr('font-size', 10.5)
      .text(label);
    row.append('text')
      .attr('x', width - 28)
      .attr('y', y)
      .attr('text-anchor', 'end')
      .attr('fill', index === rows.length - 1 ? '#fef3c7' : 'rgba(230,238,247,0.86)')
      .attr('font-size', 10.5)
      .attr('font-weight', index === rows.length - 1 ? 750 : 600)
      .text(formatValue(value));
  });
}

function drawOutputMap(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, trace: ConvTrace, width: number) {
  const mapSize = 112;
  const x0 = (width - mapSize) / 2;
  const y0 = 38;
  const values = Array.from(trace.output.values);
  const extent = d3.extent(values) as [number, number];
  const color = d3.scaleSequential(d3.interpolateMagma).domain([extent[0] ?? 0, extent[1] || 1]);
  const cellW = mapSize / trace.output.width;
  const cellH = mapSize / trace.output.height;

  svg.append('text')
    .attr('x', x0)
    .attr('y', y0 - 8)
    .attr('fill', 'rgba(138,160,184,0.9)')
    .attr('font-size', 10)
    .text('destination feature map');

  const g = svg.append('g').attr('transform', `translate(${x0},${y0})`);
  g.selectAll('rect')
    .data(values.map((value, index) => ({ value, x: index % trace.output.width, y: Math.floor(index / trace.output.width) })))
    .enter()
    .append('rect')
    .attr('x', (d) => d.x * cellW)
    .attr('y', (d) => d.y * cellH)
    .attr('width', Math.max(1, cellW + 0.2))
    .attr('height', Math.max(1, cellH + 0.2))
    .attr('fill', (d) => color(d.value));

  g.append('rect')
    .attr('x', trace.outX * cellW)
    .attr('y', trace.outY * cellH)
    .attr('width', cellW)
    .attr('height', cellH)
    .attr('fill', 'none')
    .attr('stroke', '#fef3c7')
    .attr('stroke-width', 2.2);

  svg.append('path')
    .attr('d', `M ${width / 2} ${y0 + mapSize + 8} L ${width / 2} ${y0 + mapSize + 24}`)
    .attr('stroke', 'rgba(254,243,199,0.75)')
    .attr('stroke-width', 1.4)
    .attr('marker-end', null);
}

function drawMatrix(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  opts: {
    title: string;
    values: number[];
    size: number;
    x: number;
    y: number;
    color: (value: number) => string;
    signed: boolean;
  }
) {
  const cell = 12;
  const g = svg.append('g').attr('transform', `translate(${opts.x},${opts.y})`);
  g.append('text')
    .attr('x', 0)
    .attr('y', -8)
    .attr('fill', 'rgba(138,160,184,0.9)')
    .attr('font-size', 9.5)
    .text(opts.title);

  g.selectAll('rect')
    .data(opts.values.map((value, index) => ({ value, x: index % opts.size, y: Math.floor(index / opts.size) })))
    .enter()
    .append('rect')
    .attr('x', (d) => d.x * cell)
    .attr('y', (d) => d.y * cell)
    .attr('width', cell - 1)
    .attr('height', cell - 1)
    .attr('rx', 2)
    .attr('fill', (d) => opts.color(d.value))
    .attr('stroke', 'rgba(255,255,255,0.08)');

  g.append('text')
    .attr('x', (opts.size * cell) / 2)
    .attr('y', opts.size * cell + 13)
    .attr('text-anchor', 'middle')
    .attr('fill', 'rgba(230,238,247,0.72)')
    .attr('font-size', 9)
    .text(opts.signed ? signedRange(opts.values) : '0..1');
}

function drawOperator(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, x: number, y: number, text: string) {
  svg.append('text')
    .attr('x', x)
    .attr('y', y)
    .attr('text-anchor', 'middle')
    .attr('fill', 'rgba(230,238,247,0.82)')
    .attr('font-size', 15)
    .attr('font-weight', 800)
    .text(text);
}

function signedRange(values: number[]) {
  const extent = d3.extent(values) as [number, number];
  return `${formatValue(extent[0] ?? 0)}..${formatValue(extent[1] ?? 0)}`;
}

function maxAbs(values: number[]) {
  return Math.max(0.001, ...values.map((value) => Math.abs(value)));
}

function formatValue(value: number) {
  const abs = Math.abs(value);
  if (abs >= 10) return value.toFixed(1);
  if (abs >= 1) return value.toFixed(2);
  return value.toFixed(3);
}
