import * as d3 from 'd3';

type Conf = { labels: number[]; matrix: number[][] };

export function renderConfusionMatrix(
  el: HTMLElement,
  conf: Conf,
  opts: { onSelect?: (trueLabel: number, predLabel: number) => void } = {}
) {
  const size = Math.min(el.clientWidth || 320, 320);
  const pad = 24;
  const w = size;
  const h = size;

  el.innerHTML = '';
  const svg = d3.select(el).append('svg').attr('width', w).attr('height', h);

  const n = conf.labels.length;
  const grid = (w - pad - 8) / n;
  const max = d3.max(conf.matrix.flat()) ?? 1;

  const color = d3.scaleSequential(d3.interpolateMagma).domain([0, max]);

  const g = svg.append('g').attr('transform', `translate(${pad},8)`);

  const cells = g
    .selectAll('rect')
    .data(conf.matrix.flatMap((row, i) => row.map((v, j) => ({ i, j, v }))))
    .enter()
    .append('rect')
    .attr('x', (d) => d.j * grid)
    .attr('y', (d) => d.i * grid)
    .attr('width', grid - 1)
    .attr('height', grid - 1)
    .attr('fill', (d) => color(d.v))
    .attr('stroke', 'rgba(255,255,255,0.06)')
    .style('cursor', 'pointer')
    .on('click', (_, d) => opts.onSelect?.(conf.labels[d.i], conf.labels[d.j]));

  cells.append('title').text((d) => `true ${conf.labels[d.i]} → pred ${conf.labels[d.j]}: ${d.v}`);

  // axes labels
  const ax = svg.append('g').attr('transform', `translate(${pad},${8 + n * grid})`);
  ax
    .selectAll('text')
    .data(conf.labels)
    .enter()
    .append('text')
    .attr('x', (_, j) => j * grid + grid / 2)
    .attr('y', 14)
    .attr('text-anchor', 'middle')
    .attr('fill', 'rgba(255,255,255,0.65)')
    .attr('font-size', 10)
    .text((d) => d);

  const ay = svg.append('g').attr('transform', `translate(${pad - 6},8)`);
  ay
    .selectAll('text')
    .data(conf.labels)
    .enter()
    .append('text')
    .attr('x', -2)
    .attr('y', (_, i) => i * grid + grid / 2 + 3)
    .attr('text-anchor', 'end')
    .attr('fill', 'rgba(255,255,255,0.65)')
    .attr('font-size', 10)
    .text((d) => d);

  svg
    .append('text')
    .attr('x', 8)
    .attr('y', 14)
    .attr('fill', 'rgba(255,255,255,0.55)')
    .attr('font-size', 10)
    .text('true');
  svg
    .append('text')
    .attr('x', w - 10)
    .attr('y', h - 6)
    .attr('fill', 'rgba(255,255,255,0.55)')
    .attr('font-size', 10)
    .attr('text-anchor', 'end')
    .text('pred');
}
