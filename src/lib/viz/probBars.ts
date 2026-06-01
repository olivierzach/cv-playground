import * as d3 from 'd3';

export function renderProbBars(el: HTMLElement, probs: Float32Array) {
  const w = el.clientWidth || 280;
  const h = 160;
  el.innerHTML = '';

  const svg = d3.select(el).append('svg').attr('width', w).attr('height', h);

  const data = Array.from({ length: probs.length }, (_, i) => ({ digit: i, p: probs[i] }));
  const margin = { top: 6, right: 10, bottom: 18, left: 18 };
  const iw = w - margin.left - margin.right;
  const ih = h - margin.top - margin.bottom;

  const x = d3
    .scaleBand<number>()
    .domain(data.map((d) => d.digit))
    .range([0, iw])
    .padding(0.22);

  const y = d3.scaleLinear().domain([0, 1]).range([ih, 0]);

  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  g.append('g')
    .call(d3.axisBottom(x).tickSizeOuter(0))
    .attr('transform', `translate(0,${ih})`)
    .call((s) => s.selectAll('path,line').attr('stroke', 'rgba(255,255,255,0.10)'))
    .call((s) => s.selectAll('text').attr('fill', 'rgba(255,255,255,0.65)').attr('font-size', 11));

  g.append('g')
    .call(d3.axisLeft(y).ticks(3).tickFormat((d) => `${Math.round(Number(d) * 100)}%`))
    .call((s) => s.selectAll('path,line').attr('stroke', 'rgba(255,255,255,0.10)'))
    .call((s) => s.selectAll('text').attr('fill', 'rgba(255,255,255,0.65)').attr('font-size', 11));

  g.selectAll('rect')
    .data(data)
    .enter()
    .append('rect')
    .attr('x', (d) => x(d.digit)!)
    .attr('y', (d) => y(d.p))
    .attr('width', x.bandwidth())
    .attr('height', (d) => ih - y(d.p))
    .attr('rx', 4)
    .attr('fill', 'rgba(125,211,252,0.80)');
}
