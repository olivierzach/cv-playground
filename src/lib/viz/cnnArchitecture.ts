import * as d3 from 'd3';
import type { ModelEntry, ModelLayer } from '$lib/models/types';

type Stage = {
  id: string;
  label: string;
  shape: string;
  kind: 'input' | 'conv' | 'pool' | 'flat' | 'dense' | 'logits';
  channels?: number;
  tile?: [number, number];
  op?: string;
};

export function renderCnnArchitecture(el: HTMLElement, model: ModelEntry | null | undefined) {
  const width = Math.max(252, el.clientWidth || 252);
  const height = 470;
  el.innerHTML = '';

  const svg = d3.select(el)
    .append('svg')
    .attr('class', 'cnn-arch-svg')
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

  if (!model?.layers?.length) {
    svg.append('text')
      .attr('x', 14)
      .attr('y', 30)
      .attr('fill', 'rgba(138,160,184,0.9)')
      .attr('font-size', 12)
      .text('CNN architecture metadata unavailable.');
    return;
  }

  const stages = makeStages(model.layers);
  svg.append('text')
    .attr('x', 14)
    .attr('y', 20)
    .attr('fill', 'rgba(230,238,247,0.92)')
    .attr('font-size', 12)
    .attr('font-weight', 800)
    .text(`${model.name} architecture`);

  const y = d3.scalePoint<string>()
    .domain(stages.map((d) => d.id))
    .range([58, height - 36])
    .padding(0.2);

  const x0 = 36;
  const x1 = width - 58;

  stages.forEach((stage, index) => {
    const cy = y(stage.id) ?? 0;
    if (index > 0) {
      const prev = stages[index - 1];
      const py = y(prev.id) ?? 0;
      drawArrow(svg, x1 - 10, py + 18, x1 - 10, cy - 24, stage.op);
    }

    if (stage.kind === 'flat' || stage.kind === 'dense' || stage.kind === 'logits') {
      drawVector(svg, x0 + 18, cy - 22, stage);
    } else {
      drawVolume(svg, x0 + 6, cy - 26, stage);
    }

    svg.append('text')
      .attr('x', x0 + 106)
      .attr('y', cy - 7)
      .attr('fill', 'rgba(230,238,247,0.9)')
      .attr('font-size', 11)
      .attr('font-weight', 750)
      .text(stage.label);

    svg.append('text')
      .attr('x', x0 + 106)
      .attr('y', cy + 9)
      .attr('fill', 'rgba(138,160,184,0.9)')
      .attr('font-size', 10)
      .text(stage.shape);
  });
}

function makeStages(layers: ModelLayer[]): Stage[] {
  const conv1 = layers.find((d) => d.id === 'conv1');
  const pool1 = layers.find((d) => d.id === 'pool1');
  const conv2 = layers.find((d) => d.id === 'conv2');
  const pool2 = layers.find((d) => d.id === 'pool2');
  const flat = pool2 ? pool2.channels * pool2.tile[0] * pool2.tile[1] : 3136;
  return [
    { id: 'input', label: 'Input image', shape: '1 x 28 x 28', kind: 'input', channels: 1, tile: [28, 28] },
    layerStage(conv1, 'Conv 1 + ReLU', 'conv', '5x5 kernel, pad 2, stride 1'),
    layerStage(pool1, 'Max pool 1', 'pool', '2x2 window, stride 2'),
    layerStage(conv2, 'Conv 2 + ReLU', 'conv', '5x5 kernel, pad 2, stride 1'),
    layerStage(pool2, 'Max pool 2', 'pool', '2x2 window, stride 2'),
    { id: 'flatten', label: 'Flatten', shape: `${flat} features`, kind: 'flat', op: 'reshape' },
    { id: 'fc1', label: 'Dense + ReLU', shape: '64 embedding units', kind: 'dense', op: 'linear projection' },
    { id: 'logits', label: 'Class logits', shape: '10 scores -> softmax', kind: 'logits', op: 'linear classifier' }
  ];
}

function layerStage(layer: ModelLayer | undefined, label: string, kind: 'conv' | 'pool', op: string): Stage {
  const tile = layer?.tile ?? [28, 28];
  const channels = layer?.channels ?? 1;
  return {
    id: layer?.id ?? label,
    label,
    shape: `${channels} x ${tile[0]} x ${tile[1]}`,
    kind,
    channels,
    tile,
    op
  };
}

function drawVolume(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, x: number, y: number, stage: Stage) {
  const tile = stage.tile ?? [28, 28];
  const channelDepth = Math.min(9, Math.max(1, Math.round(Math.log2(stage.channels ?? 1) * 1.8)));
  const size = Math.max(24, Math.min(54, tile[0] * 1.55));
  const fill = stage.kind === 'input'
    ? '#e2e8f0'
    : stage.kind === 'conv'
      ? '#38bdf8'
      : '#fbbf24';
  const stroke = stage.kind === 'input' ? '#94a3b8' : 'rgba(255,255,255,0.72)';
  const g = svg.append('g').attr('transform', `translate(${x},${y})`);

  for (let i = channelDepth - 1; i >= 0; i--) {
    g.append('rect')
      .attr('x', i * 2.2)
      .attr('y', i * -1.7)
      .attr('width', size)
      .attr('height', size)
      .attr('rx', 3)
      .attr('fill', fill)
      .attr('opacity', 0.18 + (channelDepth - i) / channelDepth * 0.36)
      .attr('stroke', stroke)
      .attr('stroke-width', 0.7);
  }

  if (stage.kind === 'conv') {
    g.append('rect')
      .attr('x', 9)
      .attr('y', 9)
      .attr('width', Math.max(10, size * 0.22))
      .attr('height', Math.max(10, size * 0.22))
      .attr('fill', 'none')
      .attr('stroke', '#fef3c7')
      .attr('stroke-width', 1.6);
  }

  if (stage.kind === 'pool') {
    for (let ix = 0; ix < 3; ix++) {
      g.append('line')
        .attr('x1', 6 + ix * 10)
        .attr('x2', 6 + ix * 10)
        .attr('y1', 7)
        .attr('y2', Math.min(size - 7, 38))
        .attr('stroke', 'rgba(3,7,18,0.52)')
        .attr('stroke-width', 1);
      g.append('line')
        .attr('x1', 6)
        .attr('x2', Math.min(size - 7, 36))
        .attr('y1', 7 + ix * 10)
        .attr('y2', 7 + ix * 10)
        .attr('stroke', 'rgba(3,7,18,0.52)')
        .attr('stroke-width', 1);
    }
  }
}

function drawVector(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, x: number, y: number, stage: Stage) {
  const g = svg.append('g').attr('transform', `translate(${x},${y})`);
  const n = stage.kind === 'logits' ? 10 : 14;
  const fill = stage.kind === 'logits' ? '#fb7185' : stage.kind === 'dense' ? '#a78bfa' : '#34d399';
  for (let i = 0; i < n; i++) {
    g.append('rect')
      .attr('x', 0)
      .attr('y', i * 3.2)
      .attr('width', stage.kind === 'flat' ? 46 : 26)
      .attr('height', 2.2)
      .attr('rx', 1)
      .attr('fill', fill)
      .attr('opacity', 0.28 + (i / n) * 0.5);
  }
  if (stage.kind === 'flat') {
    g.append('text')
      .attr('x', 54)
      .attr('y', 25)
      .attr('fill', 'rgba(138,160,184,0.9)')
      .attr('font-size', 9)
      .text('64 x 7 x 7');
  }
}

function drawArrow(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  label?: string
) {
  const id = `arrow-${Math.random().toString(36).slice(2)}`;
  svg.append('defs')
    .append('marker')
    .attr('id', id)
    .attr('viewBox', '0 0 10 10')
    .attr('refX', 8)
    .attr('refY', 5)
    .attr('markerWidth', 5)
    .attr('markerHeight', 5)
    .attr('orient', 'auto-start-reverse')
    .append('path')
    .attr('d', 'M 0 0 L 10 5 L 0 10 z')
    .attr('fill', 'rgba(125,211,252,0.78)');

  svg.append('line')
    .attr('x1', x1)
    .attr('y1', y1)
    .attr('x2', x2)
    .attr('y2', y2)
    .attr('stroke', 'rgba(125,211,252,0.55)')
    .attr('stroke-width', 1.2)
    .attr('marker-end', `url(#${id})`);

  if (label) {
    svg.append('text')
      .attr('x', x1 - 5)
      .attr('y', (y1 + y2) / 2)
      .attr('text-anchor', 'end')
      .attr('fill', 'rgba(138,160,184,0.9)')
      .attr('font-size', 8.8)
      .text(label);
  }
}
