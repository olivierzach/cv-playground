export type SpriteIndex = {
  tileSize: number;
  cols: number;
  count: number;
  labels: number[];
  sampleIds: number[];
};

export async function loadImage(url: string): Promise<HTMLImageElement> {
  const img = new Image();
  img.decoding = 'async';
  img.src = url;
  await img.decode();
  return img;
}

export function drawTile(
  ctx: CanvasRenderingContext2D,
  sprite: HTMLImageElement,
  idx: SpriteIndex,
  tileIndex: number,
  dx: number,
  dy: number,
  dSize: number
) {
  const { tileSize, cols } = idx;
  const sx = (tileIndex % cols) * tileSize;
  const sy = Math.floor(tileIndex / cols) * tileSize;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(sprite, sx, sy, tileSize, tileSize, dx, dy, dSize, dSize);
}
