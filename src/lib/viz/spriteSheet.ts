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

export function drawTileInverted(
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
  const tmp = document.createElement('canvas');
  tmp.width = tileSize;
  tmp.height = tileSize;
  const tmpCtx = tmp.getContext('2d', { willReadFrequently: true });
  if (!tmpCtx) return;

  tmpCtx.drawImage(sprite, sx, sy, tileSize, tileSize, 0, 0, tileSize, tileSize);
  const image = tmpCtx.getImageData(0, 0, tileSize, tileSize);
  for (let i = 0; i < image.data.length; i += 4) {
    image.data[i] = 255 - image.data[i];
    image.data[i + 1] = 255 - image.data[i + 1];
    image.data[i + 2] = 255 - image.data[i + 2];
  }
  tmpCtx.putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp, 0, 0, tileSize, tileSize, dx, dy, dSize, dSize);
}
