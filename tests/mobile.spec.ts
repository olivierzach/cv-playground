import { test, expect } from '@playwright/test';

async function expectNoHorizontalOverflow(page: import('@playwright/test').Page) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(1);
}

test('play mobile puts drawing workflow before controls and avoids overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/play');
  await expect(page.getByText('ready')).toBeVisible({ timeout: 30_000 });

  const drawHeading = page.getByText('Draw', { exact: true });
  const controlsHeading = page.getByText('Controls', { exact: true });
  await expect(drawHeading).toBeVisible();
  await expect(controlsHeading).toBeVisible();

  const drawBox = await drawHeading.boundingBox();
  const controlsBox = await controlsHeading.boundingBox();
  expect(drawBox).toBeTruthy();
  expect(controlsBox).toBeTruthy();
  expect(drawBox!.y).toBeLessThan(controlsBox!.y);

  const canvasBox = await page.locator('canvas.draw-canvas').boundingBox();
  expect(canvasBox).toBeTruthy();
  expect(canvasBox!.width).toBeLessThanOrEqual(366);

  await expect(page.getByRole('button', { name: 'Predict' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Sample' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Clear' })).toBeVisible();
  await expect(page.getByText('Embedding cloud')).toBeVisible();
  await expect(page.getByText('Probabilities')).toBeVisible();
  await expectNoHorizontalOverflow(page);
});

test('explore mobile stacks analysis panels without horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/explore');
  await expect(page.getByText('ready')).toBeVisible({ timeout: 30_000 });

  await expect(page.getByText('Model', { exact: true })).toBeVisible();
  await expect(page.getByText('Confusion matrix')).toBeVisible();
  await expect(page.getByText('3D embedding (UMAP)')).toBeVisible();
  await expect(page.getByText('Selected feature maps')).toBeVisible();
  await expectNoHorizontalOverflow(page);
});
