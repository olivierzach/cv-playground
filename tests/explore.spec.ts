import { test, expect } from '@playwright/test';

test('explore loads metrics, samples grid, and renders softmax for selected sample', async ({ page }) => {
  await page.goto('/explore');

  // Status should reach 'ready'
  await expect(page.getByText('Status')).toBeVisible();
  await expect(page.getByText('ready')).toBeVisible({ timeout: 30_000 });

  // Confusion matrix should render an svg (D3)
  await expect(page.locator('text=Confusion matrix')).toBeVisible();
  await expect(page.locator('svg').first()).toBeVisible();

  // Sample grid canvas should exist
  const grid = page.locator('canvas').first();
  await expect(grid).toBeVisible();

  // Selected sample section should exist
  await expect(page.locator('text=Selected sample')).toBeVisible();

  // Confusion, probability bars, embedding, and feature map plots should all render.
  await expect.poll(async () => page.locator('svg').count()).toBeGreaterThanOrEqual(6);
  await expect.poll(async () => page.locator('svg.heatmap-svg').count()).toBeGreaterThanOrEqual(2);
  await expect(page.locator('svg.umap-svg')).toBeVisible();
});
