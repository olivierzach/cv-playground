import { test, expect } from '@playwright/test';

test('explore loads metrics, samples grid, and renders softmax for selected sample', async ({ page }) => {
  await page.goto('/explore');

  // Status should reach 'ready'
  await expect(page.getByText('Status')).toBeVisible();
  await expect(page.getByText('ready')).toBeVisible({ timeout: 30_000 });

  // Confusion matrix should have an svg
  await expect(page.locator('text=Confusion matrix')).toBeVisible();
  await expect(page.locator('svg')).toBeVisible();

  // Sample grid canvas should exist
  const grid = page.locator('canvas').first();
  await expect(grid).toBeVisible();

  // Softmax svg should exist in selected sample panel
  await expect(page.locator('text=Selected sample')).toBeVisible();
  await expect(page.locator('div').filter({ hasText: 'Selected sample' })).toBeVisible();

  // There should be an svg under the bars container (D3)
  await expect(page.locator('svg').nth(1)).toBeVisible();
});
