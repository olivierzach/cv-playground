import { test, expect } from '@playwright/test';

test('play loads model and renders probability bars', async ({ page }) => {
  await page.goto('/play');

  // Wait for model ready
  await expect(page.getByText('Status')).toBeVisible();
  await expect(page.getByText('ready')).toBeVisible({ timeout: 30_000 });

  // Probability chart should render as svg
  await expect(page.getByText('Probabilities')).toBeVisible();
  await expect(page.locator('svg')).toBeVisible();

  // Canvas exists
  await expect(page.locator('canvas')).toBeVisible();
});
