import { mkdirSync } from 'node:fs';
import { test, expect } from '@playwright/test';

test.skip(process.env.RUN_SCREENSHOTS !== '1', 'screenshot proof suite is opt-in');

const OUT = 'artifacts/screenshots';

test.beforeAll(() => {
  mkdirSync(OUT, { recursive: true });
});

test('captures play empty, sample, and prediction states', async ({ page }) => {
  await page.goto('/play');
  await expect(page.getByText('ready')).toBeVisible({ timeout: 30_000 });
  await page.screenshot({ path: `${OUT}/play-empty.png`, fullPage: true });

  await page.getByRole('button', { name: 'Sample' }).click();
  await expect(page.getByText('Sample label')).toBeVisible();
  await page.screenshot({ path: `${OUT}/play-sample.png`, fullPage: true });

  await page.getByRole('button', { name: 'Predict' }).click();
  await expect(page.getByText('Embedding cloud')).toBeVisible();
  await expect(page.getByText('Logit decision space', { exact: true })).toBeVisible();
  await expect(page.locator('.umap-frame')).toHaveCount(2);
  await expect.poll(
    async () => page.locator('svg.umap-svg').count(),
    { timeout: 30_000 }
  ).toBeGreaterThanOrEqual(1);
  await expect(page.getByText('Feature heatmaps')).toBeVisible();
  await page.screenshot({ path: `${OUT}/play-prediction.png`, fullPage: true });
});

test('captures explore overview, confusion filter, and selected feature maps', async ({ page }) => {
  await page.goto('/explore');
  await expect(page.getByText('ready')).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText('Selected feature maps')).toBeVisible();
  await page.screenshot({ path: `${OUT}/explore-overview.png`, fullPage: true });

  await page.locator('svg rect').first().click();
  await expect(page.getByText('Clear filter')).toBeVisible();
  await page.screenshot({ path: `${OUT}/explore-confusion-filter.png`, fullPage: true });

  await page.locator('canvas').first().click({ position: { x: 20, y: 20 } });
  await expect(page.getByText('Feature map artifacts')).toBeVisible();
  await page.screenshot({ path: `${OUT}/explore-selected-featuremaps.png`, fullPage: true });
});
