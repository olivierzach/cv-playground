import { test, expect, type Page } from '@playwright/test';

test.skip(process.env.RUN_FUZZ !== '1', 'seeded fuzz suite is opt-in');

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

async function assertNoBrowserErrors(page: Page, action: () => Promise<void>) {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text());
  });
  await action();
  expect(errors).toEqual([]);
}

test('seeded draw canvas strokes keep prediction panels stable', async ({ page }) => {
  await assertNoBrowserErrors(page, async () => {
    await page.goto('/play');
    await expect(page.getByText('ready')).toBeVisible({ timeout: 30_000 });
    const canvas = page.locator('canvas').first();
    const box = await canvas.boundingBox();
    expect(box).toBeTruthy();
    const rand = rng(20260601);

    for (let s = 0; s < 8; s++) {
      const x0 = box!.x + 30 + rand() * (box!.width - 60);
      const y0 = box!.y + 30 + rand() * (box!.height - 60);
      await page.mouse.move(x0, y0);
      await page.mouse.down();
      for (let i = 0; i < 8; i++) {
        await page.mouse.move(
          box!.x + 20 + rand() * (box!.width - 40),
          box!.y + 20 + rand() * (box!.height - 40)
        );
      }
      await page.mouse.up();
    }

    await page.getByRole('button', { name: 'Predict' }).click();
    await expect(page.getByText('Top predictions')).toBeVisible();
    await expect(page.getByText('Embedding cloud')).toBeVisible();
    await expect(page.locator('svg.umap-svg .umap-query')).toBeVisible();
    await expect(page.getByText('Feature heatmaps')).toBeVisible();
  });
});

test('extreme inputs and switching do not break inference UI', async ({ page }) => {
  await assertNoBrowserErrors(page, async () => {
    await page.goto('/play');
    await expect(page.getByText('ready')).toBeVisible({ timeout: 30_000 });
    const canvas = page.locator('canvas').first();
    const box = await canvas.boundingBox();
    expect(box).toBeTruthy();

    await page.getByRole('button', { name: 'Predict' }).click();
    await expect(page.getByText('Probabilities')).toBeVisible();

    await page.evaluate(() => {
      const c = document.querySelector('canvas') as HTMLCanvasElement;
      const ctx = c.getContext('2d')!;
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, c.width, c.height);
    });
    await page.getByRole('button', { name: 'Predict' }).click();
    await expect(page.getByText('Top predictions')).toBeVisible();

    await page.getByRole('button', { name: 'Clear' }).click();
    await page.mouse.move(box!.x + box!.width / 2, box!.y + box!.height / 2);
    await page.mouse.down();
    await page.mouse.move(box!.x + box!.width / 2 + 3, box!.y + box!.height / 2 + 3);
    await page.mouse.up();
    await page.getByRole('button', { name: 'Predict' }).click();
    await expect(page.getByText('Feature heatmaps')).toBeVisible();

    const select = page.locator('select');
    const values = await select.locator('option').evaluateAll((opts) => opts.map((o) => (o as HTMLOptionElement).value));
    for (const value of values) {
      await select.selectOption(value);
      await expect(page.getByText('ready')).toBeVisible({ timeout: 30_000 });
    }
  });
});

test('explore repeated filtering and sample selection remains interactive', async ({ page }) => {
  await assertNoBrowserErrors(page, async () => {
    await page.goto('/explore');
    await expect(page.getByText('ready')).toBeVisible({ timeout: 30_000 });
    for (let i = 0; i < 3; i++) {
      await page.locator('svg rect').nth(i).click();
      await expect(page.getByText('Filter', { exact: true })).toBeVisible();
      await page.locator('canvas').first().click({ position: { x: 30 + i * 15, y: 30 } });
      await expect(page.getByText('Selected feature maps')).toBeVisible();
    }
    await page.getByRole('button', { name: 'Clear filter' }).click();
    await expect(page.getByRole('button', { name: 'all' })).toBeVisible();
  });
});
