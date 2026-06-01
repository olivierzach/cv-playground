# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: explore.spec.ts >> explore loads metrics, samples grid, and renders softmax for selected sample
- Location: tests/explore.spec.ts:3:1

# Error details

```
Error: expect(locator).toBeVisible() failed

Locator: getByText('ready')
Expected: visible
Timeout: 30000ms
Error: element(s) not found

Call log:
  - Expect "toBeVisible" with timeout 30000ms
  - waiting for getByText('ready')

```

```yaml
- text: MNIST Playground
- navigation:
  - link "Home":
    - /url: /
  - link "Explore":
    - /url: /explore
  - link "Play":
    - /url: /play
- text: Model
- combobox
- separator
- text: Status loading… Test accuracy — Test loss — Selected — → —
- separator
- text: Curated samples (click) Confusion matrix
- img: 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 true pred
- separator
- text: Selected sample
- img: 0 1 2 3 4 5 6 7 8 9 0% 50% 100%
- text: 3D embedding (UMAP) Drag to rotate. Scroll to zoom. Double-click a point to select that sample. Feature maps across layers are next (we have placeholders in
- code
- text: ).
```

# Test source

```ts
  1  | import { test, expect } from '@playwright/test';
  2  | 
  3  | test('explore loads metrics, samples grid, and renders softmax for selected sample', async ({ page }) => {
  4  |   await page.goto('/explore');
  5  | 
  6  |   // Status should reach 'ready'
  7  |   await expect(page.getByText('Status')).toBeVisible();
> 8  |   await expect(page.getByText('ready')).toBeVisible({ timeout: 30_000 });
     |                                         ^ Error: expect(locator).toBeVisible() failed
  9  | 
  10 |   // Confusion matrix should have an svg
  11 |   await expect(page.locator('text=Confusion matrix')).toBeVisible();
  12 |   await expect(page.locator('svg')).toBeVisible();
  13 | 
  14 |   // Sample grid canvas should exist
  15 |   const grid = page.locator('canvas').first();
  16 |   await expect(grid).toBeVisible();
  17 | 
  18 |   // Softmax svg should exist in selected sample panel
  19 |   await expect(page.locator('text=Selected sample')).toBeVisible();
  20 |   await expect(page.locator('div').filter({ hasText: 'Selected sample' })).toBeVisible();
  21 | 
  22 |   // There should be an svg under the bars container (D3)
  23 |   await expect(page.locator('svg').nth(1)).toBeVisible();
  24 | });
  25 | 
```