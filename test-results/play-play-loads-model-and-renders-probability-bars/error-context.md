# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: play.spec.ts >> play loads model and renders probability bars
- Location: tests/play.spec.ts:3:1

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
- text: Controls
- combobox:
  - option "MLP (h=256)" [selected]
  - option "Small CNN (16/32)"
- button "Clear"
- text: Draw a digit. On pointer-up, the model predicts instantly (in a Web Worker).
- separator
- text: "Status loading model: MLP (h=256)… Prediction - Confidence 0.0% Draw Tip: draw big and centered; MNIST preprocessing will be improved in V2. Probabilities"
```

# Test source

```ts
  1  | import { test, expect } from '@playwright/test';
  2  | 
  3  | test('play loads model and renders probability bars', async ({ page }) => {
  4  |   await page.goto('/play');
  5  | 
  6  |   // Wait for model ready
  7  |   await expect(page.getByText('Status')).toBeVisible();
> 8  |   await expect(page.getByText('ready')).toBeVisible({ timeout: 30_000 });
     |                                         ^ Error: expect(locator).toBeVisible() failed
  9  | 
  10 |   // Probability chart should render as svg
  11 |   await expect(page.getByText('Probabilities')).toBeVisible();
  12 |   await expect(page.locator('svg')).toBeVisible();
  13 | 
  14 |   // Canvas exists
  15 |   await expect(page.locator('canvas')).toBeVisible();
  16 | });
  17 | 
```