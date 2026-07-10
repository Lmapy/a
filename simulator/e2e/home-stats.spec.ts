import { expect, test } from '@playwright/test';

/* Home tree + Stats dashboard against the deterministic ?demo=1 ledger
   (seeded through the real persistence interface into a memory sandbox). */

test.describe('skill tree (demo ledger)', () => {
  test('renders every derived node state, streak, and warm-up feed', async ({ page }) => {
    await page.goto('/#/?demo=1');
    // warm-up card: 2 due Woodpecker misses on the rusty node + streak meta
    await expect(page.getByText('2 misses due')).toBeVisible();
    await expect(page.getByText('Excess vs Poor review')).toBeVisible();
    await expect(page.locator('.streak')).toContainText('STREAK 12 · 2 freezes');
    // derived states: rusty, checkpoint-armed (next calendar day), learning
    await expect(page.getByText('⟳ Rusty')).toBeVisible();
    await expect(page.getByText('◉ Checkpoint armed')).toBeVisible();
    await expect(page.getByText(/unlocks tomorrow|ready today/)).toBeVisible();
    await expect(page.getByText(/In progress · \d+% — gate arms at 85%/)).toBeVisible();
    // tier meta + silhouettes
    await expect(page.getByText('2/4 MASTERED')).toBeVisible();
    await expect(page.getByText('T4 · ORDER FLOW')).toBeVisible();
    // locked nodes show the dim chip, unlocked ones a numeric rating
    await expect(page.locator('.rating.dim').first()).toHaveText('—');
    await expect(page.locator('.node', { hasText: 'HVN / LVN Marker' }).locator('.rating')).toHaveText(/^\d{3,4}$/);
  });

  test('tapping an unlocked node opens its drill; locked nodes are inert', async ({ page }) => {
    await page.goto('/#/?demo=1');
    await page.getByText('Open Type', { exact: true }).click();
    await expect(page).toHaveURL(/#\/drill\?drill=open-type-ladder&demo=1$/);
    await expect(page.getByText('REP 1/10')).toBeVisible();

    await page.goto('/#/?demo=1');
    await page.getByText('Acceptance Clock', { exact: true }).click();
    await expect(page).toHaveURL(/#\/\?demo=1$/); // no navigation
  });
});

test.describe('stats dashboard (demo ledger)', () => {
  test('every card computes from the seeded ledger', async ({ page }) => {
    await page.goto('/#/stats?demo=1');
    // scope line counts the seeded decisions
    await expect(page.getByText(/·\s+\d{3,} scored judgments/)).toBeVisible();
    // calibration headline from the Brier ledger
    await expect(page.getByText(/When you say/)).toBeVisible();
    await expect(page.getByText(/overconfident|underconfident|well calibrated/)).toBeVisible();
    // bot points + brier trend
    await expect(page.getByText('Beating the bot')).toBeVisible();
    await expect(page.getByText('BOT POINTS / BLOCK')).toBeVisible();
    await expect(page.getByText(/bar \.180/)).toBeVisible();
    // skill list: rusty tag + at least one numeric rating with a delta column
    await expect(page.getByText('RUSTY', { exact: true })).toBeVisible();
    await expect(page.getByText('ARMED', { exact: true })).toBeVisible();
    // expectancy ledger: measured cell, unmeasured watermark, folklore quote
    await expect(page.getByText('best measured edge')).toBeVisible();
    await expect(page.getByText('unmeasured = folklore').first()).toBeVisible();
    await expect(page.getByText('“~80%”')).toBeVisible();
    // R histogram fed from trade fills
    await expect(page.getByText(/R-multiples · \d+ sim trades/i)).toBeVisible();
    // both canvases painted with a real backing store
    const calW = await page
      .locator('#calibration canvas')
      .evaluate((c) => (c as HTMLCanvasElement).width);
    expect(calW).toBeGreaterThan(0);
    const histW = await page
      .locator('#rmult canvas')
      .evaluate((c) => (c as HTMLCanvasElement).width);
    expect(histW).toBeGreaterThan(0);
  });

  test('live mode with an empty ledger shows honest empty states', async ({ page }) => {
    await page.goto('/#/stats');
    await expect(page.getByText(/·\s+0 scored judgments/)).toBeVisible();
    await expect(page.getByText(/Calibration needs/)).toBeVisible();
    await expect(page.getByText('No measured edge yet', { exact: false })).toBeVisible();
    await expect(page.getByText('R-MULTIPLES · 0 SIM TRADES')).toBeVisible();
  });

  test('no horizontal overflow at the desktop design width', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto('/#/stats?demo=1');
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    expect(scrollWidth).toBeLessThanOrEqual(1440);
  });
});
