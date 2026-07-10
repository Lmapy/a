import { expect, test } from '@playwright/test';

/* The drill loop, end-to-end against the production build.
   Dev params (documented in Drill.svelte): ?drill=&seed=&state=verdict&pick=
   make every assertion deterministic — same seed ⇒ identical item. */

declare global {
  interface Window {
    __auction?: { lastRenderMs?: number; verdictPaintMs?: number };
  }
}

test.describe('drill loop', () => {
  test('verdict moment is complete: glyph + colors + explanation + confidence echo', async ({ page }) => {
    // seed 82 rep 1 = shape item whose measured truth is 'b' (unit-pinned)
    await page.goto('/#/drill?drill=shape-alphabet&seed=82&state=verdict&pick=P');
    await expect(page.getByText('You said P')).toBeVisible();
    await expect(page.getByText('answer b')).toBeVisible();
    // the canonical GDD P/b template, verbatim
    await expect(page.getByText("That's a b, not a P", { exact: false })).toBeVisible();
    // latency vs par (dev latency fixed at 1.8s)
    await expect(page.getByText('1.8s', { exact: false })).toBeVisible();
    await expect(page.getByText('par 3.0', { exact: false })).toBeVisible();
    // chips: picked-wrong carries YOU + ✗ badge; the truth chip is green ✓;
    // the correct answer is never painted red (CVD-redundant glyphs)
    await expect(page.locator('.chip.picked .who')).toHaveText('YOU');
    await expect(page.locator('.chip.picked .badge')).toHaveText('✗');
    await expect(page.locator('.chip.truth .badge')).toHaveText('✓');
    // confidence ride-along echo (implicit 75% for lean)
    await expect(page.getByText('logged 75%')).toBeVisible();
    // replay chip present (player-invoked micro-replay)
    await expect(page.getByRole('button', { name: /Replay/ })).toBeVisible();
  });

  test('full profile redraw stays under the 8ms budget', async ({ page }) => {
    await page.goto('/#/drill?drill=shape-alphabet&seed=82');
    await expect(page.locator('.chip').first()).toBeVisible();
    // sample several full redraws (ResizeObserver-independent: direct redraws
    // happen on arm/verdict; the recorded value is the LAST full draw)
    const ms = await page.evaluate(() => window.__auction?.lastRenderMs);
    expect(ms).toBeDefined();
    expect(ms!).toBeLessThan(8);
  });

  test('tap → verdict paints within the 100ms budget; NEXT advances the block', async ({ page }) => {
    await page.goto('/#/drill?drill=excess-or-poor&seed=5');
    await expect(page.getByText('REP 1/10')).toBeVisible();
    const chip = page.locator('.chip').first();
    await expect(chip).toBeVisible();
    await chip.click();
    await expect(page.getByText(/par 2\.0/)).toBeVisible();
    const paint = await page.evaluate(() => window.__auction?.verdictPaintMs);
    expect(paint).toBeDefined();
    expect(paint!).toBeLessThan(100);
    // no naked verdicts: an explanation always accompanies the glyph
    await expect(page.locator('.explain')).not.toBeEmpty();
    await page.getByRole('button', { name: /NEXT/ }).click();
    await expect(page.getByText('REP 2/10')).toBeVisible();
  });

  test('snap drill: canvas tap snaps to a row and grades against core rows', async ({ page }) => {
    await page.goto('/#/drill?drill=poc-va-snap&seed=11');
    await expect(page.getByText('TAP THE CHART', { exact: false })).toBeVisible();
    const canvas = page.locator('canvas');
    const box = (await canvas.boundingBox())!;
    await canvas.click({ position: { x: Math.round(box.width * 0.3), y: Math.round(box.height * 0.5) } });
    await expect(page.locator('.explain')).toBeVisible();
    await expect(page.locator('.verdict-line')).toBeVisible();
  });

  test('open-type ladder renders the first-90-min line and four chips', async ({ page }) => {
    await page.goto('/#/drill?drill=open-type-ladder&seed=21');
    await expect(page.getByText('How did this session open?')).toBeVisible();
    await expect(page.getByText('FIRST 90 MIN', { exact: false })).toBeVisible();
    await expect(page.locator('.chip')).toHaveCount(4);
  });

  test('regime gate serves a mid-session snapshot with three playbook chips', async ({ page }) => {
    await page.goto('/#/drill?drill=regime-gate&seed=31');
    await expect(page.getByText("Balance or imbalance — what's the playbook?")).toBeVisible();
    await expect(page.locator('.chip')).toHaveCount(3);
    await expect(page.getByText(/open (in value|out of value|out of range)/)).toBeVisible();
  });

  test('a block of 10 ends in the summary: accuracy, latency vs par, Bot Points, base rate', async ({ page }) => {
    test.setTimeout(60_000);
    await page.goto('/#/drill?drill=excess-or-poor&seed=9');
    for (let i = 1; i <= 10; i++) {
      await expect(page.getByText(`REP ${i}/10`)).toBeVisible();
      await page.locator('.chip').first().click();
      await page.getByRole('button', { name: /NEXT|RESULTS/ }).click();
    }
    await expect(page.getByText('BLOCK COMPLETE')).toBeVisible();
    await expect(page.getByText('ACCURACY', { exact: true })).toBeVisible();
    await expect(page.getByText('BOT POINTS', { exact: true })).toBeVisible();
    // the round calibration sentence (GDD §8 screen 3), measured from the block
    await expect(page.getByText(/When you said (sure|lean|guess), you were right \d+%/)).toBeVisible();
    // the base-rate interstitial line is MEASURED from the served block
    await expect(page.getByText(/highlighted extremes were poor \d+% of the time/)).toBeVisible();
    // film strip: one tappable thumbnail per judged profile, tap → replay
    await expect(page.locator('.strip .thumb')).toHaveCount(10);
    await page.locator('.strip .thumb').first().click();
    await expect(page.locator('.film canvas')).toBeVisible();
    await expect(page.locator('.film-line .rep')).toHaveText('REP 1');
    await expect(page.getByRole('button', { name: /AGAIN/ })).toBeVisible();

    // every rep was recorded through schedule/persist: the ledger holds all
    // 10 decisions, the node rating was written, misses entered the queue
    await page.waitForTimeout(400); // persistBlock flush
    const queuedText = (await page.locator('.queued').first().textContent()) ?? '';
    const missMatch = queuedText.match(/(\d+) MISS/);
    const counts = await page.evaluate(async () => {
      const open = indexedDB.open('the-auction');
      const db: IDBDatabase = await new Promise((res, rej) => {
        open.onsuccess = () => res(open.result);
        open.onerror = () => rej(open.error);
      });
      const count = (store: string): Promise<number> =>
        new Promise((res, rej) => {
          const rq = db.transaction(store, 'readonly').objectStore(store).count();
          rq.onsuccess = () => res(rq.result);
          rq.onerror = () => rej(rq.error);
        });
      return {
        decisions: await count('decisions'),
        ratings: await count('ratings'),
        queue: await count('queue'),
      };
    });
    expect(counts.decisions).toBeGreaterThanOrEqual(10);
    expect(counts.ratings).toBeGreaterThanOrEqual(1);
    if (missMatch) expect(counts.queue).toBeGreaterThanOrEqual(Number(missMatch[1]));
  });

  test('canvas is crisp at devicePixelRatio 2 (backing store = css × dpr)', async ({ browser }) => {
    const ctx = await browser.newContext({
      viewport: { width: 390, height: 844 },
      deviceScaleFactor: 2,
    });
    const page = await ctx.newPage();
    await page.goto('/#/drill?drill=shape-alphabet&seed=82');
    await expect(page.locator('.chip').first()).toBeVisible();
    const ok = await page.evaluate(() => {
      const cv = document.querySelector('canvas')!;
      const r = cv.getBoundingClientRect();
      return Math.abs(cv.width - Math.round(r.width * 2)) <= 1 && Math.abs(cv.height - Math.round(r.height * 2)) <= 1;
    });
    expect(ok).toBe(true);
    await ctx.close();
  });
});
