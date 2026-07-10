import { expect, test } from '@playwright/test';

/* Boss 1 — Trend-Day Fade Gauntlet, end-to-end against the production build.
   Dev params (documented in BossRun.svelte):
     #/boss?stage=run&seed=<dec>&snap=decision1|verdict-fade|interrupt|debrief
   Seed 20260709 is the pinned demo family (unit-pinned: 3 fades, 3 losses,
   daily-limit breach). */

declare global {
  interface Window {
    __auction?: { lastRenderMs?: number };
  }
}

test.describe('boss intro', () => {
  test('mockup content: name, expectation, trap, RoE, best attempt, single CTA', async ({ page }) => {
    await page.goto('/#/boss');
    await expect(page.getByText('Trend-Day Fade Gauntlet')).toBeVisible();
    await expect(page.getByText('~40%')).toBeVisible();
    await expect(page.getByText('Errors are the content.')).toBeVisible();
    // the trap (misconception this boss punishes) + measured tells
    await expect(page.getByText('Every fade from VAH will look perfect')).toBeVisible();
    await expect(page.getByText(/IB .*normal width/)).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText(/pullbacks never exceed/)).toBeVisible();
    await expect(page.getByText(/impulse volume .*rising/)).toBeVisible();
    // rules of engagement (engine constants, R-denominated)
    await expect(page.getByText('Daily loss limit')).toBeVisible();
    await expect(page.getByText('−2.5R')).toBeVisible();
    await expect(page.getByText('Trailing drawdown')).toBeVisible();
    await expect(page.getByText('KILL / HOLD interrupts')).toBeVisible();
    await expect(page.getByText('5.0s')).toBeVisible();
    await expect(page.getByText('A breach ends the session — open positions included.')).toBeVisible();
    // best attempt line (fresh profile: no attempts)
    await expect(page.getByText('BEST ATTEMPT')).toBeVisible();
    // single CTA
    await expect(page.getByRole('button', { name: /ENTER THE GAUNTLET/ })).toBeVisible();
    // demo tape is a computed run: the caption carries its measured summary
    const ms = await page.evaluate(() => window.__auction?.lastRenderMs);
    expect(ms).toBeDefined();
    expect(ms!).toBeLessThan(8);
  });
});

test.describe('boss run', () => {
  test('live playback reaches the first scripted decision and halts', async ({ page }) => {
    await page.goto('/#/boss?stage=run&seed=20260709');
    // 30-min brackets tick by on the panel head while the tape plays
    await expect(page.getByText(/BRACKET [A-M] · \d\d:\d\d/)).toBeVisible({ timeout: 10_000 });
    // decision 1 sits at the first impulse end (bar 105 ⇒ 11:15 for this seed)
    await expect(page.getByText('ORDER TICKET · DECISION 1/3 · 11:15')).toBeVisible({ timeout: 15_000 });
  });

  test('decision moment: ticket forces the declared read before any action', async ({ page }) => {
    await page.goto('/#/boss?stage=run&seed=20260709&snap=decision1');
    await expect(page.getByText('ORDER TICKET · DECISION 1/3', { exact: false })).toBeVisible({ timeout: 10_000 });
    // all three actions locked until the read is declared (GDD §8 screen 6)
    const fade = page.getByRole('button', { name: 'FADE VAH' });
    const go = page.getByRole('button', { name: 'GO WITH' });
    const stand = page.getByRole('button', { name: 'STAND ASIDE' });
    await expect(fade).toBeDisabled();
    await expect(go).toBeDisabled();
    await expect(stand).toBeDisabled();
    await expect(page.getByText('DECLARE THE READ TO UNLOCK THE TICKET')).toBeVisible();
    // engine-computed structural stop/target previews on the ticket
    await expect(page.getByText(/FADE\s+in \d+\.\d{2} · stop \d+\.\d{2} · tgt \d+\.\d{2}/)).toBeVisible();
    // declare a read → ticket unlocks
    await page.locator('#declared-read').selectOption('imbalance');
    await expect(go).toBeEnabled();
    await go.click();
    // legal initiative trade with the right read
    await expect(page.locator('.explain')).toContainText('Imbalance read, initiative trade');
    await page.getByRole('button', { name: /CONTINUE/ }).click();
  });

  test('fading the labeled trend day is cardinal: ✗✗ + forced tells replay', async ({ page }) => {
    await page.goto('/#/boss?stage=run&seed=20260709&snap=verdict-fade');
    await expect(page.locator('.glyph').getByText('✗✗')).toBeVisible({ timeout: 10_000 });
    // the canonical template (one-timeframing since the trend-run start)
    await expect(page.locator('.explain')).toContainText('one-timeframing since 10:30');
    await expect(page.locator('.explain')).toContainText('every one loses');
    // the forced micro-replay surfaces the measured tells
    await expect(page.getByText('THE TELLS OVERRIDDEN')).toBeVisible();
    await expect(page.getByText(/IB .*normal width/)).toBeVisible();
  });

  test('KILL/HOLD reflex interrupts a with-trend holder', async ({ page }) => {
    await page.goto('/#/boss?stage=run&seed=20260709&snap=interrupt');
    await expect(page.getByText(/REFLEX · LONG/)).toBeVisible({ timeout: 10_000 });
    await expect(page.getByRole('button', { name: 'KILL' })).toBeVisible();
    const hold = page.getByRole('button', { name: 'HOLD' });
    await hold.click();
    // holding a with-trend pullback is the correct reflex
    await expect(page.locator('.explain')).toContainText('Trend pullbacks are entries, not exits');
  });

  test('debrief: Read Score vs P&L with the luck line, guide quote, recorded run', async ({ page }) => {
    await page.goto('/#/boss?stage=run&seed=20260709&snap=debrief');
    await expect(page.getByText('READ SCORE')).toBeVisible({ timeout: 15_000 });
    // outcome ≠ decision quality (GDD §5 bright line)
    await expect(page.getByText('Outcome ≠ decision quality — P&L is shown, never scored.')).toBeVisible();
    // snap=debrief fades every decision: the losses are the ERROR channel,
    // never laundered as variance (GDD §5 luck-channel bright line)
    await expect(
      page.getByText(/Your reads earned .* of decisions; wrong reads cost .* — that isn't variance; variance handed you/),
    ).toBeVisible();
    // the GDD debrief line + the guide's trend-day logic (Part VII.1), quoted
    await expect(page.getByText('Every fade looked perfect. Every fade lost. That is what a trend day does.')).toBeVisible();
    await expect(page.getByText(/a market accepting price outside value is not a fade/)).toBeVisible();
    await expect(page.getByText('Guide, Part VII.1', { exact: false })).toBeVisible();
    // timeline carries the three cardinal fades with their realized R
    await expect(page.getByText('FADE VAH · read balance').first()).toBeVisible();
    await expect(page.getByText('−1.0R').first()).toBeVisible();
    // breach ended the run (pinned demo family behavior)
    await expect(page.getByText(/limit breached — session over|drawdown breached — session over/)).toBeVisible();

    // "one thing to drill" queues tomorrow's Woodpecker
    await page.getByRole('button', { name: /ONE THING TO DRILL/ }).click();
    await expect(page.getByRole('button', { name: /QUEUED FOR TOMORROW/ })).toBeVisible();

    // the run is in the ledger: boss-mode decisions with executed trades
    // (rMultiple feeds the Stats R-histogram) + queue entry present
    const counts = await page.evaluate(async () => {
      const open = indexedDB.open('the-auction');
      const db: IDBDatabase = await new Promise((res, rej) => {
        open.onsuccess = () => res(open.result);
        open.onerror = () => rej(open.error);
      });
      const rows: unknown[] = await new Promise((res, rej) => {
        const rq = db.transaction('decisions', 'readonly').objectStore('decisions').getAll();
        rq.onsuccess = () => res(rq.result);
        rq.onerror = () => rej(rq.error);
      });
      const queue: unknown[] = await new Promise((res, rej) => {
        const rq = db.transaction('queue', 'readonly').objectStore('queue').getAll();
        rq.onsuccess = () => res(rq.result);
        rq.onerror = () => rej(rq.error);
      });
      const boss = (rows as { mode: string; trade?: { rMultiple: number }; nodeId: string }[]).filter(
        (r) => r.mode === 'boss',
      );
      return {
        bossDecisions: boss.length,
        trades: boss.filter((r) => r.trade && typeof r.trade.rMultiple === 'number').length,
        reflexes: boss.filter((r) => r.nodeId === 'one-timeframing-buzzer').length,
        queued: queue.length,
      };
    });
    expect(counts.bossDecisions).toBeGreaterThanOrEqual(3);
    expect(counts.trades).toBeGreaterThanOrEqual(3);
    expect(counts.reflexes).toBeGreaterThanOrEqual(1);
    expect(counts.queued).toBeGreaterThanOrEqual(1);

    // best attempt now shows on the intro
    await page.goto('/#/boss');
    await expect(page.getByText(/READ \d+ ·/)).toBeVisible();
    await expect(page.getByText('LAST RUN OF THIS BOSS')).toBeVisible({ timeout: 10_000 });
  });
});
