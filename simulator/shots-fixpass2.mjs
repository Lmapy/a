/* Screenshot harness for fix pass 2 (dev tool, not shipped): checkpoint mode,
   boss tape label fixes, mobile boss nav, provisional-rating copy. Run from
   app root: node shots-fixpass2.mjs <outDir> (preview server on :4173) */
import { chromium } from '@playwright/test';

const out = process.argv[2] ?? '.';
const base = 'http://localhost:4173';
const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
});
const ctx = await browser.newContext({
  viewport: { width: 390, height: 844 },
  deviceScaleFactor: 2,
});
const page = await ctx.newPage();

// 1 — checkpoint armed (slate frame, GATE counter, no verdict chrome)
await page.goto(`${base}/#/drill?drill=excess-or-poor&mode=checkpoint&seed=901`);
await page.waitForTimeout(900);
await page.screenshot({ path: `${out}/fix-checkpoint-armed.png` });
console.log('fix-checkpoint-armed ok');

// 2 — the committed beat right after a tap (no truth reveal)
await page.locator('.chip').first().click();
await page.screenshot({ path: `${out}/fix-checkpoint-committed.png` });
console.log('fix-checkpoint-committed ok');

// 3 — checkpoint debrief (play through; first chip each time; auto-advance)
await page.goto(`${base}/#/drill?drill=excess-or-poor&mode=checkpoint&seed=902`);
await page.waitForTimeout(700);
for (let i = 0; i < 10; i++) {
  await page.locator('.chip:not([disabled])').first().click();
  await page.waitForTimeout(500);
}
await page.waitForTimeout(500);
await page.screenshot({ path: `${out}/fix-checkpoint-debrief.png` });
console.log('fix-checkpoint-debrief ok');

// 4 — home after mastery (mastered dot + unlocked branch): seed a passing
// checkpoint set straight into the ledger, then reload the tree
await page.goto(`${base}/#/`);
await page.waitForTimeout(700);
await page.evaluate(async () => {
  const end = Date.now() - 3600_000;
  const rows = Array.from({ length: 10 }, (_, i) => ({
    decisionId: `shot:${i}`,
    nodeId: 'excess-or-poor',
    drillId: 'excess-or-poor',
    seed: '1',
    paramsVersion: 'shot',
    answer: { itemId: `shot:${i}`, choice: 'EXCESS', confidence: null, latencyMs: 1400 },
    verdict: {
      correct: i >= 1, // 9/10 — passes
      score: i >= 1 ? 100 : 0,
      explanation: 'shot',
      explanationTemplateId: 'shot',
      refs: [],
      cardinal: false,
      brier: null,
    },
    at: end - (10 - i) * 30_000,
    mode: 'checkpoint',
  }));
  const open = indexedDB.open('the-auction');
  const db = await new Promise((res, rej) => {
    open.onsuccess = () => res(open.result);
    open.onerror = () => rej(open.error);
  });
  await new Promise((res, rej) => {
    const tx = db.transaction('decisions', 'readwrite');
    for (const r of rows) tx.objectStore('decisions').put(r);
    tx.oncomplete = () => res(null);
    tx.onerror = () => rej(tx.error);
  });
  db.close();
});
await page.reload();
await page.waitForTimeout(700);
await page.screenshot({ path: `${out}/fix-home-after-checkpoint.png` });
console.log('fix-home-after-checkpoint ok');

// 5 — mobile home bottom nav with BOSS (crop the nav strip too)
await page.goto(`${base}/#/?demo=1`);
await page.waitForTimeout(700);
await page.screenshot({ path: `${out}/fix-home-bossnav.png` });
console.log('fix-home-bossnav ok');

// 6 — rated block summary: provisional rating + bot-points window label
await page.goto(`${base}/#/drill?drill=excess-or-poor&seed=9`);
await page.waitForTimeout(700);
for (let i = 0; i < 10; i++) {
  await page.locator('.chip:not([disabled])').first().click();
  await page.getByRole('button', { name: /NEXT|RESULTS/ }).click();
}
await page.waitForTimeout(400);
await page.screenshot({ path: `${out}/fix-block-summary-provisional.png` });
console.log('fix-block-summary-provisional ok');

// 7/8 — boss tape label collision fixes (same snaps the audit flagged)
await page.goto(`${base}/#/boss?stage=run&seed=20260709&snap=verdict-fade`);
await page.waitForTimeout(1200);
await page.screenshot({ path: `${out}/fix-boss-verdict-fade.png` });
console.log('fix-boss-verdict-fade ok');

await page.goto(`${base}/#/boss?stage=run&seed=20260709&snap=interrupt`);
await page.waitForTimeout(1200);
await page.screenshot({ path: `${out}/fix-boss-interrupt.png` });
console.log('fix-boss-interrupt ok');

await browser.close();
