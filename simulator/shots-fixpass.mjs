/* Screenshot harness for the fix pass (dev tool, not shipped):
   · drill block summary with film strip + round calibration sentence
   · film replay open (thumbnail tapped)
   · stats reliability plot with an observed hit rate BELOW 50% (seeded
     through the REAL Dexie ledger — the pipeline the live app uses)
   Run from app root: node shots-fixpass.mjs <outDir>  (preview on :4173) */
import { chromium } from '@playwright/test';

const out = process.argv[2] ?? '.';
const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
});
const ctx = await browser.newContext({
  viewport: { width: 390, height: 844 },
  deviceScaleFactor: 2,
});
const page = await ctx.newPage();

// ---- drill block summary: film strip + calibration sentence ----------------
await page.goto('http://localhost:4173/#/drill?drill=excess-or-poor&seed=9');
for (let i = 0; i < 10; i++) {
  await page.locator('.chip').first().click();
  await page.getByRole('button', { name: /NEXT|RESULTS/ }).click();
}
await page.waitForTimeout(500);
await page.screenshot({ path: `${out}/drill-block-summary.png` });
console.log('drill-block-summary ok');

await page.locator('.strip .thumb').nth(2).click();
await page.waitForTimeout(1800); // let the replay run to the frozen last frame
await page.screenshot({ path: `${out}/drill-block-summary-film.png` });
console.log('drill-block-summary-film ok');

// ---- stats calibration with a sub-50% observed bucket ----------------------
// Seed the real IndexedDB ledger with confidence-tapped decisions whose guess
// bucket observed ≈ 17% — the case the y-axis floor used to pin at 50.
const dctx = await browser.newContext({
  viewport: { width: 1440, height: 900 },
  deviceScaleFactor: 2,
});
const dpage = await dctx.newPage();
await dpage.goto('http://localhost:4173/');
await dpage.evaluate(async () => {
  const P = { sure: 0.9, lean: 0.75, guess: 0.55 };
  const rows = [];
  let k = 0;
  const mk = (conf, hit) => {
    const p = P[conf];
    const b = hit ? (p - 1) ** 2 : p ** 2;
    rows.push({
      decisionId: `shot:${k}`,
      nodeId: 'excess-or-poor',
      drillId: 'excess-or-poor',
      seed: String(1000 + k),
      paramsVersion: 'v1.1.0',
      answer: { itemId: `i${k}`, choice: hit ? 'POOR' : 'EXCESS', confidence: conf, latencyMs: 1400 },
      verdict: {
        correct: hit,
        score: hit ? 100 : 0,
        explanation: 'x',
        explanationTemplateId: 'excess.hit.poor',
        refs: [],
        cardinal: false,
        brier: b,
      },
      at: Date.now() - (200 - k) * 60_000,
      mode: 'rated',
      itemRating: 1139,
    });
    k++;
  };
  for (let i = 0; i < 40; i++) mk('sure', i % 10 < 8); // said 90 → 80%
  for (let i = 0; i < 30; i++) mk('lean', i % 10 < 7); // said 75 → 70%
  for (let i = 0; i < 12; i++) mk('guess', i % 6 === 0); // said 55 → ~17% (BELOW 50)
  await new Promise((res, rej) => {
    const open = indexedDB.open('the-auction');
    open.onsuccess = () => {
      const db = open.result;
      const tx = db.transaction('decisions', 'readwrite');
      const st = tx.objectStore('decisions');
      for (const r of rows) st.put(r);
      tx.oncomplete = () => res();
      tx.onerror = () => rej(tx.error);
    };
    open.onerror = () => rej(open.error);
  });
});
await dpage.goto('http://localhost:4173/#/stats');
await dpage.waitForTimeout(900);
await dpage.screenshot({ path: `${out}/stats-sub50-calibration.png` });
console.log('stats-sub50-calibration ok');
// clean the seeded ledger back out
await dpage.evaluate(async () => {
  await new Promise((res) => {
    const rq = indexedDB.deleteDatabase('the-auction');
    rq.onsuccess = rq.onerror = rq.onblocked = () => res();
  });
});

await browser.close();
