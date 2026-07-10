import { expect, test } from '@playwright/test';
import { initialRating } from '../src/lib/schedule/glicko';
import { selectKnobs } from '../src/lib/schedule/queue';
import { siblingSeed } from '../src/lib/gen/prng';
import { buildLoopItem } from '../src/lib/drills/items';

/* The delayed mastery checkpoint (GDD §3 mode 6), end-to-end: the gate exam
   is reachable through play, juice-free while running, grants mastery on the
   tree, and unlocks the next tier (the DoD path: anatomy → open-type).

   Truths are computed in Node by the SAME engine the app runs: seed 901 with
   a fresh profile (rating 1500 ⇒ selectKnobs(initialRating)), item seed for
   rep n = siblingSeed('901', 100 + n) — exactly DrillBlock's derivation. */

const SEED = '901';
const DRILL = 'excess-or-poor';

function truthOf(rep: number): string {
  const knobs = selectKnobs(initialRating(DRILL));
  const li = buildLoopItem(DRILL, siblingSeed(SEED, 100 + rep), knobs);
  return String(li.item.groundTruth);
}

test.describe('mastery checkpoint', () => {
  test('checkpoint runs juice-free, masters the node, unlocks open-type', async ({ page }) => {
    test.setTimeout(90_000);
    await page.goto(`/#/drill?drill=${DRILL}&mode=checkpoint&seed=${SEED}`);

    // distinct mode chrome: Checkpoint label + gate counter + slate frame
    await expect(page.getByText('Checkpoint · Excess or Poor')).toBeVisible();
    await expect(page.getByText('GATE 1/10', { exact: true })).toBeVisible();
    await expect(page.getByText(/NO PER-ITEM FEEDBACK/)).toBeVisible();
    await expect(page.locator('.chart.cpframe')).toBeVisible();

    for (let rep = 1; rep <= 10; rep++) {
      await expect(page.getByText(`GATE ${rep}/10`, { exact: false })).toBeVisible();
      const truth = truthOf(rep);
      await page.getByRole('button', { name: truth, exact: true }).click();
      // juice-free: no per-item verdict layer, no truth chip reveal, ever
      await expect(page.locator('.verdict')).toHaveCount(0);
      await expect(page.locator('.chip.truth')).toHaveCount(0);
    }

    // full debrief at the end: pass sentence + mastery grant
    await expect(page.getByText('CHECKPOINT COMPLETE')).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText(/Checkpoint passed/)).toBeVisible();
    await expect(page.getByText(/mastered\. 10\/10 at gate difficulty/)).toBeVisible();
    // juice-free debrief: no Bot Points row on a gate exam
    await expect(page.getByText('BOT POINTS')).toHaveCount(0);

    // the set persisted as ONE complete checkpoint block on the target node
    const idb = await page.evaluate(async () => {
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
      const cp = (rows as { mode: string; nodeId: string }[]).filter(
        (r) => r.mode === 'checkpoint',
      );
      return { total: rows.length, cp: cp.length, node: cp[0]?.nodeId };
    });
    expect(idb.cp).toBe(10);
    expect(idb.node).toBe(DRILL);

    // BACK TO TREE → the node is mastered and the T1 branch is open:
    // Open Type is now a clickable drill row (the DoD reachability path)
    await page.getByRole('link', { name: /BACK TO TREE/ }).click();
    await expect(page.getByText('DAILY WARM-UP')).toBeVisible();
    const excessRow = page.locator('a.node', { hasText: 'Excess vs Poor' });
    await expect(excessRow.locator('.dot.mastered')).toBeVisible();
    await expect(page.getByText('1/4 MASTERED')).toBeVisible();
    const openRow = page.locator('a.node', { hasText: 'Open Type' });
    await expect(openRow).toHaveAttribute('href', '#/drill?drill=open-type-ladder');
  });

  test('an armed-and-ready node on the tree routes into the gate exam', async ({ page }) => {
    // Arm the gate on a real profile: 20 rated reps at 90% / 1.5s YESTERDAY,
    // so armCheckpoint(lastRatedAt) is already past at load time.
    await page.goto('/#/'); // initialize the Dexie schema first
    await expect(page.getByText('DAILY WARM-UP')).toBeVisible();
    const yesterday = Date.now() - 24 * 3600_000;
    await page.evaluate(async (end) => {
      const rows = Array.from({ length: 20 }, (_, i) => ({
        decisionId: `e2e:${i}`,
        nodeId: 'excess-or-poor',
        drillId: 'excess-or-poor',
        seed: '1',
        paramsVersion: 'e2e',
        answer: { itemId: `e2e:${i}`, choice: 'EXCESS', confidence: null, latencyMs: 1500 },
        verdict: {
          correct: i >= 2, // 18/20 = 90%
          score: i >= 2 ? 100 : 0,
          explanation: 'e2e',
          explanationTemplateId: 'e2e',
          refs: [],
          cardinal: false,
          brier: null,
        },
        at: end - (20 - i) * 30_000,
        mode: 'rated',
      }));
      const open = indexedDB.open('the-auction');
      const db: IDBDatabase = await new Promise((res, rej) => {
        open.onsuccess = () => res(open.result);
        open.onerror = () => rej(open.error);
      });
      await new Promise((res, rej) => {
        const tx = db.transaction('decisions', 'readwrite');
        const store = tx.objectStore('decisions');
        for (const r of rows) store.put(r);
        tx.oncomplete = () => res(null);
        tx.onerror = () => rej(tx.error);
      });
      db.close();
    }, yesterday);

    await page.reload();
    const armed = page.locator('a.node', { hasText: 'Excess vs Poor' });
    await expect(armed.getByText('ready today · tap to take it')).toBeVisible();
    await expect(armed).toHaveAttribute('href', '#/drill?drill=excess-or-poor&mode=checkpoint');
    await armed.click();
    await expect(page.getByText('Checkpoint · Excess or Poor')).toBeVisible();
    await expect(page.getByText('GATE 1/10', { exact: false })).toBeVisible();
  });

  test('an armed-but-waiting node keeps its rated link (demo profile)', async ({ page }) => {
    // The demo's shape-alphabet armed the gate TODAY — the delayed checkpoint
    // opens tomorrow (GDD §3), so the row keeps routing to normal practice.
    await page.goto('/#/?demo=1');
    const armed = page.locator('a.node', { hasText: 'Shape Alphabet' });
    await expect(armed.getByText('unlocks tomorrow')).toBeVisible();
    await expect(armed).toHaveAttribute('href', '#/drill?drill=shape-alphabet&demo=1');
  });
});
