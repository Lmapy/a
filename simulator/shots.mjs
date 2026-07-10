/* Screenshot harness (dev tool, not shipped): 390×844 @ dpr2 shots of the
   drill loop + the mockup for side-by-side comparison. Run from app root:
   node shots.mjs <outDir>  (preview server must be on :4173) */
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

const shots = [
  ['drill-shape-verdict', '/#/drill?drill=shape-alphabet&seed=82&state=verdict&pick=P'],
  ['drill-shape-armed', '/#/drill?drill=shape-alphabet&seed=82'],
  ['drill-pocva-verdict', '/#/drill?drill=poc-va-snap&seed=11&state=verdict'],
  ['drill-excess-verdict', '/#/drill?drill=excess-or-poor&seed=5&state=verdict'],
  ['drill-open-verdict', '/#/drill?drill=open-type-ladder&seed=21&state=verdict'],
  ['drill-regime-verdict', '/#/drill?drill=regime-gate&seed=31&state=verdict'],
];

for (const [name, url] of shots) {
  await page.goto(`http://localhost:4173${url}`);
  await page.waitForTimeout(900); // fonts + init + verdict paint
  await page.screenshot({ path: `${out}/${name}.png` });
  console.log(name, 'ok');
}

// block summary (play a 10-rep excess block, first chip every time)
await page.goto('http://localhost:4173/#/drill?drill=excess-or-poor&seed=9');
for (let i = 0; i < 10; i++) {
  await page.locator('.chip').first().click();
  await page.getByRole('button', { name: /NEXT|RESULTS/ }).click();
}
await page.waitForTimeout(400);
await page.screenshot({ path: `${out}/drill-block-summary.png` });
console.log('drill-block-summary ok');

// the mockup itself, for pixel comparison
await page.goto('file:///home/user/a/docs/simulator/design/mockups/screen-drill-loop.html');
await page.waitForTimeout(900);
await page.screenshot({ path: `${out}/mockup-drill-loop.png` });
console.log('mockup ok');

await browser.close();
