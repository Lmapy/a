/* Boss screenshot harness (dev tool, not shipped): 390×844 @ dpr2 shots of
   the boss intro + gauntlet moments + the mockup for side-by-side compare.
   Run from app root: node shots-boss.mjs <outDir>  (preview on :4173) */
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
  ['boss-intro', '/#/boss', 1600],
  ['boss-run-decision', '/#/boss?stage=run&seed=20260709&snap=decision1', 1600],
  ['boss-run-verdict-fade', '/#/boss?stage=run&seed=20260709&snap=verdict-fade', 2600],
  ['boss-run-interrupt', '/#/boss?stage=run&seed=20260709&snap=interrupt', 1600],
  ['boss-debrief', '/#/boss?stage=run&seed=20260709&snap=debrief', 1800],
];

for (const [name, url, wait] of shots) {
  await page.goto(`http://localhost:4173${url}`);
  await page.waitForTimeout(wait); // fonts + session build + paint
  await page.screenshot({ path: `${out}/${name}.png` });
  console.log(name, 'ok');
}

// the mockup itself, for pixel comparison
await page.goto('file:///home/user/a/docs/simulator/design/mockups/screen-boss-intro.html');
await page.waitForTimeout(900);
await page.screenshot({ path: `${out}/mockup-boss-intro.png` });
console.log('mockup ok');

await browser.close();
