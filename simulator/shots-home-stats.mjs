/* Screenshot harness (dev tool, not shipped): tree @390×844 dpr2 and stats
   @1440×900 dpr2 in ?demo=1 mode, plus the two mockups for comparison.
   Run from app root: node shots-home-stats.mjs <outDir> (preview on :4173) */
import { chromium } from '@playwright/test';

const out = process.argv[2] ?? '.';
const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
});

async function shot(name, url, viewport) {
  const ctx = await browser.newContext({ viewport, deviceScaleFactor: 2 });
  const page = await ctx.newPage();
  await page.goto(url);
  await page.waitForTimeout(1100); // fonts + hydrate + canvas paint
  await page.screenshot({ path: `${out}/${name}.png` });
  console.log(name, 'ok');
  await ctx.close();
}

const mobile = { width: 390, height: 844 };
const desktop = { width: 1440, height: 900 };

await shot('home-tree-demo', 'http://localhost:4173/#/?demo=1', mobile);
await shot('mockup-skill-tree', 'file:///home/user/a/docs/simulator/design/mockups/screen-skill-tree.html', mobile);
await shot('stats-demo', 'http://localhost:4173/#/stats?demo=1', desktop);
await shot('mockup-dashboard', 'file:///home/user/a/docs/simulator/design/mockups/screen-dashboard.html', desktop);
await shot('stats-demo-mobile', 'http://localhost:4173/#/stats?demo=1', mobile);
await shot('home-tree-empty', 'http://localhost:4173/#/', mobile);
await shot('home-tree-desktop', 'http://localhost:4173/#/?demo=1', desktop);
await shot('stats-empty', 'http://localhost:4173/#/stats', desktop);

await browser.close();
