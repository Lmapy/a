/* PLAYTEST AUDIT driver (dev tool, not shipped). Drives the built app on
   :4173 through a full drill block, stats, and a boss run via the DOM,
   logging every verification the audit needs. Run: node playtest-audit.mjs */
import { chromium } from '@playwright/test';
import { readFileSync, mkdirSync } from 'node:fs';

const SHOT_DIR =
  '/tmp/claude-0/-home-user-a/a79ba79b-564a-5a92-9188-47a1d21553f5/scratchpad/playtest';
mkdirSync(SHOT_DIR, { recursive: true });

const truths = JSON.parse(
  readFileSync(
    '/tmp/claude-0/-home-user-a/a79ba79b-564a-5a92-9188-47a1d21553f5/scratchpad/audit-truths.json',
    'utf8',
  ),
).reps;

const findings = [];
const log = (...a) => console.log(...a);
const check = (ok, label) => {
  log(`${ok ? 'PASS' : 'FAIL'}  ${label}`);
  if (!ok) findings.push(label);
};

const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
});
const ctx = await browser.newContext({
  viewport: { width: 390, height: 844 },
  deviceScaleFactor: 2,
});
const page = await ctx.newPage();

const consoleErrors = [];
const badRequests = [];
const externalRequests = [];
page.on('console', (m) => {
  if (m.type() === 'error') consoleErrors.push(m.text());
});
page.on('requestfailed', (r) => badRequests.push(`${r.url()} :: ${r.failure()?.errorText}`));
page.on('response', (r) => {
  if (r.status() >= 400) badRequests.push(`${r.url()} :: HTTP ${r.status()}`);
});
page.on('request', (r) => {
  const u = new URL(r.url());
  if (u.hostname !== 'localhost') externalRequests.push(r.url());
});

const shot = async (name) => {
  await page.screenshot({ path: `${SHOT_DIR}/${name}.png` });
  log(`  [shot] ${name}`);
};
const text = async (sel) => ((await page.locator(sel).textContent()) ?? '').trim();
const color = async (sel) =>
  page.locator(sel).evaluate((el) => getComputedStyle(el).color);

/* ================= 1. HOME (fresh profile) ================= */
log('\n===== HOME (fresh) =====');
await page.goto('http://localhost:4173/#/');
await page.waitForSelector('.tree .node');
await page.waitForTimeout(600);

const fonts = await page.evaluate(() => ({
  inter: document.fonts.check('16px Inter'),
  mono: document.fonts.check('16px "JetBrains Mono"'),
  loaded: [...document.fonts].map((f) => `${f.family} ${f.weight} ${f.status}`),
}));
check(fonts.inter, `Inter font loaded (${fonts.loaded.length} faces)`);
check(fonts.mono, 'JetBrains Mono font loaded');

const startHref = await page.locator('.warmup .start').getAttribute('href');
log('warm-up START href:', startHref);
const nodeStates = await page.$$eval('.tree a.node', (ns) =>
  ns.map((n) => ({
    href: n.getAttribute('href'),
    disabled: n.getAttribute('aria-disabled'),
    text: n.textContent.replace(/\s+/g, ' ').trim().slice(0, 60),
  })),
);
for (const n of nodeStates) log(`  node: [${n.disabled ? 'LOCKED' : 'open '}] ${n.text} -> ${n.href}`);
await shot('01-home-fresh');

/* ================= 2. DRILL BLOCK (10 reps, deliberate mix) ================= */
log('\n===== DRILL BLOCK shape-alphabet seed=777 =====');
// plan: rep -> [answer, confidence]; wrong on reps 2 (B->TREND), 5 (TREND->B), 7 (P->b)
const plan = truths.map((t) => {
  let pick = t.groundTruth;
  if (t.rep === 2) pick = 'TREND';
  if (t.rep === 5) pick = 'B';
  if (t.rep === 7) pick = 'b';
  const conf = ['sure', 'lean', 'guess'][(t.rep - 1) % 3];
  return { rep: t.rep, truth: t.groundTruth, pick, conf, wrong: pick !== t.groundTruth };
});

await page.goto('http://localhost:4173/#/drill?drill=shape-alphabet&seed=777');
await page.waitForSelector('.answers .chip');
await page.waitForTimeout(400);
await shot('02-drill-armed');

const repLog = [];
for (const p of plan) {
  // armed: chips enabled
  await page.waitForFunction(
    (rep) => document.querySelector('.blocklabel')?.textContent?.includes(`REP ${rep}/10`),
    p.rep,
  );
  await page.waitForTimeout(120);
  // confidence tap
  await page.locator('.confrow .seg', { hasText: p.conf }).click();
  const confOn = await page
    .locator('.confrow .seg.on')
    .evaluate((el) => el.textContent.trim());
  // answer tap
  await page.locator(`.answers .chip:has(.letter:text-is("${p.pick}"))`).click();
  await page.waitForSelector('.verdict .verdict-line');

  const glyph = await text('.verdict-line .glyph');
  const glyphColor = await color('.verdict-line .glyph');
  const explain = await text('.verdict .explain');
  const logged = (await page.locator('.confrow .logged').count())
    ? await text('.confrow .logged')
    : null;
  const latency = await text('.verdict .latency');
  const pickedChip = await page.locator('.chip.picked').count();
  const truthChip = await page.locator('.chip.truth').count();
  repLog.push({ ...p, glyph, glyphColor, explain, logged, latency, confOn, pickedChip, truthChip });
  log(
    `rep ${p.rep}: pick=${p.pick} truth=${p.truth} conf=${p.conf} -> ${glyph} (${glyphColor}) logged="${logged}"`,
  );
  log(`   explain: ${explain}`);

  if (p.rep === 1) await shot('03-drill-verdict-correct');
  if (p.rep === 2) await shot('04-drill-verdict-wrong');
  if (p.rep === 7) await shot('05-drill-verdict-pb-confusion');

  const isCorrect = !p.wrong;
  check(glyph === (isCorrect ? '✓' : '✗'), `rep ${p.rep} glyph icon matches verdict`);
  const expectColor = isCorrect ? 'rgb(27, 175, 122)' : 'rgb(230, 103, 103)';
  check(glyphColor === expectColor, `rep ${p.rep} glyph color ${glyphColor} == token ${expectColor}`);
  check(explain.length > 20, `rep ${p.rep} explanation non-empty`);
  check(confOn === p.conf, `rep ${p.rep} confidence tap registered (${confOn})`);
  check(logged !== null && /logged \d+%/.test(logged), `rep ${p.rep} confidence logged% shown (${logged})`);
  if (!isCorrect) check(truthChip === 1, `rep ${p.rep} truth chip highlighted on miss`);

  // next / results
  await page.locator('.footer .next').click();
}

/* ================= 3. BLOCK SUMMARY ================= */
log('\n===== BLOCK SUMMARY =====');
await page.waitForSelector('.summary');
await page.waitForTimeout(300);
const ledes = await page.$$eval('.summary .lede', (els) => els.map((e) => e.textContent.trim()));
const sumgrid = await page.$$eval('.sumgrid dt, .sumgrid dd', (els) =>
  els.map((e) => e.textContent.replace(/\s+/g, ' ').trim()),
);
const queued = await page.$$eval('.queued', (els) => els.map((e) => e.textContent.trim()));
log('ledes:', JSON.stringify(ledes));
log('sumgrid:', JSON.stringify(sumgrid));
log('queued:', JSON.stringify(queued));
await shot('06-block-summary');

check(sumgrid.includes('ACCURACY') && sumgrid.includes('7/10'), 'summary accuracy 7/10 shown');
const bpIdx = sumgrid.indexOf('BOT POINTS');
check(bpIdx >= 0 && /^[+−-]?\d+(\.\d+)?$/.test(sumgrid[bpIdx + 1] ?? ''), `Bot Points numeric (${sumgrid[bpIdx + 1]})`);
const rtIdx = sumgrid.indexOf('RATING');
log('rating row:', sumgrid[rtIdx + 1]);
check(/1,?500 → /.test(sumgrid[rtIdx + 1] ?? ''), 'rating movement from 1500 shown');
check(ledes.some((l) => /When you said (sure|lean|guess), you were right \d+%/.test(l)), 'round calibration sentence present');
check(queued.some((q) => /3 MISSES QUEUED/.test(q)), 'misses queued line (3)');

// film strip replay
await page.locator('.strip .thumb').nth(1).click();
await page.waitForTimeout(1700);
await shot('07-summary-film-replay');
check((await page.locator('.film').count()) === 1, 'film strip replay opens');

/* ================= 4. STATS ================= */
log('\n===== STATS =====');
await page.locator('.summary-actions a.next.ghost').click(); // HOME
await page.waitForTimeout(400);
await shot('08-home-after-block');
await page.goto('http://localhost:4173/#/stats');
await page.waitForSelector('#calibration');
await page.waitForTimeout(800);
const scope = await text('.scope');
const calLede = await text('#calibration .lede');
log('scope:', scope);
log('calibration lede:', calLede);
const calDots = await page.evaluate(() => {
  const c = document.querySelector('#calibration canvas');
  if (!c) return -1;
  // count non-background pixels near dot color is fragile; instead expose n via view? fallback: check canvas is painted
  const g = c.getContext('2d');
  const d = g.getImageData(0, 0, c.width, c.height).data;
  let painted = 0;
  for (let i = 0; i < d.length; i += 4) if (d[i + 3] > 0) painted++;
  return painted;
});
log('calibration canvas painted px:', calDots);
log(
  /When you say \d+%/.test(calLede)
    ? 'calibration headline: qualified-bucket sentence'
    : `calibration headline: empty-line fallback (<25 taps, by design): ${calLede}`,
);
check(calDots > 500, 'calibration canvas painted');
check(/\b30 scored judgments|\b\d+ scored judgments/.test(scope), `scope line (${scope})`);

const skillRow = await page
  .locator('#skills .skill', { hasText: 'Shape Alphabet' })
  .evaluate((el) => el.textContent.replace(/\s+/g, ' ').trim());
log('shape-alphabet skill row:', skillRow);
check(!/1500(?!\d)/.test(skillRow.replace('Shape Alphabet', '')) || /[↑↓]/.test(skillRow), `skill row shows moved rating/delta (${skillRow})`);
await shot('09-stats-after-block');
await page.locator('#calibration').scrollIntoViewIfNeeded();

/* ============ 4b. OPEN-TYPE + POC/VA SNAP smoke (DoD reachability) ======== */
log('\n===== OPEN-TYPE ROUND (direct route) =====');
await page.goto('http://localhost:4173/#/drill?drill=open-type-ladder&seed=21');
await page.waitForSelector('.answers .chip');
await page.waitForTimeout(400);
const otQuestion = await text('.chart-head .q');
const otChips = await page.$$eval('.answers .chip .letter', (els) => els.map((e) => e.textContent.trim()));
log('open-type question:', otQuestion, '| chips:', JSON.stringify(otChips));
await shot('09b-open-type-armed');
await page.locator('.answers .chip').first().click();
await page.waitForSelector('.verdict .verdict-line');
const otExplain = await text('.verdict .explain');
const otGlyph = await text('.verdict-line .glyph');
log(`open-type verdict: ${otGlyph} :: ${otExplain}`);
check(otExplain.length > 20, 'open-type verdict explanation non-empty');
check(/open/i.test(otExplain), 'open-type explanation references the open');
await shot('09c-open-type-verdict');

log('\n===== POC/VA SNAP (tap the chart) =====');
await page.goto('http://localhost:4173/#/drill?drill=poc-va-snap&seed=11');
await page.waitForSelector('.tapline');
await page.waitForTimeout(400);
const snapQ = await text('.chart-head .q');
log('snap question:', snapQ);
// tap the middle of the canvas (snap-to-row means no precision needed)
const cbox = await page.locator('.chart-body canvas').boundingBox();
await page.mouse.click(cbox.x + cbox.width * 0.4, cbox.y + cbox.height * 0.55);
await page.waitForSelector('.verdict .verdict-line');
const snapExplain = await text('.verdict .explain');
const snapGlyph = await text('.verdict-line .glyph');
const snapPick = await text('.verdict-line .you');
log(`snap verdict: ${snapGlyph} pick=${snapPick} :: ${snapExplain}`);
check(snapExplain.length > 20, 'snap verdict explanation non-empty');
check(/\d/.test(snapPick), 'snap pick echoes tapped price');
await shot('09d-pocva-verdict');

/* ================= 5. BOSS ================= */
log('\n===== BOSS =====');
await page.goto('http://localhost:4173/#/boss');
await page.waitForTimeout(700);
await shot('10-boss-intro');
const enterBtn = page.locator('button', { hasText: /ENTER|BEGIN|START/i }).first();
const introText = await page.evaluate(() => document.body.innerText.slice(0, 1500));
log('intro text head:', introText.split('\n').slice(0, 12).join(' | '));
await enterBtn.click();
log('clicked enter; waiting for first decision…');

// wait for the first order ticket (playback runs ~45 bars/s)
await page.waitForSelector('.ticket', { timeout: 60000 });
await page.waitForTimeout(200);
await shot('11-boss-decision1');
const ticketLabel = await text('.ticket .seclabel');
log('ticket:', ticketLabel);
const fadeDisabledBefore = await page.locator('.ticket .chip', { hasText: 'FADE VAH' }).isDisabled();
check(fadeDisabledBefore, 'actions locked until read declared');
await page.selectOption('#declared-read', 'balance');
const fadeDisabledAfter = await page.locator('.ticket .chip', { hasText: 'FADE VAH' }).isDisabled();
check(!fadeDisabledAfter, 'declaring read unlocks ticket');
await page.locator('.ticket .chip', { hasText: 'FADE VAH' }).click();
await page.waitForSelector('.zone .verdict');
const bossGlyph = await text('.zone .verdict-line .glyph');
const bossExplain = await text('.zone .verdict .explain');
log(`boss decision 1 verdict: ${bossGlyph} :: ${bossExplain}`);
check(bossGlyph === '✗✗', 'FADE on trend day graded cardinal (double-cross)');
check(/one-timeframing|imbalance|trend/i.test(bossExplain), 'cardinal explanation quotes trend/OTF logic');
const tells = await page.locator('.tellsbox').count();
check(tells === 1, 'forced micro-replay tells box present');
await page.waitForFunction(() => {
  const b = document.querySelector('.footer .next');
  return b && !b.disabled;
}, null, { timeout: 8000 });
const tellRows = await page.$$eval('.tellrow', (els) => els.map((e) => e.textContent.replace(/\s+/g, ' ').trim()));
log('tells:', JSON.stringify(tellRows));
await shot('12-boss-cardinal-verdict');

// drive to the end: FADE every decision, HOLD every interrupt
let guard = 40;
let sawInterrupt = 0;
let interruptShot = false;
while (guard-- > 0) {
  const state = await page.evaluate(() => {
    if (document.body.innerText.includes('READ SCORE')) return 'debrief';
    if ([...document.querySelectorAll('.footer .next')].some((b) => b.textContent.includes('TO THE DEBRIEF'))) return 'breach';
    if (document.querySelector('.ticket')) return 'decision';
    if (document.querySelector('.interrupt')) return 'interrupt';
    if (document.querySelector('.zone .verdict .explain')) return 'verdict';
    return 'playing';
  });
  if (state === 'debrief') break;
  if (state === 'decision') {
    await page.selectOption('#declared-read', 'balance');
    await page.locator('.ticket .chip', { hasText: 'FADE VAH' }).click();
    await page.waitForTimeout(200);
  } else if (state === 'interrupt') {
    sawInterrupt++;
    if (!interruptShot) {
      await shot('13-boss-interrupt');
      interruptShot = true;
    }
    await page.locator('.interrupt .chip', { hasText: 'HOLD' }).click();
    await page.waitForTimeout(200);
  } else if (state === 'verdict') {
    try {
      await page.waitForFunction(() => {
        const b = document.querySelector('.footer .next');
        return b && !b.disabled;
      }, null, { timeout: 5000 });
      await page.locator('.footer .next').click();
    } catch { /* verdict gone */ }
    await page.waitForTimeout(150);
  } else if (state === 'breach') {
    await shot('14-boss-breach');
    await page.locator('.footer .next', { hasText: 'TO THE DEBRIEF' }).click();
    await page.waitForTimeout(300);
  } else {
    await page.waitForTimeout(700);
  }
}
await page.waitForTimeout(500);
const debriefText = await page.evaluate(() => document.body.innerText);
log('interrupts seen:', sawInterrupt);
log('debrief head:', debriefText.split('\n').slice(0, 30).join(' | '));
await shot('15-boss-debrief');
check(/READ SCORE|Read Score/i.test(debriefText), 'debrief shows Read Score');
check(/FAIL|FAILED|NOT PASSED|✗/i.test(debriefText), 'boss lost (fail state shown)');
check(/fade|one-timefram|trend/i.test(debriefText), 'debrief explains the cardinal fade error');

/* ================= wrap ================= */
log('\n===== NETWORK / CONSOLE =====');
log('console errors:', JSON.stringify(consoleErrors, null, 1));
log('failed/4xx requests:', JSON.stringify(badRequests, null, 1));
log('external requests:', JSON.stringify(externalRequests, null, 1));
check(consoleErrors.length === 0, 'no console errors');
check(badRequests.length === 0, 'no failed requests');
check(externalRequests.length === 0, 'no external network resources');

log('\n===== FINDINGS =====');
findings.forEach((f, i) => log(`${i + 1}. ${f}`));
log(`total FAIL: ${findings.length}`);

await browser.close();
