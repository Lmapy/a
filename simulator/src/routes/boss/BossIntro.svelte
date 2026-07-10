<script lang="ts">
  /* Boss 1 intro — pixel-matched to design/mockups/screen-boss-intro.html
     (design-system §3.9 "Boss framing"). Every number on this screen is
     engine-computed: the tape is a real simulated run (the player's last run
     re-derived from its seed + actions, or the fade-bot demo run before any
     attempt), the trap tells are measured from that session, the RoE values
     are the engine's rule constants, best attempt comes from stored runs. */
  import { onMount } from 'svelte';
  import {
    DAILY_LOSS_LIMIT_R,
    DEMO_SEED,
    KILL_WINDOW_MS,
    OTF_HOLD_THROUGH_BAR,
    TRAILING_DD_R,
    buildBossSession,
    fadeBot,
    measureTells,
    runCaption,
    runPolicy,
  } from '../../lib/boss/engine';
  import type { BossSession, PolicyRun, TrendTells } from '../../lib/boss/engine';
  import { bestAttempt, lastRun } from '../../lib/boss/store';
  import { renderTape, readTapeTheme } from '../../lib/boss/render';
  import type { TapePin, TapeTheme } from '../../lib/boss/render';
  import { devHooks } from '../../lib/render/devhooks';
  import { BOSS1_TRAP_LINE_EM, BOSS1_TRAP_LINE_HEAD, BOSS1_TRAP_LINE_TAIL } from '../../lib/boss/templates';

  const { onenter }: { onenter: () => void } = $props();

  const best = bestAttempt();
  const last = lastRun();

  let session = $state<BossSession | null>(null);
  let run = $state<PolicyRun | null>(null);
  let tells = $state<TrendTells | null>(null);

  let bodyEl = $state<HTMLDivElement | null>(null);
  let canvasEl = $state<HTMLCanvasElement | null>(null);
  let theme: TapeTheme | null = null;

  const fmtR = (r: number): string => `${r >= 0 ? '+' : '−'}${Math.abs(r).toFixed(1)}R`;

  onMount(() => {
    // build off the first paint so the shell renders instantly
    const t = setTimeout(() => {
      const s = buildBossSession(last?.seed ?? DEMO_SEED);
      const r = last
        ? runPolicy(s, (i) => {
            const a = last.actions[i];
            return a === 'fade' || a === 'go-with' ? a : null;
          })
        : fadeBot(s);
      const upTo = Math.min(OTF_HOLD_THROUGH_BAR, s.gen.bars.length - 1);
      tells = measureTells(s.gen.bars, s.gen.labels, upTo, s.gen.script.rowStep, 1, s.normalIbPts);
      session = s;
      run = r;
    }, 10);
    return () => clearTimeout(t);
  });

  function pins(): TapePin[] {
    if (!run) return [];
    return run.trades.map((t) => ({
      bar: t.bar,
      price: t.plan.entry,
      glyph: (t.exit?.rPnl ?? 0) < 0 ? '✗' : '✓',
      label: `${t.action === 'fade' ? 'FADE' : 'GO'} ${t.decisionIndex + 1}`,
      tone: (t.exit?.rPnl ?? 0) < 0 ? 'sell' : 'buy',
    }));
  }

  function draw(): void {
    if (!session || !run || !canvasEl || !bodyEl) return;
    if (!theme) theme = readTapeTheme(bodyEl);
    const w = bodyEl.clientWidth;
    const h = bodyEl.clientHeight;
    if (w < 10 || h < 10) return;
    const bars = session.gen.bars.slice(0, run.endBar + 1);
    devHooks().lastRenderMs = renderTape(canvasEl, bars, w, h, {
      theme,
      nBarsTotal: session.gen.bars.length,
      rowStep: 1, // panel head declares "1 pt rows"
      pVah: session.pVah,
      pins: pins(),
      caption: runCaption(run),
      padT: 38,
      padB: 30,
    });
  }

  $effect(() => {
    void session;
    void run;
    void canvasEl;
    draw();
  });

  $effect(() => {
    if (!bodyEl) return;
    const ro = new ResizeObserver(() => draw());
    ro.observe(bodyEl);
    return () => ro.disconnect();
  });
</script>

<section class="page">
  <!-- top chrome -->
  <div class="topbar">
    <a class="back num" href="#/"><span class="chev">‹</span>Skill tree</a>
    <div class="bosspos num">BOSS 1 / 3</div>
  </div>

  <!-- boss head -->
  <div class="head">
    <div class="overline num"><span class="skull">◆</span>Boss session<span class="rule"></span><span class="gate">GATES T2 → T3</span></div>
    <h1 class="bossname">Trend-Day Fade Gauntlet</h1>
    <p class="expect">Expect <b class="num">~40%</b> success. Errors are the content.</p>
  </div>

  <!-- tape: the last run, or the demo run computed by the same engine -->
  <div class="chart">
    <div class="panelhead num">
      <span class="cap">{last ? 'LAST RUN OF THIS BOSS' : 'DEMO — THE CLASSIC MISTAKE'}</span>
      <span>RTH · 1 pt rows</span>
    </div>
    <div class="chart-body" bind:this={bodyEl}>
      <canvas bind:this={canvasEl}></canvas>
    </div>
  </div>

  <!-- the trap -->
  <div class="trap">
    <div class="seclabel num">The trap</div>
    <p class="trapline">{BOSS1_TRAP_LINE_HEAD}<em>{BOSS1_TRAP_LINE_EM}</em>{BOSS1_TRAP_LINE_TAIL}</p>
    <div class="tells">
      {#if tells}
        <div class="tell num"><span class="dot">●</span><span>IB <b>{tells.ibRatio.toFixed(1)}×</b> normal width</span></div>
        {#if tells.maxPullbackFrac !== null}
          <div class="tell num"><span class="dot">●</span><span>pullbacks never exceed <b>{Math.round(tells.maxPullbackFrac * 100)}%</b></span></div>
        {/if}
        <div class="tell num"><span class="dot">●</span><span>impulse volume <b>{tells.impulseVolRatio.toFixed(1)}× rising</b></span></div>
      {:else}
        <div class="tell num"><span class="dot">●</span><span>measuring the tells…</span></div>
      {/if}
    </div>
  </div>

  <!-- rules of engagement -->
  <div class="roe">
    <span class="seclabel num">Rules of engagement</span>
    <div class="roerow"><span class="k">Daily loss limit</span><span class="v num">−{Math.abs(DAILY_LOSS_LIMIT_R).toFixed(1)}R</span></div>
    <div class="roerow"><span class="k">Trailing drawdown</span><span class="v num">−{TRAILING_DD_R.toFixed(1)}R <span class="u">from peak</span></span></div>
    <div class="roerow"><span class="k">KILL / HOLD interrupts</span><span class="v num">{(KILL_WINDOW_MS / 1000).toFixed(1)}s <span class="u">to answer</span></span></div>
    <div class="roefoot">A breach ends the session — open positions included.</div>
  </div>

  <!-- best attempt -->
  <div class="best">
    <span class="k num">Best attempt</span>
    {#if best}
      <span class="v num">READ <b>{best.readScore}</b> · <b>{fmtR(best.pnlR)}</b> · {best.passed ? 'passed' : 'not passed'}</span>
    {:else}
      <span class="v num">no attempts yet</span>
    {/if}
  </div>

  <!-- CTA -->
  <div class="footer">
    <button class="enter" onclick={onenter}>ENTER THE GAUNTLET<span class="arrow">▸</span></button>
  </div>
</section>

<style>
  .page {
    flex: 1;
    min-height: 0;
    height: calc(100dvh - var(--control-xl));
    max-height: calc(100dvh - var(--control-xl));
    width: 100%;
    max-width: 480px;
    margin: 0 auto;
    padding: 0 var(--s4);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ---------- top chrome ---------- */
  .topbar {
    height: 52px;
    flex: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-top: var(--s3);
  }
  .back {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: var(--text-xs);
    letter-spacing: var(--track-label);
    text-transform: uppercase;
    color: var(--ink-muted);
  }
  .back .chev {
    font-size: var(--text-body-sm);
  }
  .bosspos {
    font-size: var(--text-xs);
    letter-spacing: var(--track-label);
    color: var(--ink-muted);
  }

  /* ---------- boss head ---------- */
  .head {
    flex: none;
    padding: var(--s2) 0 var(--s4);
  }
  .overline {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: var(--text-xs);
    font-weight: var(--weight-semibold);
    letter-spacing: var(--track-overline);
    text-transform: uppercase;
    color: var(--highlight);
  }
  .overline .skull {
    font-size: 9px;
    line-height: 0;
    letter-spacing: 0;
    transform: translateY(-0.5px);
  }
  .overline .rule {
    flex: 1;
    height: var(--hairline-w);
    background: var(--hairline);
  }
  .overline .gate {
    color: var(--ink-muted);
    font-weight: var(--weight-regular);
    letter-spacing: var(--track-label);
  }
  .bossname {
    margin-top: var(--s3);
    font-size: var(--text-display);
    font-weight: var(--weight-bold);
    line-height: var(--leading-tight);
    color: var(--ink-hero);
    letter-spacing: var(--track-display);
    white-space: nowrap;
  }
  .expect {
    margin-top: var(--s2);
    font-size: var(--text-body);
    color: var(--ink-body);
  }
  .expect b {
    font-weight: var(--weight-semibold);
    color: var(--highlight);
  }

  /* ---------- chart panel ---------- */
  .chart {
    flex: 1;
    min-height: 120px;
    background: var(--surface-1);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .panelhead {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    z-index: 2;
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    padding: var(--s3) var(--s3) 0;
    font-size: var(--text-xs);
    letter-spacing: var(--track-caption);
    color: var(--ink-muted);
  }
  .panelhead .cap {
    color: var(--ink-body);
  }
  .chart-body {
    flex: 1;
    min-height: 0;
    position: relative;
  }
  .chart-body canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }

  /* ---------- the trap ---------- */
  .trap {
    flex: none;
    padding: var(--s4) 0 0;
  }
  .seclabel {
    font-size: var(--text-xs);
    font-weight: var(--weight-regular);
    letter-spacing: var(--track-overline);
    text-transform: uppercase;
    color: var(--ink-muted);
  }
  .trapline {
    margin-top: var(--s2);
    font-size: var(--text-body-lg);
    font-weight: var(--weight-medium);
    line-height: 1.4;
    color: var(--ink-body);
  }
  .trapline em {
    font-style: normal;
    color: var(--ink-hero);
  }
  .tells {
    margin-top: var(--s3);
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .tell {
    display: flex;
    align-items: baseline;
    gap: 10px;
    font-size: var(--text-sm);
    color: var(--ink-muted);
  }
  .tell .dot {
    color: var(--highlight);
    font-size: var(--text-2xs);
    transform: translateY(-1px);
  }
  .tell b {
    font-weight: var(--weight-semibold);
    color: var(--ink-body);
  }

  /* ---------- rules of engagement ---------- */
  .roe {
    flex: none;
    margin-top: var(--s4);
    background: var(--surface-1);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
    padding: var(--s3) var(--s4) var(--s4);
  }
  .roe .seclabel {
    display: block;
  }
  .roerow {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--s3);
    padding: var(--s2) 0 0;
  }
  .roerow .k {
    font-size: var(--text-body-sm);
    font-weight: var(--weight-medium);
    color: var(--ink-body);
  }
  .roerow .v {
    font-size: var(--text-body-sm);
    font-weight: var(--weight-semibold);
    color: var(--ink-hero);
    white-space: nowrap;
  }
  .roerow .v .u {
    font-weight: var(--weight-regular);
    color: var(--ink-muted);
  }
  .roefoot {
    margin-top: var(--s3);
    padding-top: 10px;
    border-top: var(--hairline-w) solid var(--hairline);
    font-size: var(--text-sm);
    color: var(--ink-muted);
    white-space: nowrap;
  }

  /* ---------- best attempt ---------- */
  .best {
    flex: none;
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    padding: var(--s4) var(--s1) 0;
  }
  .best .k {
    font-size: var(--text-xs);
    letter-spacing: var(--track-overline);
    text-transform: uppercase;
    color: var(--ink-muted);
    white-space: nowrap;
  }
  .best .v {
    font-size: var(--text-sm);
    color: var(--ink-muted);
    white-space: nowrap;
  }
  .best .v b {
    font-weight: var(--weight-semibold);
    color: var(--ink-body);
  }

  /* ---------- footer CTA ---------- */
  .footer {
    flex: none;
    padding: var(--s4) 0 var(--s4);
  }
  .enter {
    height: var(--control-lg);
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    background: var(--cta);
    color: var(--cta-ink);
    border-radius: var(--radius);
    font-size: var(--text-body);
    font-weight: var(--weight-semibold);
    letter-spacing: var(--track-kicker);
  }
  .enter .arrow {
    font-size: var(--text-sm);
  }
</style>
