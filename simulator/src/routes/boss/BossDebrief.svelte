<script lang="ts" module>
  import type { TrendTells, Decomposition } from '../../lib/boss/engine';

  /** One timeline row of the debrief (already formatted by the run). */
  export interface DebriefEvent {
    clock: string;
    glyph: string;
    tone: 'buy' | 'sell' | 'muted';
    text: string;
    r: string | null;
  }

  /** Everything the debrief displays — computed by the run, never here. */
  export interface DebriefData {
    score: number;
    pnlR: number;
    decomp: Decomposition;
    passed: boolean;
    breach: 'daily' | 'trailing' | null;
    tells: TrendTells;
    timeline: DebriefEvent[];
    fades: number;
    slept: number;
    seed: string;
    paramsVersion: string;
  }
</script>

<script lang="ts">
  /* Boss 1 debrief (GDD §8 screen 7): Read Score beside realized P&L with
     the luck decomposition — outcome ≠ decision quality — then the guide's
     trend-day logic (Part VII.1) quoted with provenance, the measured tells,
     the run timeline, and the "one thing to drill" Woodpecker hook. */
  import { PASS_READ_SCORE } from '../../lib/boss/engine';
  import {
    BOSS1_DEBRIEF_LINE,
    BOSS1_GUIDE_QUOTE,
    BOSS1_GUIDE_REF,
    OUTCOME_LINE,
  } from '../../lib/boss/templates';
  import { queueOneThingToDrill } from '../../lib/boss/store';

  const { data }: { data: DebriefData } = $props();

  let queued = $state(false);

  const fmtR = (r: number): string => `${r >= 0 ? '+' : '−'}${Math.abs(r).toFixed(1)}R`;

  async function queue(): Promise<void> {
    if (queued) return;
    await queueOneThingToDrill(data.seed, data.paramsVersion, Date.now());
    queued = true;
  }
</script>

<section class="page">
  <div class="topbar">
    <a class="back num" href="#/boss"><span class="chev">‹</span>Boss intro</a>
    <div class="bosspos num">DEBRIEF · BOSS 1 / 3</div>
  </div>

  <!-- headline: process beside outcome, luck decomposed -->
  <div class="headline num">
    <span>READ SCORE <b>{data.score}</b></span>
    <span class="sep">·</span>
    <span>P&amp;L <b>{fmtR(data.pnlR)}</b></span>
    <span class="sep">·</span>
    <span>LUCK <b>{fmtR(data.decomp.luckR)}</b></span>
  </div>
  <p class="sentence">{data.decomp.sentence} <em>{data.decomp.tail}</em></p>
  <p class="outcome num">{OUTCOME_LINE}</p>

  <div class="gate num" class:passed={data.passed}>
    {#if data.passed}
      ✓ PASSED — GATE T2 → T3 OPENS
    {:else}
      NOT PASSED — READ ≥ {PASS_READ_SCORE} AND NO BREACH REQUIRED
    {/if}
  </div>

  <div class="scroll">
    <!-- the lesson -->
    <blockquote class="bossline">{BOSS1_DEBRIEF_LINE}</blockquote>
    <blockquote class="quote">
      “{BOSS1_GUIDE_QUOTE}”
      <cite class="num">— {BOSS1_GUIDE_REF}</cite>
    </blockquote>

    <!-- the tells, measured from this session -->
    <div class="tells">
      <div class="seclabel num">The tells on the tape</div>
      <div class="tellgrid num">
        <span class="tk">IB width</span><span class="tv"><b>{data.tells.ibRatio.toFixed(1)}×</b> normal</span>
        <span class="tk">max pullback</span>
        <span class="tv">
          {#if data.tells.maxPullbackFrac !== null}<b>{Math.round(data.tells.maxPullbackFrac * 100)}%</b> of its impulse{:else}none completed{/if}
        </span>
        <span class="tk">impulse volume</span>
        <span class="tv"><b>{data.tells.impulseVolRatio.toFixed(1)}×</b> the {data.tells.impulseVolBasis === 'pullbacks' ? 'pullbacks' : 'balance phase'}</span>
        <span class="tk">one-timeframing</span><span class="tv"><b>{data.tells.otfBrackets}</b> brackets unbroken</span>
      </div>
    </div>

    <!-- run timeline -->
    <div class="timeline">
      <div class="seclabel num">Timeline</div>
      {#each data.timeline as ev, i (i)}
        <div class="evt">
          <span class="clock num">{ev.clock}</span>
          <span
            class="glyph num"
            class:buy={ev.tone === 'buy'}
            class:sell={ev.tone === 'sell'}>{ev.glyph}</span
          >
          <span class="txt">{ev.text}</span>
          {#if ev.r !== null}<span class="r num">{ev.r}</span>{/if}
        </div>
      {:else}
        <div class="evt"><span class="txt">No decisions taken — the tape ran unopposed.</span></div>
      {/each}
      {#if data.slept > 0}
        <div class="slept num">{data.slept} INTERRUPT{data.slept === 1 ? '' : 'S'} SLEPT THROUGH — SURFACED ABOVE</div>
      {/if}
    </div>
  </div>

  <!-- actions -->
  <div class="footer">
    <button class="next drill" onclick={queue} disabled={queued}>
      {queued ? '✓ QUEUED FOR TOMORROW' : 'ONE THING TO DRILL: REGIME GATE'}
    </button>
    <div class="row2">
      <a class="next ghost" href="#/boss">EXIT</a>
      <a class="next" href={`#/boss?stage=run&seed=${String((BigInt(Date.now()) << 20n) ^ BigInt(Math.floor(Math.random() * 2 ** 40)))}`}
        >RE-ENTER<span class="arrow">▸</span></a
      >
    </div>
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
  .topbar {
    height: 44px;
    flex: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-top: var(--s2);
  }
  .back {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: var(--text-xs);
    letter-spacing: var(--track-label);
    text-transform: uppercase;
    color: var(--ink-muted);
  }
  .bosspos {
    font-size: var(--text-2xs);
    letter-spacing: var(--track-label);
    color: var(--ink-muted);
  }

  .headline {
    flex: none;
    padding-top: var(--s3);
    display: flex;
    align-items: baseline;
    gap: 8px;
    font-size: var(--text-body-sm);
    letter-spacing: var(--track-caption);
    color: var(--ink-muted);
    white-space: nowrap;
  }
  .headline b {
    font-size: var(--text-title);
    font-weight: var(--weight-semibold);
    color: var(--ink-hero);
  }
  .headline .sep {
    color: var(--ink-ghost);
  }
  .sentence {
    flex: none;
    padding-top: var(--s2);
    font-size: var(--text-body-lg);
    font-weight: var(--weight-medium);
    color: var(--ink-body);
    text-wrap: balance;
  }
  .sentence em {
    font-style: normal;
    color: var(--ink-hero);
  }
  .outcome {
    flex: none;
    padding-top: var(--s1);
    font-size: var(--text-xs);
    letter-spacing: var(--track-caption);
    color: var(--ink-muted);
  }
  .gate {
    flex: none;
    margin-top: var(--s3);
    padding: var(--s2) var(--s3);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
    font-size: var(--text-xs);
    letter-spacing: var(--track-label);
    color: var(--ink-muted);
  }
  .gate.passed {
    color: var(--buy-green);
    border-color: var(--buy-green);
  }

  .scroll {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    margin-top: var(--s3);
    display: flex;
    flex-direction: column;
    gap: var(--s4);
    padding-bottom: var(--s2);
  }
  .bossline {
    font-size: var(--text-body-lg);
    font-weight: var(--weight-semibold);
    color: var(--ink-hero);
    border-left: 2px solid var(--highlight);
    padding-left: var(--s3);
  }
  .quote {
    font-size: var(--text-body-sm);
    color: var(--ink-body);
    border-left: 2px solid var(--hairline);
    padding-left: var(--s3);
  }
  .quote cite {
    display: block;
    margin-top: var(--s1);
    font-style: normal;
    font-size: var(--text-xs);
    color: var(--ink-muted);
  }

  .seclabel {
    font-size: var(--text-2xs);
    letter-spacing: var(--track-overline);
    text-transform: uppercase;
    color: var(--ink-muted);
  }
  .tells {
    background: var(--surface-1);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
    padding: var(--s3) var(--s4);
  }
  .tellgrid {
    margin-top: var(--s2);
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 6px var(--s4);
    font-size: var(--text-sm);
  }
  .tellgrid .tk {
    color: var(--ink-muted);
  }
  .tellgrid .tv {
    color: var(--ink-muted);
    text-align: right;
  }
  .tellgrid b {
    color: var(--ink-body);
    font-weight: var(--weight-semibold);
  }

  .timeline {
    display: flex;
    flex-direction: column;
    gap: var(--s2);
  }
  .evt {
    display: flex;
    align-items: baseline;
    gap: var(--s2);
    font-size: var(--text-sm);
    color: var(--ink-body);
  }
  .evt .clock {
    color: var(--ink-muted);
    flex: none;
  }
  .evt .glyph {
    font-weight: var(--weight-bold);
    flex: none;
    color: var(--ink-muted);
  }
  .evt .glyph.buy {
    color: var(--buy-green);
  }
  .evt .glyph.sell {
    color: var(--sell-red);
  }
  .evt .txt {
    flex: 1;
  }
  .evt .r {
    color: var(--ink-muted);
    white-space: nowrap;
  }
  .slept {
    font-size: var(--text-2xs);
    letter-spacing: var(--track-label);
    color: var(--highlight);
  }

  .footer {
    flex: none;
    padding: var(--s3) 0 var(--s4);
    display: flex;
    flex-direction: column;
    gap: var(--s2);
  }
  .next {
    width: 100%;
    height: var(--control-lg);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    background: var(--surface-2);
    color: var(--ink-hero);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
    font-size: var(--text-sm);
    font-weight: var(--weight-semibold);
    letter-spacing: var(--track-kicker);
  }
  .next:disabled {
    color: var(--ink-muted);
  }
  .next.drill {
    background: var(--cta);
    color: var(--cta-ink);
    border: none;
  }
  .next.drill:disabled {
    background: var(--surface-2);
    color: var(--ink-muted);
  }
  .row2 {
    display: flex;
    gap: var(--s2);
  }
  .row2 .next {
    flex: 1;
  }
  .next.ghost {
    background: transparent;
    color: var(--ink-muted);
  }
  .next .arrow {
    font-size: var(--text-sm);
    color: var(--ink-muted);
  }
</style>
