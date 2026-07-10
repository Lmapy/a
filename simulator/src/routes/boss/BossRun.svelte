<script lang="ts">
  /* Boss 1 — the gauntlet run (GDD §6 Boss 1, §8 screen 6 adapted portrait).

     Flow: 30-min-bracket playback of a generated trend day on the canvas
     tape (price path + developing 1-pt profile). At the scripted decision
     bars (impulse ends) playback halts and the order ticket arms: the
     declared read (balance/imbalance) is REQUIRED before any action exists;
     actions are FADE VAH / GO WITH / STAND ASIDE with engine-computed
     structural stop/target previews. KILL/HOLD reflex interrupts fire
     mid-position (5s window, <2s for full credit). The prop-desk limits
     (daily −2.5R, trailing −3.0R from realized peak) end the run on breach,
     open positions included. Every decision is graded on setup legality vs
     the session labels — a fade on this labeled trend day is the cardinal
     error (3× weight + forced micro-replay of the tells).

     Dev/e2e params (#/boss?stage=run&seed=<dec>&snap=…):
       snap=decision1    → jump to the first armed order ticket
       snap=verdict-fade → commit FADE·balance at decision 1 (cardinal verdict)
       snap=interrupt    → GO WITH at decision 1, jump to the KILL/HOLD moment
       snap=debrief      → fade every decision, sleep interrupts, to debrief */
  import { onMount } from 'svelte';
  import type { Bar, Verdict } from '../../lib/types';
  import {
    DAILY_LOSS_LIMIT_R,
    INTERRUPT_DELAY_BARS,
    KILL_WINDOW_MS,
    OTF_HOLD_THROUGH_BAR,
    PASS_READ_SCORE,
    TRAILING_DD_R,
    barClock,
    buildBossSession,
    closeAtEnd,
    decompose,
    enterPosition,
    gradeDecision,
    gradeReflex,
    killPosition,
    measureTells,
    newEquity,
    planFade,
    planGoWith,
    readScore,
    reflexTruth,
    stepBar,
  } from '../../lib/boss/engine';
  import type {
    BossAction,
    BossSession,
    DeclaredRead,
    EquityState,
    GradedEvent,
    ReflexAnswer,
    TicketPlan,
    TradeExit,
    TrendTells,
  } from '../../lib/boss/engine';
  import { bossTemplate, renderBossTemplate } from '../../lib/boss/templates';
  import {
    decisionRecordOf,
    persistRun,
    reflexRecordOf,
    saveRun,
  } from '../../lib/boss/store';
  import type { BossRunSummary } from '../../lib/boss/store';
  import { renderTape, readTapeTheme } from '../../lib/boss/render';
  import type { TapeLevel, TapePin, TapeTheme } from '../../lib/boss/render';
  import { devHooks } from '../../lib/render/devhooks';
  import { BRACKET_LETTERS, bracketOf } from '../../lib/core/tpo';
  import BossDebrief from './BossDebrief.svelte';
  import type { DebriefEvent, DebriefData } from './BossDebrief.svelte';

  const { seed, snap = null }: { seed: string; snap?: string | null } = $props();

  /* ---------------- run state ---------------- */

  type Phase = 'loading' | 'playing' | 'paused' | 'decision' | 'verdict' | 'interrupt' | 'breach' | 'debrief';

  interface DecisionRow {
    decisionIndex: number; // index into rows (exit attribution)
    ordinal: number; // 1-based decision point number
    bar: number;
    action: BossAction;
    read: DeclaredRead;
    latencyMs: number;
    verdict: Verdict;
    plan: TicketPlan | null;
    exit: TradeExit | null;
  }

  interface ReflexRow {
    bar: number;
    answer: ReflexAnswer | null;
    latencyMs: number;
    verdict: Verdict;
  }

  let session = $state<BossSession | null>(null);
  let phase = $state<Phase>('loading');
  let revealBar = $state(1);
  let eq = $state<EquityState>(newEquity());
  let rows = $state<DecisionRow[]>([]);
  let reflexRows = $state<ReflexRow[]>([]);
  let events = $state<GradedEvent[]>([]);
  let actionsTaken = $state<(BossAction | 'skip')[]>([]);
  let nextDecision = $state(0);
  let feed = $state('tape rolling — decisions come to you');
  let debrief = $state<DebriefData | null>(null);

  // decision-ticket state
  let read = $state<DeclaredRead | ''>('');
  let armedAt = 0;
  // verdict state
  let verdict = $state<Verdict | null>(null);
  let verdictKind = $state<'decision' | 'reflex' | null>(null);
  // forced micro-replay (cardinal)
  let replayActive = $state(false);
  let replayDone = $state(true);
  let replayBar = $state(2);
  let verdictTells = $state<TrendTells | null>(null);
  // interrupt state
  let interruptBar: number | null = null;
  let interruptDeadline = 0;
  let countdown = $state(1);
  let interruptFrozen = false; // snap screenshots freeze the timer
  // breach note
  let breachText = $state('');

  const fmtR = (r: number): string => `${r >= 0 ? '+' : '−'}${Math.abs(r).toFixed(1)}R`;

  const bars = (): Bar[] => session!.gen.bars;
  const rowStep = (): number => session!.gen.script.rowStep;

  const liveScore = $derived(events.length > 0 ? readScore(events) : null);
  const dayFill = $derived(Math.min(1, Math.max(0, -eq.equityR / Math.abs(DAILY_LOSS_LIMIT_R))));
  const trailFill = $derived(Math.min(1, Math.max(0, (eq.peakR - eq.equityR) / TRAILING_DD_R)));

  /* ---------------- playback ---------------- */

  /** Bars advanced per second of wall clock (≈0.67s per 30-min bracket). */
  const BARS_PER_SEC = 45;
  let raf = 0;
  let lastTick = 0;
  let carry = 0;

  function startPlayback(): void {
    phase = 'playing';
    lastTick = performance.now();
    carry = 0;
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(tick);
  }

  function tick(now: number): void {
    if (phase !== 'playing' || !session) return;
    carry += ((now - lastTick) / 1000) * BARS_PER_SEC;
    lastTick = now;
    let n = Math.floor(carry);
    carry -= n;
    while (n-- > 0) {
      if (!advanceOneBar()) return;
    }
    raf = requestAnimationFrame(tick);
  }

  /** Step one bar through the engine reducer. False = playback halted. */
  function advanceOneBar(): boolean {
    if (!session) return false;
    const t = revealBar + 1;
    if (t >= bars().length) {
      finishRun();
      return false;
    }
    stepBar(eq, bars()[t]);
    revealBar = t;
    if (eq.lastExit) onExit(eq.lastExit);
    if (eq.breach) {
      onBreach();
      return false;
    }
    if (interruptBar !== null && t === interruptBar) {
      if (eq.open) {
        armInterrupt();
        return false;
      }
      interruptBar = null; // position already closed — no holder, no interrupt
    }
    const di = session.decisionBars.indexOf(t);
    if (di >= 0 && di === nextDecision) {
      if (eq.open) {
        // the moment passes while a position is still working
        actionsTaken = [...actionsTaken, 'skip'];
        nextDecision += 1;
        feed = `${barClock(t)} · decision ${di + 1} passed — position still open`;
      } else {
        armDecision();
        return false;
      }
    }
    if (t === bars().length - 1) {
      finishRun();
      return false;
    }
    return true;
  }

  function pause(): void {
    if (phase === 'playing') {
      cancelAnimationFrame(raf);
      phase = 'paused';
    }
  }

  /* ---------------- decisions ---------------- */

  function armDecision(): void {
    read = '';
    phase = 'decision';
    armedAt = performance.now();
  }

  const fadePlan = $derived(
    session && phase === 'decision' ? planFade(bars(), revealBar, rowStep()) : null,
  );
  const goPlan = $derived(
    session && phase === 'decision' ? planGoWith(bars(), revealBar, rowStep()) : null,
  );

  function commitDecision(action: BossAction, latencyOverride?: number): void {
    if (!session || phase !== 'decision' || read === '') return;
    const bar = revealBar;
    const latencyMs = Math.max(0, Math.round(latencyOverride ?? performance.now() - armedAt));
    const plan = action === 'fade' ? planFade(bars(), bar, rowStep()) : action === 'go-with' ? planGoWith(bars(), bar, rowStep()) : null;
    const grade = gradeDecision(action, read, session.gen.labels, bar, plan, session.trendDir);
    const row: DecisionRow = {
      decisionIndex: rows.length,
      ordinal: nextDecision + 1,
      bar,
      action,
      read,
      latencyMs,
      verdict: grade.verdict,
      plan,
      exit: null,
    };
    rows = [...rows, row];
    events = [
      ...events,
      {
        kind: 'decision',
        score: grade.score,
        weight: grade.weight,
        legal: grade.legal,
        readCorrect: grade.readCorrect,
        cardinal: grade.cardinal,
        rPnl: null,
      },
    ];
    actionsTaken = [...actionsTaken, action];
    nextDecision += 1;
    if (action !== 'stand-aside' && plan) {
      enterPosition(eq, action, plan, bar, row.decisionIndex);
      interruptBar = bar + INTERRUPT_DELAY_BARS;
    }
    verdict = grade.verdict;
    verdictKind = 'decision';
    feed = `${barClock(bar)} · ${grade.verdict.correct ? '✓' : grade.cardinal ? '✗✗' : '✗'} ${action.toUpperCase()} · read ${read}`;
    phase = 'verdict';
    if (grade.cardinal) {
      verdictTells = measureTells(bars(), session.gen.labels, bar, rowStep(), session.trendDir, session.normalIbPts);
      startForcedReplay(bar);
    } else {
      verdictTells = null;
      replayActive = false;
      replayDone = true;
    }
  }

  /* ---------------- forced micro-replay (cardinal) ---------------- */

  let replayRaf = 0;
  function startForcedReplay(uptoBar: number): void {
    replayActive = true;
    replayDone = false;
    const durMs = 1600;
    const from = 2;
    const t0 = performance.now();
    cancelAnimationFrame(replayRaf);
    const step = (): void => {
      const f = Math.min(1, (performance.now() - t0) / durMs);
      replayBar = Math.max(2, Math.floor(from + (uptoBar - from) * f));
      if (f < 1) {
        replayRaf = requestAnimationFrame(step);
      } else {
        replayActive = false;
        replayDone = true;
      }
    };
    replayRaf = requestAnimationFrame(step);
  }

  function skipReplayInstantly(): void {
    cancelAnimationFrame(replayRaf);
    replayActive = false;
    replayDone = true;
  }

  /* ---------------- interrupts ---------------- */

  let countdownRaf = 0;
  function armInterrupt(): void {
    phase = 'interrupt';
    interruptDeadline = performance.now() + KILL_WINDOW_MS;
    countdown = 1;
    if (interruptFrozen) {
      countdown = 0.65;
      return;
    }
    cancelAnimationFrame(countdownRaf);
    const step = (): void => {
      if (phase !== 'interrupt') return;
      const left = interruptDeadline - performance.now();
      countdown = Math.max(0, left / KILL_WINDOW_MS);
      if (left <= 0) {
        resolveReflex(null, KILL_WINDOW_MS);
        return;
      }
      countdownRaf = requestAnimationFrame(step);
    };
    countdownRaf = requestAnimationFrame(step);
  }

  function resolveReflex(answer: ReflexAnswer | null, latencyOverride?: number): void {
    if (!session || phase !== 'interrupt' || !eq.open) return;
    cancelAnimationFrame(countdownRaf);
    const latencyMs = Math.max(
      0,
      Math.round(latencyOverride ?? KILL_WINDOW_MS - (interruptDeadline - performance.now())),
    );
    const truth = reflexTruth(eq.open, session.trendDir);
    const grade = gradeReflex(truth, answer, latencyMs);
    reflexRows = [...reflexRows, { bar: revealBar, answer, latencyMs, verdict: grade.verdict }];
    events = [
      ...events,
      {
        kind: 'reflex',
        score: grade.score,
        weight: grade.weight,
        legal: true,
        readCorrect: grade.correct,
        cardinal: false,
        rPnl: null,
      },
    ];
    if (answer === 'KILL') {
      killPosition(eq, bars()[revealBar]);
      if (eq.lastExit) onExit(eq.lastExit);
    }
    interruptBar = null;
    verdict = grade.verdict;
    verdictKind = 'reflex';
    verdictTells = null;
    replayActive = false;
    replayDone = true;
    feed = `${barClock(revealBar)} · ${grade.slept ? 'slept through' : (answer ?? '')} ${grade.correct ? '✓' : '✗'} in ${(latencyMs / 1000).toFixed(1)}s`;
    phase = 'verdict';
  }

  /* ---------------- exits / breach / finish ---------------- */

  function onExit(exit: TradeExit): void {
    const row = rows[exit.decisionIndex];
    if (!row) return;
    row.exit = exit;
    rows = [...rows];
    const ev = events.filter((e) => e.kind === 'decision')[exit.decisionIndex];
    if (ev) {
      ev.rPnl = exit.rPnl;
      events = [...events];
    }
    feed = `${barClock(exit.bar)} · ${row.action.toUpperCase()} ${exit.outcome} ${fmtR(exit.rPnl)}`;
  }

  function onBreach(): void {
    if (!eq.breach) return;
    const id = eq.breach === 'daily' ? 'boss1.breach.daily' : 'boss1.breach.trailing';
    breachText = renderBossTemplate(bossTemplate(id), {
      clock: barClock(eq.breachBar ?? revealBar),
      dd: (eq.peakR - eq.equityR).toFixed(1),
    });
    phase = 'breach';
  }

  function continueFromVerdict(): void {
    if (phase !== 'verdict' || !replayDone) return;
    verdict = null;
    verdictKind = null;
    startPlayback();
  }

  function finishRun(): void {
    if (!session || phase === 'debrief') return;
    if (!eq.breach) {
      closeAtEnd(eq, bars()[revealBar]);
      if (eq.lastExit) onExit(eq.lastExit);
    }
    const score = readScore(events);
    const pnlR = eq.realizedR;
    const passed = score >= PASS_READ_SCORE && eq.breach === null;
    const upTo = Math.min(OTF_HOLD_THROUGH_BAR, bars().length - 1);
    const tells = measureTells(bars(), session.gen.labels, upTo, rowStep(), session.trendDir, session.normalIbPts);

    const timeline: DebriefEvent[] = [];
    for (const r of rows) {
      timeline.push({
        clock: barClock(r.bar),
        glyph: r.verdict.cardinal ? '✗✗' : r.verdict.correct ? '✓' : r.action === 'stand-aside' ? '—' : '✗',
        tone: r.verdict.cardinal ? 'sell' : r.verdict.correct ? 'buy' : 'muted',
        text: `${labelOfAction(r.action)} · read ${r.read}`,
        r: r.exit ? fmtR(r.exit.rPnl) : null,
      });
    }
    for (const r of reflexRows) {
      timeline.push({
        clock: barClock(r.bar),
        glyph: r.verdict.correct ? '✓' : '✗',
        tone: r.verdict.correct ? 'buy' : 'sell',
        text: r.answer === null ? 'KILL/HOLD slept through' : `${r.answer} in ${(r.latencyMs / 1000).toFixed(1)}s`,
        r: null,
      });
    }
    timeline.sort((a, b) => a.clock.localeCompare(b.clock));
    if (eq.breach) {
      timeline.push({
        clock: barClock(eq.breachBar ?? revealBar),
        glyph: '⛔',
        tone: 'sell',
        text: eq.breach === 'daily' ? 'daily loss limit breached — session over' : 'trailing drawdown breached — session over',
        r: null,
      });
    }

    debrief = {
      score,
      pnlR,
      decomp: decompose(events, pnlR, score),
      passed,
      breach: eq.breach,
      tells,
      timeline,
      fades: rows.filter((r) => r.action === 'fade').length,
      slept: reflexRows.filter((r) => r.answer === null).length,
      seed: session.seed,
      paramsVersion: session.gen.script.paramsVersion,
    };
    phase = 'debrief';

    // persist: ledger rows (trades carry realized R for the Stats histogram)
    const at = Date.now();
    const ledger = [
      ...rows.map((r) =>
        decisionRecordOf({
          seed: session!.seed,
          paramsVersion: session!.gen.script.paramsVersion,
          decisionIndex: r.decisionIndex,
          bar: r.bar,
          action: r.action,
          read: r.read,
          latencyMs: r.latencyMs,
          verdict: r.verdict,
          rPnl: r.exit?.rPnl ?? null,
          at: at + r.decisionIndex,
        }),
      ),
      ...reflexRows.map((r) =>
        reflexRecordOf({
          seed: session!.seed,
          paramsVersion: session!.gen.script.paramsVersion,
          bar: r.bar,
          answer: r.answer,
          latencyMs: r.latencyMs,
          verdict: r.verdict,
          at: at + 100 + r.bar,
        }),
      ),
    ];
    void persistRun(ledger);
    const summary: BossRunSummary = {
      at,
      seed: session.seed,
      paramsVersion: session.gen.script.paramsVersion,
      readScore: score,
      pnlR,
      passed,
      breach: eq.breach,
      actions: actionsTaken,
    };
    saveRun(summary);
  }

  function labelOfAction(a: BossAction): string {
    return a === 'fade' ? 'FADE VAH' : a === 'go-with' ? 'GO WITH' : 'STAND ASIDE';
  }

  /* ---------------- chart ---------------- */

  let bodyEl = $state<HTMLDivElement | null>(null);
  let canvasEl = $state<HTMLCanvasElement | null>(null);
  let theme: TapeTheme | null = null;

  function pins(): TapePin[] {
    return rows
      .filter((r) => r.action !== 'stand-aside')
      .map((r) => ({
        bar: r.bar,
        price: r.plan?.entry ?? 0,
        glyph: r.exit ? (r.exit.rPnl < 0 ? '✗' : '✓') : r.action === 'fade' ? '▼' : '▲',
        label: `${r.action === 'fade' ? 'FADE' : 'GO'} ${r.ordinal}`,
        tone: r.exit ? (r.exit.rPnl < 0 ? 'sell' : 'buy') : 'ghost',
      }));
  }

  function levels(): TapeLevel[] {
    const pos = eq.open;
    if (!pos) return [];
    return [
      { price: pos.plan.stop, label: 'STOP', tone: 'sell', labelBelow: pos.dir === 1 },
      { price: pos.plan.target, label: 'TGT', tone: 'buy', labelBelow: pos.dir === -1 },
      { price: pos.plan.entry, label: 'IN', tone: 'muted', labelBelow: pos.dir === -1 },
    ];
  }

  function draw(): void {
    if (!session || !canvasEl || !bodyEl) return;
    if (!theme) theme = readTapeTheme(bodyEl);
    const w = bodyEl.clientWidth;
    const h = bodyEl.clientHeight;
    if (w < 10 || h < 10) return;
    const upTo = replayActive || (phase === 'verdict' && !replayDone) ? replayBar : revealBar;
    const slice = bars().slice(0, upTo + 1);
    const pos = eq.open;
    devHooks().lastRenderMs = renderTape(canvasEl, slice, w, h, {
      theme,
      nBarsTotal: bars().length,
      rowStep: 1,
      pVah: session.pVah,
      pins: replayActive ? [] : pins(),
      levels: replayActive ? [] : levels(),
      includePrices: pos ? [pos.plan.stop, pos.plan.target] : [],
      caption: null,
      padT: 30,
      padB: 12,
    });
  }

  $effect(() => {
    void revealBar;
    void replayBar;
    void phase;
    void rows;
    void canvasEl;
    draw();
  });

  $effect(() => {
    if (!bodyEl) return;
    const ro = new ResizeObserver(() => draw());
    ro.observe(bodyEl);
    return () => ro.disconnect();
  });

  /* ---------------- boot + dev snaps ---------------- */

  function fastForwardTo(pred: () => boolean): void {
    // deterministic, animation-free advance (dev/e2e snaps)
    let guard = 500;
    while (!pred() && guard-- > 0) {
      if (phase === 'playing' || phase === 'paused') {
        if (!advanceOneBar()) continue;
      } else {
        break;
      }
    }
  }

  function applySnap(kind: string): void {
    interruptFrozen = kind === 'interrupt';
    phase = 'paused';
    if (kind === 'decision1') {
      fastForwardTo(() => phase === 'decision');
      return;
    }
    if (kind === 'verdict-fade') {
      fastForwardTo(() => phase === 'decision');
      read = 'balance';
      commitDecision('fade', 2400);
      skipReplayInstantly();
      return;
    }
    if (kind === 'interrupt') {
      fastForwardTo(() => phase === 'decision');
      read = 'imbalance';
      commitDecision('go-with', 1500);
      continueFromVerdictSilently();
      fastForwardTo(() => phase === 'interrupt');
      return;
    }
    if (kind === 'debrief') {
      // TS narrows the $state local across these mutating calls — read it
      // through a function so the loop sees the live phase.
      const ph = (): Phase => phase;
      let guard = 12;
      while (ph() !== 'debrief' && guard-- > 0) {
        fastForwardTo(() => ph() !== 'playing' && ph() !== 'paused');
        if (ph() === 'decision') {
          read = 'balance';
          commitDecision('fade', 2400);
          skipReplayInstantly();
          continueFromVerdictSilently();
        } else if (ph() === 'interrupt') {
          resolveReflex(null, KILL_WINDOW_MS);
          continueFromVerdictSilently();
        } else if (ph() === 'verdict') {
          continueFromVerdictSilently();
        } else if (ph() === 'breach') {
          finishRun();
        }
      }
      return;
    }
  }

  function continueFromVerdictSilently(): void {
    // snap helper: leave verdict without starting the rAF playback loop
    if (phase !== 'verdict') return;
    verdict = null;
    verdictKind = null;
    phase = 'paused';
  }

  onMount(() => {
    const t = setTimeout(() => {
      session = buildBossSession(seed);
      // step bar 1 so the tape has two points
      stepBar(eq, bars()[1]);
      revealBar = 1;
      if (snap) {
        applySnap(snap);
      } else {
        startPlayback();
      }
    }, 10);
    return () => {
      clearTimeout(t);
      cancelAnimationFrame(raf);
      cancelAnimationFrame(countdownRaf);
      cancelAnimationFrame(replayRaf);
    };
  });

  const shortOrLong = $derived(eq.open ? (eq.open.dir === -1 ? 'SHORT' : 'LONG') : '');
</script>

{#if phase === 'debrief' && debrief}
  <BossDebrief data={debrief} />
{:else}
  <section class="page">
    <!-- top chrome -->
    <div class="topbar">
      <a class="back num" href="#/boss"><span class="chev">‹</span>Abandon</a>
      <div class="bosspos num">BOSS 1 / 3 · TREND-DAY FADE GAUNTLET</div>
    </div>

    <!-- desk row: live process score, marked P&L, limit meters -->
    <div class="deskrow">
      <div class="stat">
        <span class="k num">READ</span>
        <span class="v num">{liveScore === null ? '—' : liveScore}</span>
      </div>
      <div class="stat">
        <span class="k num">P&amp;L</span>
        <span class="v num">{fmtR(eq.equityR)}</span>
      </div>
      <div class="meters num">
        <div class="meter">
          <span class="mk">day {Math.abs(DAILY_LOSS_LIMIT_R).toFixed(1)}R</span>
          <div class="bar"><div class="fill" style={`width:${(dayFill * 100).toFixed(0)}%`}></div></div>
        </div>
        <div class="meter">
          <span class="mk">trail {TRAILING_DD_R.toFixed(1)}R</span>
          <div class="bar"><div class="fill" style={`width:${(trailFill * 100).toFixed(0)}%`}></div></div>
        </div>
      </div>
    </div>

    <!-- tape -->
    <div class="chart">
      <div class="panelhead num">
        <span class="cap">
          {#if session}
            BRACKET {BRACKET_LETTERS[bracketOf(revealBar)]} · {barClock(revealBar)}
          {:else}
            GENERATING SESSION…
          {/if}
        </span>
        <span>RTH · 1 pt rows</span>
      </div>
      <div class="chart-body" bind:this={bodyEl}>
        <canvas bind:this={canvasEl}></canvas>
      </div>
    </div>

    <!-- bottom zone -->
    <div class="zone">
      {#if phase === 'decision' && session}
        <div class="ticket">
          <div class="seclabel num">ORDER TICKET · DECISION {nextDecision + 1}/{session.decisionBars.length} · {barClock(revealBar)}</div>
          <div class="readrow">
            <label class="k" for="declared-read">Declared read</label>
            <select id="declared-read" class="readsel num" bind:value={read}>
              <option value="" disabled>— required —</option>
              <option value="balance">balance</option>
              <option value="imbalance">imbalance</option>
            </select>
          </div>
          {#if fadePlan && goPlan}
            <div class="plans num">
              <span>FADE&nbsp; in {fadePlan.entry.toFixed(2)} · stop {fadePlan.stop.toFixed(2)} · tgt {fadePlan.target.toFixed(2)}</span>
              <span>GO&nbsp;&nbsp;&nbsp; in {goPlan.entry.toFixed(2)} · stop {goPlan.stop.toFixed(2)} · tgt {goPlan.target.toFixed(2)}</span>
            </div>
          {/if}
          <div class="answers">
            <button class="chip" disabled={read === ''} onpointerdown={() => commitDecision('fade')} onclick={() => commitDecision('fade')}>
              <span class="letter num">FADE VAH</span>
            </button>
            <button class="chip" disabled={read === ''} onpointerdown={() => commitDecision('go-with')} onclick={() => commitDecision('go-with')}>
              <span class="letter num">GO WITH</span>
            </button>
            <button class="chip" disabled={read === ''} onpointerdown={() => commitDecision('stand-aside')} onclick={() => commitDecision('stand-aside')}>
              <span class="letter num">STAND ASIDE</span>
            </button>
          </div>
          {#if read === ''}
            <div class="gatehint num">DECLARE THE READ TO UNLOCK THE TICKET</div>
          {/if}
        </div>
      {:else if phase === 'interrupt' && eq.open}
        <div class="interrupt">
          <div class="seclabel num">REFLEX · {shortOrLong} {fmtR(eq.equityR - eq.realizedR)} OPEN · {(KILL_WINDOW_MS / 1000).toFixed(0)}s</div>
          <div class="cdbar"><div class="cdfill" style={`width:${(countdown * 100).toFixed(1)}%`}></div></div>
          <div class="answers">
            <button class="chip kill" onpointerdown={() => resolveReflex('KILL')} onclick={() => resolveReflex('KILL')}>
              <span class="letter num">KILL</span>
            </button>
            <button class="chip" onpointerdown={() => resolveReflex('HOLD')} onclick={() => resolveReflex('HOLD')}>
              <span class="letter num">HOLD</span>
            </button>
          </div>
        </div>
      {:else if phase === 'verdict' && verdict}
        <div class="verdict">
          <div class="verdict-line">
            <span class="glyph num" class:ok={verdict.correct}>{verdict.cardinal ? '✗✗' : verdict.correct ? '✓' : '✗'}</span>
            <p class="explain">{verdict.explanation.replace(/^[✗✓—]+ ?/u, '')}</p>
          </div>
          {#if verdict.cardinal && verdictTells}
            <div class="tellsbox">
              <div class="seclabel num">{replayDone ? 'THE TELLS OVERRIDDEN' : 'REPLAYING THE TELLS…'}</div>
              <div class="tellrow num"><span class="dot">●</span>IB <b>{verdictTells.ibRatio.toFixed(1)}×</b> normal width</div>
              {#if verdictTells.maxPullbackFrac !== null}
                <div class="tellrow num"><span class="dot">●</span>pullbacks ≤ <b>{Math.round(verdictTells.maxPullbackFrac * 100)}%</b> of each impulse</div>
              {/if}
              {#if verdictTells.impulseVolBasis === 'pullbacks'}
                <div class="tellrow num"><span class="dot">●</span>impulse volume <b>{verdictTells.impulseVolRatio.toFixed(1)}×</b> the pullbacks</div>
              {/if}
              <div class="tellrow num"><span class="dot">●</span>one-timeframing <b>{verdictTells.otfBrackets}</b> brackets</div>
            </div>
          {/if}
        </div>
      {:else if phase === 'breach'}
        <div class="verdict">
          <div class="verdict-line">
            <span class="glyph num">⛔</span>
            <p class="explain">{breachText}</p>
          </div>
        </div>
      {:else}
        <div class="feedline num">{session ? feed : 'compiling the trend script…'}</div>
      {/if}
    </div>

    <!-- footer -->
    <div class="footer">
      {#if phase === 'verdict'}
        <button class="next" disabled={!replayDone} onclick={continueFromVerdict}
          >{replayDone ? 'CONTINUE' : 'WATCH THE TELLS…'}<span class="arrow">▸</span></button
        >
      {:else if phase === 'breach'}
        <button class="next" onclick={finishRun}>TO THE DEBRIEF<span class="arrow">▸</span></button>
      {:else if phase === 'playing'}
        <button class="next ghost" onclick={pause}>PAUSE ⏸</button>
      {:else if phase === 'paused' && !snap}
        <button class="next" onclick={startPlayback}>RESUME ▸</button>
      {:else if phase === 'paused' && snap}
        <button class="next" onclick={startPlayback}>RESUME ▸</button>
      {/if}
    </div>
  </section>
{/if}

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

  /* ---------- desk row ---------- */
  .deskrow {
    flex: none;
    display: flex;
    align-items: center;
    gap: var(--s4);
    padding: var(--s2) 0 var(--s3);
  }
  .stat {
    display: flex;
    align-items: baseline;
    gap: 6px;
  }
  .stat .k {
    font-size: var(--text-2xs);
    letter-spacing: var(--track-label);
    color: var(--ink-muted);
  }
  .stat .v {
    font-size: var(--text-verdict);
    font-weight: var(--weight-semibold);
    color: var(--ink-hero);
  }
  .meters {
    margin-left: auto;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .meter {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .meter .mk {
    font-size: 9px;
    letter-spacing: var(--track-caption);
    color: var(--ink-muted);
    width: 64px;
    text-align: right;
  }
  .meter .bar {
    width: 56px;
    height: 4px;
    border-radius: var(--radius-xs);
    background: var(--surface-2);
    overflow: hidden;
  }
  .meter .fill {
    height: 100%;
    background: var(--ink-muted);
  }

  /* ---------- chart ---------- */
  .chart {
    flex: 1;
    min-height: 140px;
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
    padding: var(--s2) var(--s3) 0;
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

  /* ---------- bottom zone ---------- */
  .zone {
    flex: none;
    min-height: 178px;
    padding: var(--s3) 0 0;
  }
  .feedline {
    font-size: var(--text-sm);
    color: var(--ink-muted);
    letter-spacing: var(--track-meta);
    padding-top: var(--s2);
  }
  .seclabel {
    font-size: var(--text-2xs);
    letter-spacing: var(--track-overline);
    text-transform: uppercase;
    color: var(--ink-muted);
  }

  /* ticket */
  .readrow {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--s3);
    padding: var(--s2) 0;
  }
  .readrow .k {
    font-size: var(--text-body-sm);
    font-weight: var(--weight-medium);
    color: var(--ink-body);
  }
  .readsel {
    height: 36px;
    min-width: 148px;
    background: var(--surface-2);
    color: var(--ink-hero);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
    padding: 0 var(--s2);
    font-size: var(--text-sm);
  }
  .plans {
    display: flex;
    flex-direction: column;
    gap: 2px;
    font-size: var(--text-2xs);
    color: var(--ink-muted);
    letter-spacing: var(--track-meta);
    padding-bottom: var(--s2);
    white-space: pre;
  }
  .answers {
    display: flex;
    gap: var(--s2);
    padding-top: var(--s1);
  }
  .chip {
    flex: 1;
    height: var(--control-xl);
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--surface-2);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
    padding: 0 2px;
  }
  .chip:disabled {
    opacity: 0.4;
  }
  .chip .letter {
    font-size: var(--text-xs);
    font-weight: var(--weight-semibold);
    letter-spacing: var(--track-caption);
    color: var(--ink-body);
    white-space: normal;
    text-align: center;
  }
  .chip.kill .letter {
    color: var(--ink-hero);
  }
  .gatehint {
    padding-top: var(--s2);
    font-size: var(--text-2xs);
    letter-spacing: var(--track-label);
    color: var(--ink-ghost);
  }

  /* interrupt */
  .interrupt .cdbar {
    margin-top: var(--s2);
    height: 4px;
    border-radius: var(--radius-xs);
    background: var(--surface-2);
    overflow: hidden;
  }
  .interrupt .cdfill {
    height: 100%;
    background: var(--highlight);
  }
  .interrupt .answers {
    padding-top: var(--s3);
  }

  /* verdict */
  .verdict-line {
    display: flex;
    align-items: baseline;
    gap: 8px;
  }
  .verdict-line .glyph {
    color: var(--sell-red);
    font-weight: var(--weight-bold);
    font-size: var(--text-verdict);
    flex: none;
  }
  .verdict-line .glyph.ok {
    color: var(--buy-green);
  }
  .explain {
    font-size: var(--text-body);
    color: var(--ink-body);
    max-width: 60ch;
  }
  .tellsbox {
    margin-top: var(--s3);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
    background: var(--surface-1);
    padding: var(--s2) var(--s3) var(--s3);
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .tellrow {
    display: flex;
    align-items: baseline;
    gap: 8px;
    font-size: var(--text-sm);
    color: var(--ink-muted);
  }
  .tellrow .dot {
    color: var(--highlight);
    font-size: var(--text-2xs);
  }
  .tellrow b {
    color: var(--ink-body);
    font-weight: var(--weight-semibold);
  }

  /* ---------- footer ---------- */
  .footer {
    flex: none;
    padding: var(--s3) 0 var(--s4);
    min-height: calc(var(--control-lg) + var(--s3) + var(--s4));
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
    font-size: var(--text-body);
    font-weight: var(--weight-semibold);
    letter-spacing: var(--track-kicker);
  }
  .next:disabled {
    color: var(--ink-muted);
    opacity: 0.7;
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
