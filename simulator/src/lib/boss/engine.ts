/* ============================================================================
   @boss/engine — Boss 1 "Trend-Day Fade Gauntlet" (GDD §6): pure simulation +
   grading logic. Everything the boss UI displays is computed here from the
   generated session and its labels — nothing hand-placed (GDD §9).

   Pieces:
   · buildBossSession — a seeded, verified trend day (dir up, one-timeframing
     never breaks, poor high overhead) via the standard gen pipeline.
   · decision points — scripted: the end bar of each impulse leg in the
     session script (the "textbook fade" moments).
   · order-ticket plans — structural stop/target prices computed from the
     developing profile and swing structure (guide §4.11: structural always).
   · equity reducer — per-bar mark-to-market in R with the prop-desk limits:
     daily loss −2.5R, trailing drawdown 1.5R from peak; a breach ends the
     run, open positions included (GDD §6).
   · grading — setup legality vs the session labels (fade on a labeled trend
     day = cardinal, 3× weight, forced micro-replay), declared-read
     correctness vs the script's regime timeline, KILL/HOLD reflex latency.
   · Read Score + luck decomposition — legality-graded process score beside
     realized P&L; legal-correct losses → luck channel; illegal wins → zero
     credit (GDD §5 bright line).
   · measureTells — the trend tells the fader overrode, MEASURED from bars
     (IB width ratio, max pullback %, impulse volume ratio, one-timeframing
     brackets) for the forced micro-replay and the intro trap list.
   · runPolicy / fadeBot — replayable whole-run simulation used by the intro
     "last run / demo run" tape and by tests.

   Determinism: buildBossSession(seed) is pure in the seed (sibling-seed
   retry loop with a fixed variant base). Pure TS — zero DOM/Svelte imports.
   ========================================================================== */

import type { Bar, Regime, SessionLabels, Segment, Verdict } from '../types';
import { Prng, siblingSeed } from '../gen/prng';
import { compileScript, EASY_KNOBS, NORMAL_IB_POINTS } from '../gen/scripts';
import type { CompiledScript, DifficultyKnobs } from '../gen/scripts';
import { generateSession } from '../gen/generator';
import type { GeneratedSession } from '../gen/generator';
import { buildProfile, rowToPrice } from '../core/profile';
import { bracketOf, buildTpoProfile, findOneTimeframingBreak, BARS_PER_BRACKET } from '../core/tpo';
import { bossTemplate, renderBossTemplate } from './templates';

/* ----------------------------------------------------------------------------
   Constants (rules of engagement — displayed on the intro RoE panel)
   -------------------------------------------------------------------------- */

/** Daily loss limit in R. Crossing it ends the run, open positions included. */
export const DAILY_LOSS_LIMIT_R = -2.5;
/**
 * Trailing drawdown, in R, from the best REALIZED (flat) equity — open-trade
 * marks can breach it, but cannot raise the peak (an open profit never
 * banked is not a peak a desk trails from).
 */
export const TRAILING_DD_R = 3.0;
/** KILL/HOLD reflex window (GDD §6: 5-second interrupts). */
export const KILL_WINDOW_MS = 5000;
/** Latency for full reflex credit (sharp reflex; task spec <2s). */
export const KILL_SHARP_MS = 2000;
/** Read Score needed (with no breach) to pass the gauntlet. */
export const PASS_READ_SCORE = 70;
/** Cardinal errors weigh 3× in the Read Score and the rating pool (GDD §5). */
export const CARDINAL_WEIGHT = 3;
/** Bars after a fill at which the KILL/HOLD interrupt fires (scripted). */
export const INTERRUPT_DELAY_BARS = 30;
/** Structural stop pad beyond the reference extreme, in profile rows. */
export const STOP_PAD_ROWS = 2;
/** Swing lookback (bars) for the go-with structural stop. */
export const SWING_LOOKBACK = 45;
/** Initiative target projection for go-with, in R (structural v1 policy). */
export const GOWITH_TARGET_R = 2;

/** Knobs the boss compiles with (clean tape; the trap is cognitive, not noise). */
export const BOSS_KNOBS: DifficultyKnobs = { ...EASY_KNOBS };

/**
 * Seed of the intro screen's "demo run — the classic mistake" (shown before
 * a first attempt exists): the fade-bot on this session takes 3 fades, loses
 * all 3, and breaches the daily limit — the canonical Boss 1 story, computed
 * by the same engine that will grade the player (pinned by unit test).
 */
export const DEMO_SEED = '20260709';

/* ----------------------------------------------------------------------------
   Session
   -------------------------------------------------------------------------- */

export interface BossSession {
  /** The run seed the player sees / that is persisted with every decision. */
  seed: string;
  /** Sibling variant the retry loop settled on (re-derivability). */
  variant: number;
  /** The generated trend day (bars + labels + fabricated priors). */
  gen: GeneratedSession;
  /** Scripted decision bars — the end of each impulse leg (first three). */
  decisionBars: number[];
  /** Prior-day (priors[2]) landmarks, measured from its profile. */
  pVah: number;
  pVal: number;
  pPoc: number;
  /** Trend direction of the day: +1 up (forced by the seed retry loop). */
  trendDir: 1 | -1;
  /**
   * "Normal" IB width in points, MEASURED as the mean IB width of the three
   * fabricated prior (normal-day) sessions — the denominator of the IB tell.
   */
  normalIbPts: number;
}

/** Impulse legs of the script (the volume-expanding one-timeframing pushes). */
export function impulseSegments(script: Segment[]): Segment[] {
  return script.filter((s) => s.regime === 'imbalance' && s.volumeBoost >= 1.3);
}

/**
 * The scripted decision points: the end bar of each of the first three
 * impulse legs — price extended, "clean VA edge below, poor high overhead",
 * the classic fade temptation (GDD §6 Boss 1). Derived from the script,
 * never hand-placed.
 */
export function decisionBarsOf(labels: SessionLabels, nBars: number): number[] {
  return impulseSegments(labels.script)
    .map((s) => s.endBar)
    .filter((b) => b >= 60 && b <= nBars - 2 * BARS_PER_BRACKET)
    .slice(0, 3);
}

/**
 * Max generateSession calls the measured-predicate loop may spend. The
 * measured OTF-hold is the binding predicate (a few % of script-qualifying
 * candidates on unlucky seeds), so the budget is sized to make a full
 * qualifier — OTF hold AND pullback AND IB tell — the overwhelmingly common
 * outcome; the tier fallback stays as the deterministic last resort.
 */
const MAX_GEN_ATTEMPTS = 120;
/**
 * The tell contract: measured pullbacks must stay under this fraction of
 * their impulse (scripted ≤40% + noise headroom; a retrace reclaiming half
 * the impulse would genuinely question the trend read).
 */
export const PULLBACK_TELL_MAX = 0.5;
/**
 * IB tell contract: the MEASURED IB-width ratio (today's IB ÷ the mean of
 * the fabricated priors' measured IBs — exactly what the intro/debrief
 * display) must stay at or under this. GDD §7 scripts trend IB at 0.3–0.5×
 * normal; 0.55 is the noise headroom (wicks widen today's measured IB and
 * the prior draw varies the denominator). A session measuring wider would
 * contradict the "IB 0.3–0.5× normal" tell the debrief teaches.
 */
export const IB_TELL_MAX = 0.55;
/**
 * Measured bracket one-timeframing must hold at least through this bar —
 * past every decision point (≤295) and every scripted interrupt (≤325), so
 * "one-timeframing never breaks" is measurably true for the whole gauntlet.
 */
export const OTF_HOLD_THROUGH_BAR = 335;

/**
 * Build the gauntlet session. Retries sibling seeds until:
 *  script-level — (a) trend day in the up frame, (b) scripted one-timeframing
 *  never breaks, (c) a poor high overhead (the tidy-stop seduction);
 *  measured-level — (d) bracket one-timeframing holds through every decision
 *  and interrupt (findOneTimeframingBreak null or ≥ OTF_HOLD_THROUGH_BAR:
 *  the boss's core claim must be measurably true while the player acts),
 *  (e) every completed pullback ≤ PULLBACK_TELL_MAX of its impulse (the
 *  "pullbacks never exceed 40%" tell must not be contradicted by noise) AND
 *  the measured IB ratio ≤ IB_TELL_MAX (the "IB 0.3–0.5× normal" tell).
 * Preference degrades gracefully (d∧e → d → script-only) so the loop always
 * terminates deterministically on the same session for the same seed.
 */
export function buildBossSession(seed: string): BossSession {
  let tier1: BossSession | null = null; // script conditions only
  let tier2: BossSession | null = null; // + measured OTF never breaks
  let gens = 0;
  // ~11% of scripts qualify at the script level, so the variant budget must
  // stay much larger than the generation budget for the gen cap to bind.
  for (let v = 0; v < 1200 && gens < MAX_GEN_ATTEMPTS; v++) {
    const candidate = v === 0 ? seed : siblingSeed(seed, 9100 + v);
    const cs = compileScript(new Prng(candidate), 'trend', 'open-drive', 'out-of-range', BOSS_KNOBS);
    if (cs.dir !== 1) continue;
    if (cs.oneTimeframingBreakBar !== null) continue;
    if (cs.extremeHigh.mode !== 'poor') continue;
    const session = finishSession(cs, v);
    gens++;
    const { bars, labels } = session.gen;
    const otfBreak = otfBreakSince(bars, TREND_RUN_START_BAR, 'up');
    if (otfBreak !== null && otfBreak < OTF_HOLD_THROUGH_BAR) {
      if (tier1 === null) tier1 = session;
      continue;
    }
    const tells = measureTells(bars, labels, bars.length - 1, cs.rowStep, 1, session.normalIbPts);
    if (
      (tells.maxPullbackFrac !== null && tells.maxPullbackFrac > PULLBACK_TELL_MAX) ||
      tells.ibRatio > IB_TELL_MAX
    ) {
      if (tier2 === null) tier2 = session;
      continue;
    }
    return session;
  }
  const best = tier2 ?? tier1;
  /* v8 ignore next 3 — P(zero qualifying scripts in the budget) ≈ 0 */
  if (best === null) throw new Error('buildBossSession: no qualifying trend session in budget');
  return best;
}

function finishSession(cs: CompiledScript, variant: number): BossSession {
  const gen = generateSession(cs);
  const p2 = gen.priors[2].profile;
  let ibSum = 0;
  for (const p of gen.priors) {
    const tpo = buildTpoProfile(p.bars, cs.rowStep);
    ibSum += tpo.ibHigh - tpo.ibLow;
  }
  return {
    seed: cs.seed,
    variant,
    gen,
    decisionBars: decisionBarsOf(gen.labels, gen.script.nBars),
    pVah: rowToPrice(p2, p2.vah),
    pVal: rowToPrice(p2, p2.val),
    pPoc: rowToPrice(p2, p2.poc),
    trendDir: 1,
    normalIbPts: ibSum / gen.priors.length,
  };
}

/**
 * The bar the trend run starts (end of the IB / first impulse leg): the
 * anchor of the "one-timeframing since {10:30}" claim. Pre-IB rotation is
 * not part of the trend's one-timeframing story.
 */
export const TREND_RUN_START_BAR = 60;

/**
 * Measured one-timeframing break anchored at `fromBar` (bracket clock
 * restarts there). Returns the absolute break bar index, or null if control
 * never breaks through the end of `bars`.
 */
export function otfBreakSince(bars: Bar[], fromBar: number, direction: 'up' | 'down'): number | null {
  const shifted = bars.slice(fromBar).map((b, i) => ({ ...b, t: i }));
  const brk = findOneTimeframingBreak(shifted, direction);
  return brk === null ? null : brk + fromBar;
}

/** Session clock label for a bar index (RTH open = 09:30). */
export function barClock(bar: number): string {
  const m = 9 * 60 + 30 + bar;
  return `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`;
}

/* ----------------------------------------------------------------------------
   Order-ticket plans (structural stops/targets, guide §4.11)
   -------------------------------------------------------------------------- */

export type BossAction = 'fade' | 'go-with' | 'stand-aside';
export type DeclaredRead = Regime; // 'balance' | 'imbalance'

export interface TicketPlan {
  /** Fill price = the decision bar's close. */
  entry: number;
  /** Structural stop price. */
  stop: number;
  /** Structural target price. */
  target: number;
  /** Initial risk in points (|entry − stop|). */
  riskPts: number;
  /** Reward at target, in R. */
  rewardR: number;
}

/**
 * The responsive fade every trend day makes look perfect: short at the
 * extended price, tidy structural stop just above the (poor) session high,
 * target back at the developing POC — the classic edge-to-POC responsive
 * rotation (guide §4.1). All three prices measured, never invented.
 */
export function planFade(bars: Bar[], upTo: number, rowStep: number): TicketPlan {
  const entry = bars[upTo].c;
  let hi = -Infinity;
  for (let i = 0; i <= upTo; i++) if (bars[i].h > hi) hi = bars[i].h;
  const stop = hi + STOP_PAD_ROWS * rowStep;
  const riskPts = stop - entry;
  const prof = buildProfile(bars.slice(0, upTo + 1), rowStep);
  // target floor: the ticket always projects ≥2R (a rotation target closer
  // than that is no responsive trade — take the deeper of POC and 2R)
  const target = Math.min(rowToPrice(prof, prof.poc), entry - 2 * riskPts);
  return { entry, stop, target, riskPts, rewardR: (entry - target) / riskPts };
}

/** Initiative go-with: stop under the swing low, 2R structural projection. */
export function planGoWith(bars: Bar[], upTo: number, rowStep: number): TicketPlan {
  const entry = bars[upTo].c;
  let lo = Infinity;
  for (let i = Math.max(0, upTo - (SWING_LOOKBACK - 1)); i <= upTo; i++) {
    if (bars[i].l < lo) lo = bars[i].l;
  }
  const stop = lo - STOP_PAD_ROWS * rowStep;
  const riskPts = entry - stop;
  const target = entry + GOWITH_TARGET_R * riskPts;
  return { entry, stop, target, riskPts, rewardR: GOWITH_TARGET_R };
}

/* ----------------------------------------------------------------------------
   Equity reducer (per-bar mark-to-market in R + prop-desk limits)
   -------------------------------------------------------------------------- */

export interface OpenPosition {
  action: Extract<BossAction, 'fade' | 'go-with'>;
  /** +1 long, −1 short. */
  dir: 1 | -1;
  entryBar: number;
  plan: TicketPlan;
  /** Index of the decision that opened it (exit attribution). */
  decisionIndex: number;
}

export type ExitOutcome = 'stop' | 'target' | 'close' | 'breach';

export interface TradeExit {
  bar: number;
  rPnl: number;
  outcome: ExitOutcome;
  decisionIndex: number;
}

export interface EquityState {
  realizedR: number;
  /** realizedR + open-position mark at the last stepped bar's close. */
  equityR: number;
  peakR: number;
  open: OpenPosition | null;
  breach: 'daily' | 'trailing' | null;
  /** Bar index the breach fired on (clock for the verdict), or null. */
  breachBar: number | null;
  /** Exit realized while stepping the most recent bar, if any. */
  lastExit: TradeExit | null;
}

export function newEquity(): EquityState {
  return {
    realizedR: 0,
    equityR: 0,
    peakR: 0,
    open: null,
    breach: null,
    breachBar: null,
    lastExit: null,
  };
}

/** Unrealized R of a position marked at `price`. */
export function unrealizedR(pos: OpenPosition, price: number): number {
  return (pos.dir * (price - pos.plan.entry)) / pos.plan.riskPts;
}

/**
 * Step one bar: resolve stop/target (stop priority when both print inside
 * the same bar — conservative fill convention), mark equity, and test the
 * desk limits. A breach force-closes any open position at the bar's close.
 * Mutates and returns `st`.
 */
export function stepBar(st: EquityState, bar: Bar): EquityState {
  st.lastExit = null;
  if (st.breach) return st;

  const pos = st.open;
  if (pos) {
    const stopHit = pos.dir === -1 ? bar.h >= pos.plan.stop : bar.l <= pos.plan.stop;
    const targetHit = pos.dir === -1 ? bar.l <= pos.plan.target : bar.h >= pos.plan.target;
    if (stopHit || targetHit) {
      const rPnl = stopHit ? -1 : pos.plan.rewardR;
      st.realizedR += rPnl;
      st.lastExit = { bar: bar.t, rPnl, outcome: stopHit ? 'stop' : 'target', decisionIndex: pos.decisionIndex };
      st.open = null;
    }
  }

  st.equityR = st.realizedR + (st.open ? unrealizedR(st.open, bar.c) : 0);
  // peak trails REALIZED equity only (flat ⇒ equity == realized)
  if (st.open === null && st.equityR > st.peakR) st.peakR = st.equityR;

  if (st.equityR <= DAILY_LOSS_LIMIT_R) st.breach = 'daily';
  else if (st.peakR - st.equityR >= TRAILING_DD_R) st.breach = 'trailing';

  if (st.breach) {
    st.breachBar = bar.t;
    if (st.open) {
      const rPnl = unrealizedR(st.open, bar.c);
      st.realizedR += rPnl;
      st.lastExit = { bar: bar.t, rPnl, outcome: 'breach', decisionIndex: st.open.decisionIndex };
      st.open = null;
    }
    st.equityR = st.realizedR;
  }
  return st;
}

/** Enter a position at the close of `bar` (call AFTER stepping that bar). */
export function enterPosition(
  st: EquityState,
  action: Extract<BossAction, 'fade' | 'go-with'>,
  plan: TicketPlan,
  entryBar: number,
  decisionIndex: number,
): EquityState {
  st.open = { action, dir: action === 'fade' ? -1 : 1, entryBar, plan, decisionIndex };
  return st;
}

/** Player-invoked KILL: realize the open position at the bar's close. */
export function killPosition(st: EquityState, bar: Bar): EquityState {
  if (!st.open) return st;
  const rPnl = unrealizedR(st.open, bar.c);
  st.realizedR += rPnl;
  st.lastExit = { bar: bar.t, rPnl, outcome: 'close', decisionIndex: st.open.decisionIndex };
  st.open = null;
  st.equityR = st.realizedR;
  if (st.equityR > st.peakR) st.peakR = st.equityR;
  return st;
}

/** Session over: realize any open position at the final close. */
export function closeAtEnd(st: EquityState, lastBar: Bar): EquityState {
  if (st.open && !st.breach) {
    const rPnl = unrealizedR(st.open, lastBar.c);
    st.realizedR += rPnl;
    st.lastExit = { bar: lastBar.t, rPnl, outcome: 'close', decisionIndex: st.open.decisionIndex };
    st.open = null;
    st.equityR = st.realizedR;
  }
  return st;
}

/* ----------------------------------------------------------------------------
   Grading (legality vs labels — no EV oracle, GDD §5/§6)
   -------------------------------------------------------------------------- */

/** Regime the script declares at a bar (ground truth for the declared read). */
export function scriptRegimeAt(labels: SessionLabels, bar: number): Regime {
  for (const s of labels.script) if (bar >= s.startBar && bar <= s.endBar) return s.regime;
  return labels.script[labels.script.length - 1].regime;
}

/** Clock the current one-timeframing run started (contiguous imbalance run). */
export function otfSinceClock(labels: SessionLabels, atBar: number): string {
  let start = atBar;
  for (let i = labels.script.length - 1; i >= 0; i--) {
    const s = labels.script[i];
    if (atBar < s.startBar) continue;
    if (s.regime !== 'imbalance') break;
    start = s.startBar;
  }
  return barClock(start);
}

export interface DecisionGrade {
  /** Setup legality vs the labels (fade on a labeled trend day = illegal). */
  legal: boolean;
  /** Cardinal error → 3× weight + forced micro-replay (GDD §6). */
  cardinal: boolean;
  /** Was the declared read the script's regime at the decision bar? */
  readCorrect: boolean;
  /** 0–100 process score for this decision. */
  score: number;
  /** Read-Score weight (cardinal ×3). */
  weight: number;
  /** Rendered Verdict (canonical template string, refs attached). */
  verdict: Verdict;
}

/**
 * Grade one order-ticket decision against the session labels.
 * Scores: go-with+imbalance 100 · stand-aside+imbalance 75 · go-with+balance
 * 55 · stand-aside+balance 40 · any fade on a trend day 0 (cardinal ×3).
 */
export function gradeDecision(
  action: BossAction,
  read: DeclaredRead,
  labels: SessionLabels,
  atBar: number,
  plan: TicketPlan | null,
  trendDir: 1 | -1,
): DecisionGrade {
  const truthRegime = scriptRegimeAt(labels, atBar);
  const readCorrect = read === truthRegime;
  const dirWord = trendDir === 1 ? 'up' : 'down';

  const make = (
    legal: boolean,
    cardinal: boolean,
    score: number,
    templateId: string,
    slots: Record<string, string | number>,
  ): DecisionGrade => {
    const t = bossTemplate(templateId);
    return {
      legal,
      cardinal,
      readCorrect,
      score,
      weight: cardinal ? CARDINAL_WEIGHT : 1,
      verdict: {
        correct: legal && readCorrect,
        score,
        explanation: renderBossTemplate(t, slots),
        explanationTemplateId: templateId,
        refs: t.refs,
        cardinal,
        brier: null,
      },
    };
  };

  if (action === 'fade' && labels.dayType === 'trend') {
    return make(false, true, 0, 'boss1.fade.cardinal', { time: otfSinceClock(labels, atBar) });
  }
  if (action === 'go-with') {
    return readCorrect
      ? make(true, false, 100, 'boss1.gowith.hit', { stop: (plan?.stop ?? 0).toFixed(2) })
      : make(true, false, 55, 'boss1.gowith.wrongRead', { dir: dirWord });
  }
  // stand-aside (also the fade path on a non-trend day, unreachable in boss 1)
  return readCorrect
    ? make(true, false, 75, 'boss1.stand.hit', {})
    : make(true, false, 40, 'boss1.stand.wrongRead', { dir: dirWord });
}

/* ---- KILL/HOLD reflex ------------------------------------------------------ */

export type ReflexAnswer = 'KILL' | 'HOLD';

/** Ground truth: with-trend position → HOLD, counter-trend → KILL. */
export function reflexTruth(pos: OpenPosition, trendDir: 1 | -1): ReflexAnswer {
  return pos.dir === trendDir ? 'HOLD' : 'KILL';
}

export interface ReflexGrade {
  correct: boolean;
  /** No answer inside the window. */
  slept: boolean;
  /** 100 sharp (≤2s) · 60 slow (≤5s) · 0 wrong/slept. */
  score: number;
  weight: number;
  verdict: Verdict;
}

/** Grade a KILL/HOLD reflex answer on correctness AND latency (GDD §6). */
export function gradeReflex(
  truth: ReflexAnswer,
  answer: ReflexAnswer | null,
  latencyMs: number,
): ReflexGrade {
  const make = (correct: boolean, slept: boolean, score: number, templateId: string, slots: Record<string, string | number> = {}): ReflexGrade => {
    const t = bossTemplate(templateId);
    return {
      correct,
      slept,
      score,
      weight: 1,
      verdict: {
        correct,
        score,
        explanation: renderBossTemplate(t, slots),
        explanationTemplateId: templateId,
        refs: t.refs,
        cardinal: false,
        brier: null,
      },
    };
  };
  if (answer === null) {
    return make(false, true, 0, 'boss1.reflex.slept', { sec: (KILL_WINDOW_MS / 1000).toFixed(1) });
  }
  if (answer === truth) {
    const sharp = latencyMs <= KILL_SHARP_MS;
    return make(true, false, sharp ? 100 : 60, truth === 'KILL' ? 'boss1.kill.hit' : 'boss1.hold.hit');
  }
  return make(false, false, 0, truth === 'KILL' ? 'boss1.kill.miss' : 'boss1.hold.miss');
}

/* ----------------------------------------------------------------------------
   Read Score + luck decomposition (GDD §5 boss scoring)
   -------------------------------------------------------------------------- */

export interface GradedEvent {
  kind: 'decision' | 'reflex';
  /** 0–100 process score. */
  score: number;
  /** Read-Score weight (cardinal decisions carry 3). */
  weight: number;
  /** Setup legality (reflex events count as legal process events). */
  legal: boolean;
  /** Declared-read correctness (reflex: answer correctness). */
  readCorrect: boolean;
  cardinal: boolean;
  /** Realized R of the decision's trade; null when no position was taken. */
  rPnl: number | null;
}

/** Weighted mean of process scores — the Read Score (0–100, rounded). */
export function readScore(events: GradedEvent[]): number {
  if (events.length === 0) return 0;
  let s = 0;
  let w = 0;
  for (const e of events) {
    s += e.score * e.weight;
    w += e.weight;
  }
  return Math.round(s / w);
}

export interface Decomposition {
  /** R earned by legal, correctly-read decisions that paid. */
  earnedR: number;
  /**
   * R LOST on wrong-process decisions (illegal setups or misread regimes) —
   * the predictable cost of the mistake, never variance. ≤ 0.
   */
  errorR: number;
  /**
   * The variance channel: losses on right-process decisions ("right process,
   * unlucky") plus wins on wrong-process decisions (windfalls that earn zero
   * credit). pnl − earned − error.
   */
  luckR: number;
  /** The decomposition sentence (GDD §8 screen 7). */
  sentence: string;
  /** The quadrant tail ("Right process, unlucky outcome."). */
  tail: string;
}

const fmtR = (r: number): string => `${r >= 0 ? '+' : '−'}${Math.abs(r).toFixed(1)}R`;

/**
 * Label-based luck decomposition (no EV oracle, GDD §5): legal + correctly
 * read decisions that made money are "earned"; legal-correct losses are
 * attributed to the luck channel; illegal reads that made money earn zero
 * credit — no XP for wrong-process lucky wins. Wrong-process LOSSES are the
 * error channel, not luck: a trend-day fade losing is the predictable cost
 * of the illegal read, and calling it "variance" would launder the mistake
 * the boss exists to expose.
 */
export function decompose(events: GradedEvent[], pnlR: number, score: number): Decomposition {
  let earnedR = 0;
  let errorR = 0;
  for (const e of events) {
    if (e.kind !== 'decision' || e.rPnl === null) continue;
    const rightProcess = e.legal && e.readCorrect;
    if (rightProcess && e.rPnl > 0) earnedR += e.rPnl;
    if (!rightProcess && e.rPnl < 0) errorR += e.rPnl;
  }
  const luckR = pnlR - earnedR - errorR;
  const sentence =
    errorR < 0
      ? `Your reads earned ${fmtR(earnedR)} of decisions; wrong reads cost ${fmtR(errorR)} — that isn't variance; variance handed you ${fmtR(luckR)}.`
      : `Your reads earned ${fmtR(earnedR)} of decisions; variance handed you ${fmtR(luckR)}.`;
  let tail: string;
  if (score >= PASS_READ_SCORE && pnlR < 0) tail = 'Right process, unlucky outcome.';
  else if (score < PASS_READ_SCORE && pnlR > 0) tail = 'Lucky outcome, wrong process — no credit.';
  else if (score >= PASS_READ_SCORE) tail = 'Right process, paid.';
  else tail = 'Process and outcome agree — drill the tells.';
  return { earnedR, errorR, luckR, sentence, tail };
}

/* ----------------------------------------------------------------------------
   Trend tells — measured, for the forced micro-replay + intro trap list
   -------------------------------------------------------------------------- */

export interface TrendTells {
  /** IB width ÷ normal IB width (measured vs the priors; trend ≈ 0.3–0.6). */
  ibRatio: number;
  /** Deepest completed pullback as a fraction of its impulse; null if none yet. */
  maxPullbackFrac: number | null;
  /**
   * Mean impulse-bar volume ÷ mean rotation-bar volume (>1 = volume expands
   * with the move). Rotations = the pullback legs when at least one has
   * completed ('pullbacks' basis — the guide's actual tell); before the
   * first pullback the only fair baseline is the post-IB balance bars
   * ('balance' basis), which the U-shaped seasonal keeps weak.
   */
  impulseVolRatio: number;
  impulseVolBasis: 'pullbacks' | 'balance';
  /**
   * Completed 30-min brackets of unbroken one-timeframing in the trend run
   * (anchored at TREND_RUN_START_BAR) through `upTo`.
   */
  otfBrackets: number;
}

/**
 * Measure the trend tells from the bars through `upTo` (inclusive).
 * `normalIbPts` should be the measured mean prior IB width
 * (BossSession.normalIbPts); it defaults to the script constant.
 */
export function measureTells(
  bars: Bar[],
  labels: SessionLabels,
  upTo: number,
  rowStep: number,
  trendDir: 1 | -1 = 1,
  normalIbPts: number = NORMAL_IB_POINTS,
): TrendTells {
  const slice = bars.slice(0, upTo + 1);
  const tpo = buildTpoProfile(slice, rowStep);
  const ibRatio = (tpo.ibHigh - tpo.ibLow) / normalIbPts;

  const impulses = impulseSegments(labels.script);
  // pullback legs: post-IB script segments between impulses with volume fade
  const pulls = labels.script.filter(
    (s) => s.startBar >= 60 && s.regime === 'imbalance' && s.volumeBoost < 1,
  );

  let maxPullbackFrac: number | null = null;
  for (const p of pulls) {
    if (p.endBar > upTo) break;
    const imp = impulses.find((s) => s.endBar === p.startBar - 1);
    if (!imp) continue;
    let impHi = -Infinity;
    let impLo = Infinity;
    for (let i = imp.startBar; i <= imp.endBar; i++) {
      if (bars[i].h > impHi) impHi = bars[i].h;
      if (bars[i].l < impLo) impLo = bars[i].l;
    }
    let pullExt = trendDir === 1 ? Infinity : -Infinity;
    for (let i = p.startBar; i <= p.endBar; i++) {
      pullExt = trendDir === 1 ? Math.min(pullExt, bars[i].l) : Math.max(pullExt, bars[i].h);
    }
    const move = impHi - impLo;
    if (move <= 0) continue;
    const frac = trendDir === 1 ? (impHi - pullExt) / move : (pullExt - impLo) / move;
    if (maxPullbackFrac === null || frac > maxPullbackFrac) maxPullbackFrac = frac;
  }

  let impVol = 0;
  let impN = 0;
  let pullVol = 0;
  let pullN = 0;
  let balVol = 0;
  let balN = 0;
  for (let i = 0; i <= upTo; i++) {
    if (impulses.some((s) => i >= s.startBar && i <= s.endBar)) {
      impVol += bars[i].v;
      impN++;
    } else if (pulls.some((s) => i >= s.startBar && i <= s.endBar)) {
      pullVol += bars[i].v;
      pullN++;
    } else if (i >= 30) {
      // post-open-seasonal baseline (the first half hour's λ spike would
      // poison any "volume expands with the move" comparison)
      balVol += bars[i].v;
      balN++;
    }
  }
  const impulseVolBasis: TrendTells['impulseVolBasis'] = pullN > 0 ? 'pullbacks' : 'balance';
  const denom = pullN > 0 ? pullVol / pullN : balN > 0 ? balVol / balN : 0;
  const impulseVolRatio = impN > 0 && denom > 0 ? impVol / impN / denom : 1;

  const breakBar =
    upTo > TREND_RUN_START_BAR
      ? otfBreakSince(slice, TREND_RUN_START_BAR, trendDir === 1 ? 'up' : 'down')
      : null;
  const otfBrackets =
    upTo <= TREND_RUN_START_BAR
      ? 0
      : breakBar === null
        ? bracketOf(upTo) - 1
        : Math.max(0, bracketOf(breakBar) - 2);

  return { ibRatio, maxPullbackFrac, impulseVolRatio, impulseVolBasis, otfBrackets };
}

/* ----------------------------------------------------------------------------
   Whole-run policy simulation (intro tape + tests)
   -------------------------------------------------------------------------- */

export interface PolicyTrade {
  decisionIndex: number;
  bar: number;
  action: Extract<BossAction, 'fade' | 'go-with'>;
  plan: TicketPlan;
  exit: TradeExit | null;
}

export interface PolicyRun {
  trades: PolicyTrade[];
  pnlR: number;
  breach: 'daily' | 'trailing' | null;
  breachBar: number | null;
  /** Last bar stepped (nBars−1, or the breach bar). */
  endBar: number;
}

/**
 * Run a full session under a fixed decision policy (null = stand aside).
 * The same reducer the live run uses — the intro "last run" tape and the
 * demo run are this function's output, never hand-drawn.
 */
export function runPolicy(
  session: BossSession,
  policy: (decisionIndex: number, bar: number) => Extract<BossAction, 'fade' | 'go-with'> | null,
): PolicyRun {
  const { bars } = session.gen;
  const rowStep = session.gen.script.rowStep;
  const st = newEquity();
  const trades: PolicyTrade[] = [];
  let endBar = bars.length - 1;

  for (let t = 0; t < bars.length; t++) {
    stepBar(st, bars[t]);
    if (st.lastExit) {
      const tr = trades[st.lastExit.decisionIndex];
      if (tr) tr.exit = st.lastExit;
    }
    if (st.breach) {
      endBar = t;
      break;
    }
    const di = session.decisionBars.indexOf(t);
    if (di >= 0 && !st.open) {
      const action = policy(di, t);
      if (action) {
        const plan = action === 'fade' ? planFade(bars, t, rowStep) : planGoWith(bars, t, rowStep);
        enterPosition(st, action, plan, t, trades.length);
        trades.push({ decisionIndex: trades.length, bar: t, action, plan, exit: null });
      }
    }
  }
  closeAtEnd(st, bars[endBar]);
  if (st.lastExit) {
    const tr = trades[st.lastExit.decisionIndex];
    if (tr && !tr.exit) tr.exit = st.lastExit;
  }
  return { trades, pnlR: st.realizedR, breach: st.breach, breachBar: st.breachBar, endBar };
}

/** The classic mistake, simulated: fade every decision point. */
export function fadeBot(session: BossSession): PolicyRun {
  return runPolicy(session, () => 'fade');
}

/** Intro-tape caption for a policy run — every number computed. */
export function runCaption(run: PolicyRun): string {
  const fades = run.trades.filter((t) => t.action === 'fade');
  const losses = fades.filter((t) => (t.exit?.rPnl ?? 0) < 0).length;
  const head = `${fades.length} fade${fades.length === 1 ? '' : 's'} taken · ${losses} ${losses === 1 ? 'loss' : 'losses'}`;
  if (run.breach) return `${head} · limit breached ${barClock(run.breachBar ?? run.endBar)}`;
  return `${head} · closed ${fmtR(run.pnlR)}`;
}
