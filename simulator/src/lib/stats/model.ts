/* ============================================================================
   @stats/model — the Stats dashboard view-model (GDD §5 scoring surfaces,
   §8 screen 8). Every number the screen shows is computed here from the
   persisted decision ledger and the boss trade fills — never invented in the
   UI layer.

   Surfaces (GDD §5, "the Metaculus-trap rule"):
   1. Bot Points per block of 25 — AP = Σ(Brier_bot − Brier_you)×100, the bot
      answering the pool's declared base rate (for chip drills the uniform
      1/nChoices chooser, the same bot the drill-block summary uses).
   2. Reliability curve behind a sentence headline, dots sized by n.
   3. Raw Brier trend (rolling BRIER_WINDOW) vs the Drill E mastery bar.
   Plus the expectancy ledger: setup × regime, cells gray until n ≥ 30 —
   "unmeasured = folklore" (GDD §5). Folklore column strings quote ONLY claims
   the domain guide actually records (guide Part IV / VI.3); setups the guide
   gives no number for show none.

   Pure TS — zero DOM/Svelte imports.
   ========================================================================== */

import type { DecisionRecord, DrillId, Regime, TradeFill } from '../types';
import {
  BLOCK_SIZE,
  BRIER_WINDOW,
  blockBotPoints,
  brierLedger,
  calibrationBuckets,
  calibrationEntryOf,
  rollingBrier,
} from '../schedule/persist';
import type { BotComparableEntry, CalibrationEntry } from '../schedule/persist';
import { MASTERY_CRITERIA } from '../schedule/queue';

/* ----------------------------------------------------------------------------
   Scope
   -------------------------------------------------------------------------- */

/** The dashboard's scope window in days ("last 90 days · all modes"). */
export const SCOPE_DAYS = 90;

/** Decisions inside the scope window, oldest first. */
export function scopeDecisions(decisions: DecisionRecord[], now: number): DecisionRecord[] {
  const cutoff = now - SCOPE_DAYS * 24 * 60 * 60 * 1000;
  return decisions
    .filter((d) => d.at >= cutoff)
    .sort((a, b) => a.at - b.at);
}

/* ----------------------------------------------------------------------------
   Calibration ledger → bot points + Brier trend
   -------------------------------------------------------------------------- */

/**
 * Answer-space size per drill (a structural property of each drill's chip
 * set, GDD §4) — the uniform base-rate bot answers 1/n. Drills with a
 * continuous answer space carry no chip bot (0).
 */
export const DRILL_CHOICES: Record<DrillId, number> = {
  'poc-va-snap': 0,
  'excess-or-poor': 2,
  'hvn-lvn-marker': 0,
  'shape-alphabet': 5,
  'calibration-range': 0,
  'open-type-ladder': 4,
  'one-timeframing-buzzer': 0,
  'acceptance-clock': 2,
  'regime-gate': 3,
  'setup-picker': 10,
};

/**
 * The bot-comparable calibration stream: every decision that carries a
 * probability (confidence tap / slider) on a drill with a declared chip base
 * rate, oldest first. calibration-range sliders are excluded from the BOT
 * comparison (their base rate is the pool's declared conditional, which the
 * ledger row does not carry) but still feed the reliability curve via
 * brierLedger.
 */
export function botEntries(decisions: DecisionRecord[]): BotComparableEntry[] {
  const out: BotComparableEntry[] = [];
  for (const d of decisions.slice().sort((a, b) => a.at - b.at)) {
    const n = DRILL_CHOICES[d.drillId];
    if (!n || n < 2) continue;
    const e = calibrationEntryOf(d);
    if (!e) continue;
    out.push({ ...e, baseRate: 1 / n });
  }
  return out;
}

/** Bot Points per completed block of 25 (GDD §5 surface 1), oldest first. */
export function botBlockSeries(decisions: DecisionRecord[]): number[] {
  return blockBotPoints(botEntries(decisions)).map((x) => Math.round(x));
}

/**
 * Rolling Brier sampled at each completed block boundary (GDD §5 surface 3),
 * oldest first. Empty until the first full block.
 */
export function brierBlockSeries(entries: CalibrationEntry[]): number[] {
  const out: number[] = [];
  for (let k = BLOCK_SIZE; k <= entries.length; k += BLOCK_SIZE) {
    const b = rollingBrier(entries.slice(0, k), BRIER_WINDOW);
    if (b !== null) out.push(b);
  }
  return out;
}

/** The Drill E mastery Brier bar (GDD §4: ≤0.18 over rolling 50). */
export const BRIER_BAR = MASTERY_CRITERIA['calibration-range'].maxBrier ?? 0.18;

/** "Beating the bot {won} of the last {of} blocks." */
export function beatingTheBot(botBlocks: number[], lastN = 12): { won: number; of: number } {
  const tail = botBlocks.slice(-lastN);
  return { won: tail.filter((x) => x > 0).length, of: tail.length };
}

/* ----------------------------------------------------------------------------
   Reliability curve + headline sentence
   -------------------------------------------------------------------------- */

/** Minimum judgments before a bucket earns a headline. */
export const MIN_BUCKET_N = 10;
/** Gap (percentage points) below which a bucket counts as honest. */
export const HONEST_GAP_PTS = 5;

/** One plotted reliability dot. */
export interface CalibrationDot {
  /** Mean stated probability of the bucket, %. */
  saidPct: number;
  /** Observed hit rate, %. */
  hapPct: number;
  /** Judgments in the bucket. */
  n: number;
}

/** The calibration card view. */
export interface CalibrationView {
  dots: CalibrationDot[];
  /** Total probability-tagged judgments. */
  n: number;
  /** The worst bucket with n ≥ MIN_BUCKET_N; null when nothing qualifies. */
  worst: CalibrationDot | null;
  /** Sentence parts, so the UI can set the numbers in mono ("When you say X%, it happens Y% — …"). */
  headline: { said: number; happened: number; clause: string } | null;
  /** Fallback sentence when no bucket qualifies yet. */
  emptyLine: string;
}

/** Build the reliability-curve view from the calibration ledger. */
export function calibrationView(entries: CalibrationEntry[]): CalibrationView {
  const dots: CalibrationDot[] = calibrationBuckets(entries, 10)
    .filter((b) => b.n > 0 && b.meanP !== null && b.hitRate !== null)
    .map((b) => ({
      saidPct: Math.round((b.meanP as number) * 100),
      hapPct: Math.round((b.hitRate as number) * 100),
      n: b.n,
    }));

  const qualified = dots.filter((d) => d.n >= MIN_BUCKET_N);
  let worst: CalibrationDot | null = null;
  for (const d of qualified) {
    if (!worst || Math.abs(d.hapPct - d.saidPct) > Math.abs(worst.hapPct - worst.saidPct)) {
      worst = d;
    }
  }

  let headline: CalibrationView['headline'] = null;
  if (worst) {
    const gap = worst.hapPct - worst.saidPct;
    let clause: string;
    if (Math.abs(gap) <= HONEST_GAP_PTS) {
      clause = 'well calibrated';
    } else {
      const dir = gap < 0 ? 'overconfident' : 'underconfident';
      // Region qualifier: the lowest stated % from which every qualified
      // bucket leans the same way as the worst one.
      const sameWay = qualified.filter(
        (d) => Math.sign(d.hapPct - d.saidPct) === Math.sign(gap) && Math.abs(d.hapPct - d.saidPct) > HONEST_GAP_PTS,
      );
      const lo = Math.min(...sameWay.map((d) => d.saidPct));
      clause = lo < worst.saidPct ? `${dir} above ${lo}` : `${dir} at ${worst.saidPct}`;
      // Honesty rider for the guess band.
      const guessBand = qualified.filter((d) => d.saidPct <= 60);
      if (
        guessBand.length > 0 &&
        guessBand.every((d) => Math.abs(d.hapPct - d.saidPct) <= HONEST_GAP_PTS)
      ) {
        clause += '. Guesses are honest';
      }
    }
    headline = { said: worst.saidPct, happened: worst.hapPct, clause };
  }

  return {
    dots,
    n: entries.length,
    worst,
    headline,
    emptyLine: `Calibration needs ~${BLOCK_SIZE} probability-tagged judgments — the curve unlocks as confidence taps accumulate.`,
  };
}

/* ----------------------------------------------------------------------------
   Expectancy ledger (setup × regime) + R-multiple histogram
   -------------------------------------------------------------------------- */

/** A boss/sim trade row extracted from the ledger. */
export interface SimTrade extends TradeFill {
  at: number;
}

/** Extract every executed trade from boss-mode decisions, oldest first. */
export function tradesOf(decisions: DecisionRecord[]): SimTrade[] {
  return decisions
    .filter((d): d is DecisionRecord & { trade: TradeFill } => d.mode === 'boss' && d.trade !== undefined)
    .sort((a, b) => a.at - b.at)
    .map((d) => ({ ...d.trade, at: d.at }));
}

/** Cells stay gray until n ≥ 30 (GDD §5 — "unmeasured = folklore"). */
export const MEASURED_MIN_N = 30;

/**
 * The 9 playbook setups (guide Part IV §4.1–4.9) with the folklore claims the
 * guide actually records (Part IV / VI.3 provenance table). Setups the guide
 * gives no probability lore for carry null — inventing one would BE folklore.
 */
export const PLAYBOOK_SETUPS: { name: string; folklore: string | null }[] = [
  { name: 'VA-Edge Fade', folklore: null },
  { name: 'Value Breakout', folklore: null },
  { name: '80% Rule', folklore: '“~80%”' }, // measured ~60–67% (guide §4.3)
  { name: 'Open-Drive Go', folklore: null },
  { name: 'Look-Above & Fail', folklore: '“~70–75%”' }, // vendor statistic (guide §4.5)
  { name: 'nPOC Magnet', folklore: '“80% in 10 sess”' }, // vendor claim (guide §4.6)
  { name: 'LVN Break', folklore: null },
  { name: 'Anchored Pullback', folklore: null },
  { name: 'POC Reversion', folklore: null },
];

/** One measured cell of the expectancy table. */
export interface ExpectancyCell {
  n: number;
  /** Mean R-multiple (E = p·W − (1−p)·L collapses to the mean of R). */
  expectancy: number;
  /** Win rate in [0,1]. */
  winRate: number;
  /** True when n ≥ MEASURED_MIN_N — only then is the cell shown. */
  measured: boolean;
}

/** One row of the expectancy table. */
export interface ExpectancyRow {
  setup: string;
  folklore: string | null;
  balance: ExpectancyCell | null;
  imbalance: ExpectancyCell | null;
}

function cellOf(trades: SimTrade[]): ExpectancyCell | null {
  if (trades.length === 0) return null;
  const n = trades.length;
  return {
    n,
    expectancy: trades.reduce((s, t) => s + t.rMultiple, 0) / n,
    winRate: trades.filter((t) => t.rMultiple > 0).length / n,
    measured: n >= MEASURED_MIN_N,
  };
}

/** The full setup × regime expectancy table, in playbook order. */
export function expectancyTable(trades: SimTrade[]): ExpectancyRow[] {
  return PLAYBOOK_SETUPS.map(({ name, folklore }) => {
    const mine = trades.filter((t) => t.setup === name);
    return {
      setup: name,
      folklore,
      balance: cellOf(mine.filter((t) => t.regime === 'balance')),
      imbalance: cellOf(mine.filter((t) => t.regime === 'imbalance')),
    };
  });
}

/** The best measured edge, for the card's lede; null when nothing is measured. */
export function bestMeasuredEdge(
  rows: ExpectancyRow[],
): { setup: string; regime: Regime; cell: ExpectancyCell } | null {
  let best: { setup: string; regime: Regime; cell: ExpectancyCell } | null = null;
  for (const r of rows) {
    for (const regime of ['balance', 'imbalance'] as const) {
      const cell = r[regime];
      if (cell && cell.measured && (!best || cell.expectancy > best.cell.expectancy)) {
        best = { setup: r.setup, regime, cell };
      }
    }
  }
  return best;
}

/** R-multiple histogram: fixed 0.5R bins from −2R to +3R, outliers clamped in. */
export interface RHistogram {
  /** Bin left edge in R for bins[i] = lo + i*step. */
  lo: number;
  step: number;
  counts: number[];
  /** Mean R over all trades. */
  avg: number;
  n: number;
}

export const HIST_LO = -2;
export const HIST_STEP = 0.5;
export const HIST_BINS = 10;

export function rHistogram(trades: SimTrade[]): RHistogram {
  const counts = new Array<number>(HIST_BINS).fill(0);
  for (const t of trades) {
    const i = Math.min(
      HIST_BINS - 1,
      Math.max(0, Math.floor((t.rMultiple - HIST_LO) / HIST_STEP)),
    );
    counts[i]++;
  }
  return {
    lo: HIST_LO,
    step: HIST_STEP,
    counts,
    avg: trades.length > 0 ? trades.reduce((s, t) => s + t.rMultiple, 0) / trades.length : 0,
    n: trades.length,
  };
}

/* ----------------------------------------------------------------------------
   Skill-list trend bars
   -------------------------------------------------------------------------- */

/**
 * Accuracy over `k` sequential slices of a node's rating-moving decisions —
 * the dashboard's tiny trend bars (each bar = measured accuracy of that
 * quarter of the node's rated history). Null until the node has ≥ k reps.
 */
export function accuracyBars(nodeDecisions: DecisionRecord[], k = 4): number[] | null {
  const rated = nodeDecisions
    .filter((d) => d.mode === 'rated' || d.mode === 'checkpoint')
    .sort((a, b) => a.at - b.at);
  if (rated.length < k) return null;
  const out: number[] = [];
  for (let i = 0; i < k; i++) {
    const s = Math.floor((i * rated.length) / k);
    const e = Math.floor(((i + 1) * rated.length) / k);
    const slice = rated.slice(s, e);
    out.push(slice.filter((d) => d.verdict.correct).length / slice.length);
  }
  return out;
}

/** Full calibration ledger of a decision list (re-export convenience). */
export function calibrationLedger(decisions: DecisionRecord[]): CalibrationEntry[] {
  return brierLedger(decisions);
}
