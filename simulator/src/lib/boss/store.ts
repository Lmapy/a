/* ============================================================================
   @boss/store — boss-run persistence: the decision ledger rows (Dexie via
   @schedule/persist, mode 'boss'), the best-attempt / last-run summaries
   (trivial prefs), and the "one thing to drill" Woodpecker queue hook.

   Ledger contract (GDD §9): every graded boss decision is recorded with
   (seed, paramsVersion) and — for trade decisions — the executed TradeFill
   (setup, declared regime, realized R). The Stats R-histogram and expectancy
   ledger read decisions where mode === 'boss' && trade != null.

   Attribution (GDD §6 "every decision is attributed back to its atomic
   skill's stat pool"): order-ticket decisions → 'regime-gate'; KILL/HOLD
   reflexes → 'one-timeframing-buzzer'. Boss mode never moves Glicko rating
   (movesRating('boss') === false).

   Pure TS — zero DOM/Svelte imports.
   ========================================================================== */

import type { Answer, DecisionRecord, Verdict } from '../types';
import { flush, getPref, record, setPref, store } from '../schedule/persist';
import { enqueueMiss } from '../schedule/queue';
import type { BossAction, DeclaredRead, ReflexAnswer } from './engine';

/** Playbook setup names for the ticket actions (guide Part IV). */
export const SETUP_OF_ACTION: Record<Exclude<BossAction, 'stand-aside'>, string> = {
  fade: 'va-edge-fade', // §4.1 responsive fade at the value-area edge
  'go-with': 'initiative-breakout', // §4.2 initiative breakout of value
};

/* -------------------------------------------------------------- ledger rows */

export interface TradeDecisionRow {
  seed: string;
  paramsVersion: string;
  decisionIndex: number;
  bar: number;
  action: BossAction;
  read: DeclaredRead;
  latencyMs: number;
  verdict: Verdict;
  /** Realized R of the trade; null for stand-aside. */
  rPnl: number | null;
  at: number;
}

export interface ReflexRow {
  seed: string;
  paramsVersion: string;
  bar: number;
  answer: ReflexAnswer | null;
  latencyMs: number;
  verdict: Verdict;
  at: number;
}

/**
 * Plain snapshot of a Verdict: callers may hand us Svelte $state proxies,
 * which fail IndexedDB's structured clone (see drills/block.svelte.ts).
 */
function snapshotVerdict(v: Verdict): Verdict {
  return { ...v, refs: [...v.refs] };
}

/** Build the ledger row for one order-ticket decision. */
export function decisionRecordOf(row: TradeDecisionRow): DecisionRecord {
  const answer: Answer = {
    itemId: `boss1:${row.seed}:${row.decisionIndex}`,
    choice: `${row.action}·${row.read}`,
    confidence: null,
    latencyMs: row.latencyMs,
  };
  return {
    decisionId: `boss1:${row.seed}:${row.at}:${row.decisionIndex}`,
    nodeId: 'regime-gate',
    drillId: 'regime-gate',
    seed: row.seed,
    paramsVersion: row.paramsVersion,
    answer,
    verdict: snapshotVerdict(row.verdict),
    at: row.at,
    mode: 'boss',
    ...(row.action !== 'stand-aside' && row.rPnl !== null
      ? {
          trade: {
            setup: SETUP_OF_ACTION[row.action],
            regime: row.read,
            rMultiple: row.rPnl,
          },
        }
      : {}),
  };
}

/** Build the ledger row for one KILL/HOLD reflex. */
export function reflexRecordOf(row: ReflexRow): DecisionRecord {
  return {
    decisionId: `boss1:${row.seed}:${row.at}:reflex:${row.bar}`,
    nodeId: 'one-timeframing-buzzer',
    drillId: 'one-timeframing-buzzer',
    seed: row.seed,
    paramsVersion: row.paramsVersion,
    answer: {
      itemId: `boss1:${row.seed}:reflex:${row.bar}`,
      choice: row.answer ?? 'SLEPT',
      confidence: null,
      latencyMs: row.latencyMs,
    },
    verdict: snapshotVerdict(row.verdict),
    at: row.at,
    mode: 'boss',
  };
}

/** Record every row of a finished run into the ledger and flush (idle path). */
export async function persistRun(rows: DecisionRecord[]): Promise<void> {
  for (const r of rows) record(r);
  try {
    await flush();
  } catch {
    /* persistence is best-effort in v1 (private mode safe) */
  }
}

/* ---------------------------------------------------- run summaries (prefs) */

/** A finished-run summary (intro "best attempt" line + "last run" tape). */
export interface BossRunSummary {
  at: number;
  seed: string;
  paramsVersion: string;
  readScore: number;
  pnlR: number;
  passed: boolean;
  breach: 'daily' | 'trailing' | null;
  /** Actions taken, by decision index ('skip' = decision never reached). */
  actions: (BossAction | 'skip')[];
}

const BEST_KEY = 'boss1:best';
const LAST_KEY = 'boss1:last';

function parseSummary(raw: string): BossRunSummary | null {
  try {
    const v = JSON.parse(raw) as BossRunSummary;
    return typeof v?.seed === 'string' && typeof v?.readScore === 'number' ? v : null;
  } catch {
    return null;
  }
}

export function bestAttempt(): BossRunSummary | null {
  return parseSummary(getPref(BEST_KEY, 'null'));
}

export function lastRun(): BossRunSummary | null {
  return parseSummary(getPref(LAST_KEY, 'null'));
}

/** Better run = passed beats not-passed, then higher Read Score, then P&L. */
export function isBetterAttempt(a: BossRunSummary, b: BossRunSummary | null): boolean {
  if (b === null) return true;
  if (a.passed !== b.passed) return a.passed;
  if (a.readScore !== b.readScore) return a.readScore > b.readScore;
  return a.pnlR > b.pnlR;
}

/** Store a finished run: always as last run, as best when it beats the best. */
export function saveRun(summary: BossRunSummary): void {
  setPref(LAST_KEY, JSON.stringify(summary));
  if (isBetterAttempt(summary, bestAttempt())) setPref(BEST_KEY, JSON.stringify(summary));
}

/* ------------------------------------------------- "one thing to drill" */

/**
 * Queue tomorrow's Woodpecker with the run's session (GDD §6: the debrief
 * ends with a "one thing to drill" button). The regime-gate node owns the
 * fade-on-trend-day read; Woodpecker serves a sibling, never the original.
 */
export async function queueOneThingToDrill(
  seed: string,
  paramsVersion: string,
  now: number,
): Promise<void> {
  const item = enqueueMiss(`boss1:${seed}:${now}`, 'regime-gate', 'regime-gate', seed, paramsVersion, now);
  try {
    await store().putQueueItems([item]);
  } catch {
    /* best-effort */
  }
}
