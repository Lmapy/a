/* ============================================================================
   @schedule/persist — persistence of the decision ledger, rating states, and
   the spaced-repetition queue (GDD §9), plus the calibration/Brier ledger
   (GDD §5) consumed by the Stats screen.

   Storage: Dexie 4 / IndexedDB in the browser; a functionally identical
   in-memory store is the automatic fallback where IndexedDB is unavailable
   (Node/Vitest) and can be forced with useMemoryStore() in tests.

   Ledger contract: every scored decision carries (seed, paramsVersion) so any
   verdict is re-derivable forever. NEVER await a write inside the verdict
   path — record() is synchronous; flush() on idle/round end.

   Pure of DOM/Svelte imports; localStorage is used ONLY for trivial prefs and
   only via the guarded helpers below.
   ========================================================================== */

import Dexie, { type Table } from 'dexie';
import type { Confidence, DecisionRecord, QueueItem, RatingState } from '../types';

/** Current ledger schema version. Bump + migrate, never mutate in place. */
export const DB_VERSION = 1;

/** The Auction's local database. */
export class AuctionDB extends Dexie {
  /** Every scored decision, re-derivable via (seed, paramsVersion). */
  decisions!: Table<DecisionRecord, string>;
  /** Per-node Glicko-2 state. */
  ratings!: Table<RatingState, string>;
  /** Spaced-repetition queue entries. */
  queue!: Table<QueueItem, string>;

  constructor(name = 'the-auction') {
    super(name);
    this.version(DB_VERSION).stores({
      // primary key, then indexed fields
      decisions: 'decisionId, nodeId, drillId, at, mode',
      ratings: 'nodeId',
      queue: 'id, nodeId, dueAt',
    });
  }
}

let _db: AuctionDB | null = null;

/** Lazily opened singleton (so pure-TS tests can import without touching IDB). */
export function db(): AuctionDB {
  if (!_db) _db = new AuctionDB();
  return _db;
}

/* ----------------------------------------------------------------------------
   Store abstraction: Dexie in the browser, in-memory in Node/tests
   -------------------------------------------------------------------------- */

/** The persistence surface the app uses — identical for Dexie and memory. */
export interface LedgerStore {
  putDecisions(rows: DecisionRecord[]): Promise<void>;
  decisionsForNode(nodeId: string): Promise<DecisionRecord[]>;
  allDecisions(): Promise<DecisionRecord[]>;
  putRating(rating: RatingState): Promise<void>;
  getRating(nodeId: string): Promise<RatingState | undefined>;
  allRatings(): Promise<RatingState[]>;
  putQueueItems(items: QueueItem[]): Promise<void>;
  removeQueueItem(id: string): Promise<void>;
  allQueueItems(): Promise<QueueItem[]>;
  clear(): Promise<void>;
}

/** In-memory LedgerStore — Node/Vitest fallback, byte-compatible semantics. */
export class MemoryStore implements LedgerStore {
  private decisions = new Map<string, DecisionRecord>();
  private ratings = new Map<string, RatingState>();
  private queue = new Map<string, QueueItem>();

  async putDecisions(rows: DecisionRecord[]): Promise<void> {
    for (const r of rows) this.decisions.set(r.decisionId, r);
  }
  async decisionsForNode(nodeId: string): Promise<DecisionRecord[]> {
    return [...this.decisions.values()].filter((d) => d.nodeId === nodeId);
  }
  async allDecisions(): Promise<DecisionRecord[]> {
    return [...this.decisions.values()];
  }
  async putRating(rating: RatingState): Promise<void> {
    this.ratings.set(rating.nodeId, rating);
  }
  async getRating(nodeId: string): Promise<RatingState | undefined> {
    return this.ratings.get(nodeId);
  }
  async allRatings(): Promise<RatingState[]> {
    return [...this.ratings.values()];
  }
  async putQueueItems(items: QueueItem[]): Promise<void> {
    for (const q of items) this.queue.set(q.id, q);
  }
  async removeQueueItem(id: string): Promise<void> {
    this.queue.delete(id);
  }
  async allQueueItems(): Promise<QueueItem[]> {
    return [...this.queue.values()];
  }
  async clear(): Promise<void> {
    this.decisions.clear();
    this.ratings.clear();
    this.queue.clear();
  }
}

/** Dexie-backed LedgerStore (the browser path). */
class DexieStore implements LedgerStore {
  constructor(private readonly d: AuctionDB) {}

  async putDecisions(rows: DecisionRecord[]): Promise<void> {
    await this.d.decisions.bulkPut(rows);
  }
  async decisionsForNode(nodeId: string): Promise<DecisionRecord[]> {
    return this.d.decisions.where('nodeId').equals(nodeId).toArray();
  }
  async allDecisions(): Promise<DecisionRecord[]> {
    return this.d.decisions.toArray();
  }
  async putRating(rating: RatingState): Promise<void> {
    await this.d.ratings.put(rating);
  }
  async getRating(nodeId: string): Promise<RatingState | undefined> {
    return this.d.ratings.get(nodeId);
  }
  async allRatings(): Promise<RatingState[]> {
    return this.d.ratings.toArray();
  }
  async putQueueItems(items: QueueItem[]): Promise<void> {
    await this.d.queue.bulkPut(items);
  }
  async removeQueueItem(id: string): Promise<void> {
    await this.d.queue.delete(id);
  }
  async allQueueItems(): Promise<QueueItem[]> {
    return this.d.queue.toArray();
  }
  async clear(): Promise<void> {
    await Promise.all([this.d.decisions.clear(), this.d.ratings.clear(), this.d.queue.clear()]);
  }
}

let _store: LedgerStore | null = null;

/**
 * The active store: Dexie where IndexedDB exists, MemoryStore otherwise
 * (Node/Vitest). Resolution happens once, lazily.
 */
export function store(): LedgerStore {
  if (!_store) {
    _store =
      typeof indexedDB !== 'undefined' ? new DexieStore(db()) : new MemoryStore();
  }
  return _store;
}

/** Force an empty in-memory store (tests). Returns it for direct inspection. */
export function useMemoryStore(): MemoryStore {
  const m = new MemoryStore();
  _store = m;
  return m;
}

/**
 * Drop any forced store so the next store() call re-resolves to the default
 * (Dexie in the browser). Lets a ?demo=1 session leave the demo sandbox
 * without carrying the memory store into live play.
 */
export function useDefaultStore(): void {
  _store = null;
}

/* --- Write-behind buffer: verdict path calls record(), flush happens later -- */

const pending: DecisionRecord[] = [];

/** Buffer a decision without awaiting IO (call from the verdict path). */
export function record(decision: DecisionRecord): void {
  pending.push(decision);
}

/** Number of buffered, unflushed decisions (for tests / idle scheduling). */
export function pendingCount(): number {
  return pending.length;
}

/** Flush buffered decisions to the active store. Call on idle / round end. */
export async function flush(): Promise<void> {
  if (pending.length === 0) return;
  const batch = pending.splice(0, pending.length);
  await store().putDecisions(batch);
}

/* ----------------------------------------------------------------------------
   Brier / calibration ledger (GDD §5)
   -------------------------------------------------------------------------- */

/**
 * Confidence taps on binary drills fold into the calibration ledger as
 * implicit probabilities (GDD §4 Drill B / §5): sure→0.90, lean→0.75,
 * guess→0.55.
 */
export const CONFIDENCE_PROB: Record<Confidence, number> = {
  sure: 0.9,
  lean: 0.75,
  guess: 0.55,
};

/** Probability slider bounds: 5–95% cap, taught explicitly (GDD §5). */
export const P_MIN = 0.05;
export const P_MAX = 0.95;

/** Blocks of 25 for Bot Points and the base-rate reveal (GDD §5). */
export const BLOCK_SIZE = 25;

/** Rolling window for the Drill E mastery Brier (GDD §4: Brier ≤0.18 / 50). */
export const BRIER_WINDOW = 50;

/** Clamp a probability to the taught 5–95% band. */
export function clampProb(p: number): number {
  return Math.min(P_MAX, Math.max(P_MIN, p));
}

/** The implicit probability of a confidence tap. */
export function confidenceProb(c: Confidence): number {
  return CONFIDENCE_PROB[c];
}

/** Brier component (p − o)² for probability p on binary outcome o. */
export function brier(p: number, outcome: boolean): number {
  const o = outcome ? 1 : 0;
  return (p - o) * (p - o);
}

/** Brier component for a confidence tap on a binary drill (grader helper). */
export function brierFromConfidence(c: Confidence, correct: boolean): number {
  return brier(CONFIDENCE_PROB[c], correct);
}

/** One row of the calibration ledger. */
export interface CalibrationEntry {
  /** The probability the player effectively stated. */
  p: number;
  /** Did the event occur (was the confident answer correct)? */
  hit: boolean;
  /** Epoch ms of the decision. */
  at: number;
}

/**
 * Extract the calibration-ledger row from a scored decision, or null when the
 * decision carries no probability: p is the slider value (calibration-range)
 * or the confidence tap's implicit probability; the outcome is recovered from
 * the stored Brier component (brier = (p−1)² on a hit, p² on a miss — pick
 * the exact match).
 */
export function calibrationEntryOf(d: DecisionRecord): CalibrationEntry | null {
  const b = d.verdict.brier;
  if (b == null) return null;
  let p: number | null = null;
  if (d.drillId === 'calibration-range' && typeof d.answer.choice === 'number') {
    p = clampProb(d.answer.choice);
  } else if (d.answer.confidence) {
    p = CONFIDENCE_PROB[d.answer.confidence];
  }
  if (p == null) return null;
  const hit = Math.abs((p - 1) * (p - 1) - b) <= Math.abs(p * p - b);
  return { p, hit, at: d.at };
}

/** The full calibration ledger of a decision list, oldest first. */
export function brierLedger(decisions: DecisionRecord[]): CalibrationEntry[] {
  return decisions
    .slice()
    .sort((a, b) => a.at - b.at)
    .map(calibrationEntryOf)
    .filter((e): e is CalibrationEntry => e !== null);
}

/**
 * Mean Brier over the most recent `window` entries (Drill E mastery input);
 * null when the ledger is empty.
 */
export function rollingBrier(entries: CalibrationEntry[], window = BRIER_WINDOW): number | null {
  if (entries.length === 0) return null;
  const tail = entries.slice(-window);
  return tail.reduce((s, e) => s + brier(e.p, e.hit), 0) / tail.length;
}

/* --- Calibration bucketing (reliability curve, GDD §5 surface 2) ----------- */

/** One reliability-curve bucket: dots sized by n, meanP vs hitRate. */
export interface CalibrationBucket {
  /** Bucket lower probability edge (inclusive). */
  pLo: number;
  /** Bucket upper probability edge (exclusive; last bucket inclusive). */
  pHi: number;
  /** Judgments in the bucket. */
  n: number;
  /** Mean stated probability of the bucket (null when empty). */
  meanP: number | null;
  /** Observed hit rate of the bucket (null when empty). */
  hitRate: number | null;
}

/**
 * Bin the calibration ledger into `nBins` equal-width probability buckets for
 * the Stats reliability curve ("When you say 80%, it happens 64%").
 */
export function calibrationBuckets(entries: CalibrationEntry[], nBins = 10): CalibrationBucket[] {
  const buckets: CalibrationBucket[] = Array.from({ length: nBins }, (_, i) => ({
    pLo: i / nBins,
    pHi: (i + 1) / nBins,
    n: 0,
    meanP: null,
    hitRate: null,
  }));
  const sumP = new Array<number>(nBins).fill(0);
  const hits = new Array<number>(nBins).fill(0);
  for (const e of entries) {
    const i = Math.min(nBins - 1, Math.max(0, Math.floor(e.p * nBins)));
    buckets[i].n++;
    sumP[i] += e.p;
    if (e.hit) hits[i]++;
  }
  for (let i = 0; i < nBins; i++) {
    if (buckets[i].n > 0) {
      buckets[i].meanP = sumP[i] / buckets[i].n;
      buckets[i].hitRate = hits[i] / buckets[i].n;
    }
  }
  return buckets;
}

/* --- Bot Points (GDD §5 surface 1) ----------------------------------------- */

/** A ledger entry joined with its item pool's declared base rate. */
export interface BotComparableEntry extends CalibrationEntry {
  /** The pool's declared true base rate for the question — the bot's answer. */
  baseRate: number;
}

/**
 * Bot Points for a set of judgments: AP = Σ (Brier_bot − Brier_you) × 100,
 * where the bot always answers the pool's declared base rate (GDD §5).
 * Positive = beating the base-rate bot.
 */
export function botPoints(entries: BotComparableEntry[]): number {
  return entries.reduce(
    (s, e) => s + (brier(e.baseRate, e.hit) - brier(e.p, e.hit)) * 100,
    0,
  );
}

/**
 * Bot Points per completed block of `blockSize` (default 25 — the block
 * reveal cadence). A trailing partial block is not scored.
 */
export function blockBotPoints(entries: BotComparableEntry[], blockSize = BLOCK_SIZE): number[] {
  const out: number[] = [];
  for (let i = 0; i + blockSize <= entries.length; i += blockSize) {
    out.push(botPoints(entries.slice(i, i + blockSize)));
  }
  return out;
}

/* --- Trivial prefs (mute, reduced-motion override) — localStorage only ------ */

const PREF_PREFIX = 'auction:pref:';

/** Read a trivial pref; returns fallback when storage is unavailable (Node). */
export function getPref(key: string, fallback: string): string {
  try {
    return globalThis.localStorage?.getItem(PREF_PREFIX + key) ?? fallback;
  } catch {
    return fallback;
  }
}

/** Write a trivial pref; no-op when storage is unavailable. */
export function setPref(key: string, value: string): void {
  try {
    globalThis.localStorage?.setItem(PREF_PREFIX + key, value);
  } catch {
    /* storage unavailable — prefs are best-effort */
  }
}
