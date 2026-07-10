/* ============================================================================
   @schedule/queue — the GDD scheduling rules:
   · spaced repetition of misses on fixed expanding intervals (1d/3d/7d, §3)
   · Woodpecker sibling re-serve (same seed family, nudged variant via the
     seeded PRNG's sibling derivation — never the identical item, §3 mode 4)
   · leech rule (8 lapses → micro-lesson, §3)
   · rating-matched item selection targeting ~85% expected success via
     deterministic knob↔rating mapping (§5 "item ratings derived
     deterministically from generator difficulty knobs")
   · 3×-weighted cardinal errors (§5, via glicko.ratingWeight)
   · delayed interleaved checkpoint arming — next calendar day, 10 items,
     40/60 mix, ≥8/10 (§3 mastery mechanics)
   · mastery gates: accuracy AND latency criteria per node (§3–4)
   · Daily Warm-Up assembly (§3 mode 8)

   Pure TS — zero DOM/Svelte imports; randomness only via @gen/prng.
   ========================================================================== */

import type { DecisionRecord, DrillId, QueueItem, RatingState } from '../types';
import { siblingSeed } from '../gen/prng';
import type { DifficultyKnobs } from '../gen/scripts';
import { GLICKO2_SCALE, expectedScore, targetItemRating, toMu } from './glicko';

/** Expanding re-queue intervals in ms (index = QueueItem.intervalIndex). */
export const INTERVALS_MS = [
  1 * 24 * 60 * 60 * 1000, // 1d
  3 * 24 * 60 * 60 * 1000, // 3d
  7 * 24 * 60 * 60 * 1000, // 7d
] as const;

const DAY_MS = 24 * 60 * 60 * 1000;

/** Lapse count at which drilling stops and a micro-lesson serves instead (GDD §3). */
export const LEECH_THRESHOLD = 8;

/** Mastered nodes whose spaced-review accuracy decays below this go rusty (GDD §3). */
export const RUSTY_THRESHOLD = 0.7;

/* ----------------------------------------------------------------------------
   Miss queue: enqueue / advance / due / leech
   -------------------------------------------------------------------------- */

/**
 * Create a queue entry for a fresh miss (interval 0 = due in 1 day).
 * Woodpecker will serve a SIBLING of this seed, never the identical item.
 */
export function enqueueMiss(
  id: string,
  nodeId: string,
  drillId: DrillId,
  seed: string,
  paramsVersion: string,
  now: number,
): QueueItem {
  return {
    id,
    nodeId,
    drillId,
    seed,
    paramsVersion,
    dueAt: now + INTERVALS_MS[0],
    intervalIndex: 0,
    lapses: 1,
  };
}

/**
 * Advance an entry after a review.
 * · correct → next expanding interval (or graduation: returns null after 7d).
 * · miss    → lapse++ and reset to interval 0.
 */
export function advance(item: QueueItem, correct: boolean, now: number): QueueItem | null {
  if (correct) {
    const next = item.intervalIndex + 1;
    if (next >= INTERVALS_MS.length) return null; // graduated out of the queue
    return { ...item, intervalIndex: next, dueAt: now + INTERVALS_MS[next] };
  }
  return {
    ...item,
    lapses: item.lapses + 1,
    intervalIndex: 0,
    dueAt: now + INTERVALS_MS[0],
  };
}

/** Entries due at `now`, soonest-due first. */
export function dueItems(queue: QueueItem[], now: number): QueueItem[] {
  return queue.filter((q) => q.dueAt <= now).sort((a, b) => a.dueAt - b.dueAt);
}

/** True when the leech rule fires: stop drilling, serve the micro-lesson. */
export function isLeech(item: QueueItem): boolean {
  return item.lapses >= LEECH_THRESHOLD;
}

/** True when a mastered node's spaced-review accuracy demotes it to rusty (GDD §3). */
export function isRusty(reviewAccuracy: number): boolean {
  return reviewAccuracy < RUSTY_THRESHOLD;
}

/* ----------------------------------------------------------------------------
   Woodpecker sibling re-serve (GDD §3 mode 4)
   -------------------------------------------------------------------------- */

/**
 * The seed Woodpecker serves for this queue entry: a SIBLING of the original
 * missed item — same session family, variant-nudged via the PRNG's sibling
 * derivation — never the identical seed (kills answer memorization).
 *
 * The variant index `lapses·3 + intervalIndex` is strictly increasing along
 * every possible review path (correct: intervalIndex+1; miss: lapses+1 with
 * intervalIndex→0 and intervalIndex ≤ 2), so consecutive serves of the same
 * entry always present distinct siblings.
 */
export function woodpeckerSeed(item: QueueItem): string {
  return siblingSeed(item.seed, item.lapses * INTERVALS_MS.length + item.intervalIndex);
}

/* ----------------------------------------------------------------------------
   Rating-matched item selection (GDD §5)
   -------------------------------------------------------------------------- */

/** Knob-derived item-rating scale: floor + span cover the ~800–2400 GDD band. */
export const KNOB_RATING_FLOOR = 800;
export const KNOB_RATING_SPAN = 1600;

/** Display-rating contribution of each knob at its hardest setting. */
const KNOB_WEIGHTS = {
  noiseGain: 480, // master signal-to-noise dial, 0→1
  nu: 240, // Student-t ν 8 (tame) → 4 (spiky)
  ambiguity: 360, // blend weight 0 → 0.5
  decoys: 240, // 0 → 2 decoy near-structures
  lvnDepth: 180, // 1 (deep valley) → 0 (shallow)
  jumpIntensity: 100, // news-jump multiplier 0 → 1
} as const;

/** RD assigned to knob-derived items (until player–item co-rating, GDD §5). */
export const ITEM_RD = 60;

/**
 * Deterministic item rating from generator difficulty knobs (GDD §5: "launch
 * with item ratings derived deterministically from generator difficulty
 * knobs"). Monotone in every knob's hardness; DEFAULT easy = 800, all knobs
 * maxed = 2400.
 */
export function itemRatingFromKnobs(knobs: DifficultyKnobs): number {
  const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
  const hardness =
    clamp01(knobs.noiseGain) * KNOB_WEIGHTS.noiseGain +
    clamp01((8 - knobs.nu) / 4) * KNOB_WEIGHTS.nu +
    clamp01(knobs.ambiguity / 0.5) * KNOB_WEIGHTS.ambiguity +
    clamp01(knobs.decoys / 2) * KNOB_WEIGHTS.decoys +
    clamp01(1 - knobs.lvnDepth) * KNOB_WEIGHTS.lvnDepth +
    clamp01(knobs.jumpIntensity) * KNOB_WEIGHTS.jumpIntensity;
  return KNOB_RATING_FLOOR + hardness;
}

/**
 * Inverse mapping: the knob pack whose derived item rating ≈ the requested
 * display rating (exact up to decoy-count rounding, ±60). All knobs scale
 * together along one difficulty axis.
 */
export function knobsForRating(rating: number): DifficultyKnobs {
  const f = Math.min(1, Math.max(0, (rating - KNOB_RATING_FLOOR) / KNOB_RATING_SPAN));
  return {
    noiseGain: f,
    nu: 8 - 4 * f,
    ambiguity: 0.5 * f,
    decoys: Math.round(2 * f),
    lvnDepth: 1 - f,
    jumpIntensity: f,
  };
}

/**
 * Per-node adaptive selection (GDD §5): the knob pack for the item whose
 * rating puts the player's expected score ≈ targetE (default the 85%
 * Wilson setpoint).
 */
export function selectKnobs(player: RatingState, targetE = 0.85): DifficultyKnobs {
  return knobsForRating(targetItemRating(player, targetE));
}

/** Expected success of a player against a knob pack (for tests/telemetry). */
export function expectedSuccessAgainst(player: RatingState, knobs: DifficultyKnobs): number {
  return expectedScore(toMu(player.rating), toMu(itemRatingFromKnobs(knobs)), ITEM_RD / GLICKO2_SCALE);
}

/* ----------------------------------------------------------------------------
   Rating-moving modes (GDD §5: Rush/Streak/Woodpecker/Warm-Up never move rating)
   -------------------------------------------------------------------------- */

/** The only modes whose decisions move Glicko rating. */
export const RATED_MODES: ReadonlyArray<DecisionRecord['mode']> = ['rated', 'checkpoint'];

/** True when a decision made in this mode moves the node's rating. */
export function movesRating(mode: DecisionRecord['mode']): boolean {
  return RATED_MODES.includes(mode);
}

/* ----------------------------------------------------------------------------
   Mastery gates (GDD §3 "Arming the gate" + per-drill criteria, §4)
   -------------------------------------------------------------------------- */

/** Per-node gate criteria. Latency values are placeholders, tuned empirically (GDD §3). */
export interface MasteryCriteria {
  /** Rolling accuracy required at target difficulty. */
  minAccuracy: number;
  /**
   * Median latency ceiling in ms. Replay drills express their bracket/minute
   * criteria in 8×-replay wall-clock ms (1 bracket = 30 market-min = 225s).
   */
  maxMedianLatencyMs: number;
  /** calibration-range only: rolling-50 Brier ceiling (GDD Drill E). */
  maxBrier?: number;
  /** regime-gate only: zero cardinal errors in the checkpoint set (GDD Drill I). */
  requireZeroCardinal?: boolean;
  /** setup-picker only: minimum NO-TRADE compliance (GDD Drill J). */
  minNoTradeCompliance?: number;
}

/** The GDD §4 mastery lines, one per drill. */
export const MASTERY_CRITERIA: Record<DrillId, MasteryCriteria> = {
  'poc-va-snap': { minAccuracy: 0.92, maxMedianLatencyMs: 2500 }, // 92% ±1 row, <2.5s
  'excess-or-poor': { minAccuracy: 0.85, maxMedianLatencyMs: 2000 }, // ≥85%, <2.0s
  'hvn-lvn-marker': { minAccuracy: 0.9, maxMedianLatencyMs: 2500 }, // 90% ±1 row, <2.5s
  'shape-alphabet': { minAccuracy: 0.85, maxMedianLatencyMs: 3000 }, // ≥85%, <3s
  'calibration-range': { minAccuracy: 0, maxMedianLatencyMs: Infinity, maxBrier: 0.18 }, // Brier ≤0.18 / rolling 50
  'open-type-ladder': { minAccuracy: 0.85, maxMedianLatencyMs: 150_000 }, // call before min 20 of replay (8× → 150s)
  'one-timeframing-buzzer': { minAccuracy: 0.85, maxMedianLatencyMs: 225_000 }, // ≤1 bracket (30 min @8× → 225s)
  'acceptance-clock': { minAccuracy: 0.85, maxMedianLatencyMs: 450_000 }, // mean lag ≤2 brackets
  'regime-gate': { minAccuracy: 0.85, maxMedianLatencyMs: 5000, requireZeroCardinal: true }, // <5s, zero cardinal
  'setup-picker': { minAccuracy: 0.85, maxMedianLatencyMs: 10_000, minNoTradeCompliance: 0.9 }, // ≥90% NO-TRADE
};

/** Rolling in-drill stats a node presents to the gate. */
export interface NodeStats {
  /** Rolling accuracy at target difficulty. */
  accuracy: number;
  /** Median answer latency in ms. */
  medianLatencyMs: number;
  /** Rolling-window Brier (calibration nodes); null/undefined when unmeasured. */
  rollingBrier?: number | null;
  /** Cardinal-error count in the evaluated window. */
  cardinalErrors?: number;
  /** NO-TRADE compliance rate (setup-picker). */
  noTradeCompliance?: number;
}

/**
 * Kellman gate arming (GDD §3): in-drill rolling accuracy AND median latency
 * must both clear the node's criterion (plus the drill's special clauses).
 * Arming never grants mastery — it only unlocks the delayed checkpoint.
 */
export function isGateArmed(drillId: DrillId, stats: NodeStats): boolean {
  const c = MASTERY_CRITERIA[drillId];
  if (stats.accuracy < c.minAccuracy) return false;
  if (!(stats.medianLatencyMs < c.maxMedianLatencyMs)) return false;
  if (c.maxBrier !== undefined) {
    if (stats.rollingBrier == null || stats.rollingBrier > c.maxBrier) return false;
  }
  if (c.requireZeroCardinal && (stats.cardinalErrors ?? 0) > 0) return false;
  if (c.minNoTradeCompliance !== undefined) {
    if ((stats.noTradeCompliance ?? 0) < c.minNoTradeCompliance) return false;
  }
  return true;
}

/* ----------------------------------------------------------------------------
   Delayed interleaved checkpoint (GDD §3 "Mastery")
   -------------------------------------------------------------------------- */

/** Checkpoint shape: 10 items, ≥8 correct, 40% target skill / 60% confusable siblings. */
export const CHECKPOINT_SIZE = 10;
export const CHECKPOINT_PASS_MIN = 8;
export const CHECKPOINT_TARGET_SHARE = 0.4;
/** Failed checkpoint: retry available in 2 days (GDD §3, no-punishment copy). */
export const CHECKPOINT_RETRY_DAYS = 2;

/** Local-midnight start of the calendar day containing `now`. */
export function startOfDay(now: number): number {
  const d = new Date(now);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

/** Local midnight beginning the NEXT calendar day after `now` (DST-safe). */
export function nextCalendarDayMs(now: number): number {
  const d = new Date(now);
  d.setHours(0, 0, 0, 0);
  d.setDate(d.getDate() + 1);
  return d.getTime();
}

/**
 * Arm the mastery checkpoint: available NO SOONER than the next calendar day
 * (GDD §3 — fights the illusion of competence; in-drill accuracy never grants
 * mastery). Returns the earliest epoch-ms the checkpoint may be taken.
 */
export function armCheckpoint(armedAt: number): number {
  return nextCalendarDayMs(armedAt);
}

/** Earliest retry time after a failed checkpoint: local midnight +2 calendar days. */
export function checkpointRetryAt(failedAt: number): number {
  const d = new Date(failedAt);
  d.setHours(0, 0, 0, 0);
  d.setDate(d.getDate() + CHECKPOINT_RETRY_DAYS);
  return d.getTime();
}

/** Item counts for a checkpoint set: 40% target skill, 60% confusable siblings. */
export function checkpointComposition(size = CHECKPOINT_SIZE): {
  targetItems: number;
  siblingItems: number;
} {
  const targetItems = Math.round(size * CHECKPOINT_TARGET_SHARE);
  return { targetItems, siblingItems: size - targetItems };
}

/**
 * Checkpoint verdict: ≥8/10 at gate difficulty; regime-gate additionally
 * requires zero cardinal errors in the set (GDD Drill I mastery line).
 */
export function passesCheckpoint(
  drillId: DrillId,
  nCorrect: number,
  size = CHECKPOINT_SIZE,
  cardinalErrors = 0,
): boolean {
  const passMin = Math.ceil((CHECKPOINT_PASS_MIN / CHECKPOINT_SIZE) * size);
  if (nCorrect < passMin) return false;
  if (MASTERY_CRITERIA[drillId].requireZeroCardinal && cardinalErrors > 0) return false;
  return true;
}

/* ----------------------------------------------------------------------------
   Daily Warm-Up (GDD §3 mode 8)
   -------------------------------------------------------------------------- */

/** The assembled Daily Warm-Up plan (GDD §3 mode 8). */
export interface WarmUpPlan {
  /** Up to 2 due Woodpecker entries (sibling re-serve). */
  woodpecker: QueueItem[];
  /** Node id of the new-skill block, or null when no node is in 'learning'. */
  newSkillNodeId: string | null;
  /** Node ids of the interleaved mixed block. */
  mixedNodeIds: string[];
  /** Due leeches: served as micro-lessons, never drilled (GDD §3 leech rule). */
  leeches: QueueItem[];
}

/**
 * Assemble the ~3-minute Daily Warm-Up: 2 due misses (soonest-due first,
 * leeches excluded — they get the micro-lesson instead) + 1 new-skill block +
 * 1 interleaved mixed block of up to 3 mastered nodes, rotated by calendar
 * day so consecutive Warm-Ups interleave different confusables. Difficulty is
 * always adaptive (selectKnobs), so the streak cannot be farmed.
 */
export function assembleWarmUp(
  queue: QueueItem[],
  learningNodeIds: string[],
  masteredNodeIds: string[],
  now: number,
): WarmUpPlan {
  const due = dueItems(queue, now);
  const leeches = due.filter(isLeech);
  const servable = due.filter((q) => !isLeech(q));

  // Deterministic daily rotation of the mixed block (interleaving variety).
  const n = masteredNodeIds.length;
  const rot = n > 0 ? Math.floor(now / DAY_MS) % n : 0;
  const mixedNodeIds = masteredNodeIds
    .slice(rot)
    .concat(masteredNodeIds.slice(0, rot))
    .slice(0, 3);

  return {
    woodpecker: servable.slice(0, 2),
    newSkillNodeId: learningNodeIds[0] ?? null,
    mixedNodeIds,
    leeches,
  };
}
