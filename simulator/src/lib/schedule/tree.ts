/* ============================================================================
   @schedule/tree — the skill-tree model (GDD §3): tier layout, prerequisite
   DAG, and the derivation of every node's mastery state from the persisted
   decision ledger. The Home screen renders exactly what this module derives —
   nothing on the tree is hand-placed.

   Derivations (all measurement, per GDD §3 "Mastery mechanics"):
   · mastered      — a passed delayed checkpoint exists in the ledger
                     (consecutive sets of CHECKPOINT_SIZE checkpoint-mode
                     decisions, ≥8/10, zero-cardinal where the drill requires)
   · rusty         — mastered AND spaced-review accuracy (woodpecker/warmup
                     modes, last REVIEW_WINDOW reps) below RUSTY_THRESHOLD
   · checkpoint-armed — not mastered AND isGateArmed() on the rolling in-drill
                     stats (accuracy AND latency, GDD §4 criteria); the
                     checkpoint becomes takable at armCheckpoint(lastRatedAt)
                     — "no sooner than the next calendar day"
   · learning      — unlocked with practice history (or freshly unlocked)
   · locked        — prerequisite not yet mastered and no history

   Rating replay: Glicko-2 states are re-derived from the ledger by replaying
   every rating-moving decision through updateRating against the PERSISTED
   item rating (DecisionRecord.itemRating — the exact opponent the live loop
   used, written on every rated rep). Replayed history therefore matches the
   loop's own update path one-for-one: one computation, two surfaces. Only
   legacy rows without the field fall back to reconstructing the opponent
   from the deterministic knob↔rating mapping (the same selectKnobs path the
   loop uses) — a ≤ few-points reconstruction bound, never applied to the
   headline rating (always the loop's persisted RatingState).

   Pure TS — zero DOM/Svelte imports.
   ========================================================================== */

import type { DecisionRecord, DrillId, NodeState, RatingState } from '../types';
import { initialRating, ratingWeight, targetItemRating, updateRating } from './glicko';
import {
  CHECKPOINT_SIZE,
  ITEM_RD,
  MASTERY_CRITERIA,
  armCheckpoint,
  checkpointRetryAt,
  isGateArmed,
  isRusty,
  itemRatingFromKnobs,
  knobsForRating,
  movesRating,
  passesCheckpoint,
  startOfDay,
} from './queue';
import type { NodeStats } from './queue';
import { brierLedger, rollingBrier } from './persist';

/* ----------------------------------------------------------------------------
   Tier layout + prerequisite DAG (GDD §3 tree, v1 node set)
   -------------------------------------------------------------------------- */

/** One node of the rendered tree. */
export interface TreeNodeDef {
  id: DrillId;
  /** Display name (mockup screen-skill-tree.html naming). */
  name: string;
}

/** One tier (subway line) of the tree. */
export interface TreeTier {
  key: 'T0' | 'T1' | 'T2' | 'T3';
  title: string;
  nodes: TreeNodeDef[];
}

/** The v1 tree, tier by tier (GDD §3; drill ids are the node ids). */
export const TREE_TIERS: TreeTier[] = [
  {
    key: 'T0',
    title: 'READ THE MAP',
    nodes: [
      { id: 'poc-va-snap', name: 'POC / VA Snap' },
      { id: 'hvn-lvn-marker', name: 'HVN / LVN Marker' },
      { id: 'excess-or-poor', name: 'Excess vs Poor' },
      { id: 'shape-alphabet', name: 'Shape Alphabet' },
    ],
  },
  {
    key: 'T1',
    title: 'SESSION STRUCTURE',
    nodes: [
      { id: 'open-type-ladder', name: 'Open Type' },
      { id: 'one-timeframing-buzzer', name: 'One-Timeframing' },
      { id: 'acceptance-clock', name: 'Acceptance Clock' },
    ],
  },
  {
    key: 'T2',
    title: 'REGIME & CALIBRATION',
    nodes: [
      { id: 'regime-gate', name: 'Regime Gate' },
      { id: 'calibration-range', name: 'Calibration Range' },
    ],
  },
  {
    key: 'T3',
    title: 'PLAYBOOK',
    nodes: [{ id: 'setup-picker', name: 'Setup Picker' }],
  },
];

/** Locked future tiers rendered as ghost silhouettes (GDD §3 T4/T5). */
export const SILHOUETTE_TIERS: { key: string; title: string; boxes: number }[] = [
  { key: 'T4', title: 'ORDER FLOW', boxes: 4 },
  { key: 'T5', title: 'INTEGRATION', boxes: 3 },
];

/** All tree node ids in tier order. */
export const TREE_ORDER: DrillId[] = TREE_TIERS.flatMap((t) => t.nodes.map((n) => n.id));

/**
 * Prerequisite DAG (GDD §3 "Unlock = prerequisite mastered"; branches per the
 * §8 subway wireframe: T1 branches off excess-vs-poor, so Open Type can be in
 * progress while Shape Alphabet's checkpoint is still armed).
 */
export const PREREQUISITE: Partial<Record<DrillId, DrillId>> = {
  'hvn-lvn-marker': 'poc-va-snap',
  'excess-or-poor': 'hvn-lvn-marker',
  'shape-alphabet': 'excess-or-poor',
  'open-type-ladder': 'excess-or-poor',
  'one-timeframing-buzzer': 'open-type-ladder',
  'acceptance-clock': 'one-timeframing-buzzer',
  'regime-gate': 'acceptance-clock',
  'calibration-range': 'regime-gate',
  'setup-picker': 'calibration-range',
};

/**
 * The drills actually wired into the v1 loop (must mirror drills/items
 * LOOP_DRILLS — asserted by tree.test.ts; kept as a local constant so the
 * schedule lib stays free of app-layer imports). Nodes NOT in this set are
 * v1.1 content (GDD §11 cut list: e.g. HVN/LVN Marker folds into VA Snap
 * feedback) — they render on the tree but can neither be drilled nor
 * mastered, so they must never gate progression.
 */
export const PLAYABLE_DRILLS: ReadonlyArray<DrillId> = [
  'poc-va-snap',
  'excess-or-poor',
  'shape-alphabet',
  'open-type-ladder',
  'regime-gate',
];

/** True when the node's drill is playable in the v1 loop. */
export function isPlayable(id: DrillId): boolean {
  return PLAYABLE_DRILLS.includes(id);
}

/**
 * The nearest PLAYABLE ancestor in the prerequisite chain — the node whose
 * mastery actually gates `id` in v1. Unplayable v1.1 nodes are skipped
 * (otherwise the tree would dead-end at the first unshipped drill and the
 * open-type round could never be reached through play).
 */
export function effectivePrereq(id: DrillId): DrillId | undefined {
  let p = PREREQUISITE[id];
  while (p !== undefined && !isPlayable(p)) p = PREREQUISITE[p];
  return p;
}

/* ----------------------------------------------------------------------------
   Windows
   -------------------------------------------------------------------------- */

/** Rolling window of rated reps feeding the mastery gate (GDD §3). */
export const GATE_WINDOW = 20;
/** Rolling window of spaced-review reps feeding the rusty demotion (GDD §3). */
export const REVIEW_WINDOW = 10;

/** Modes that constitute practice evidence on the tree (boss never unlocks). */
const PRACTICE_MODES: ReadonlyArray<DecisionRecord['mode']> = [
  'rated',
  'checkpoint',
  'woodpecker',
  'warmup',
  'rush',
  'streak',
  'calibration',
];

/* ----------------------------------------------------------------------------
   Rating replay (deltas / history / trend bars)
   -------------------------------------------------------------------------- */

/** One point of a node's replayed rating history (after the decision at `at`). */
export interface RatingPoint {
  at: number;
  rating: number;
}

/** Replayed Glicko state + history for every node present in the ledger. */
export interface RatingReplay {
  states: Record<string, RatingState>;
  history: Record<string, RatingPoint[]>;
}

/**
 * Fallback opponent for LEGACY ledger rows that lack the persisted
 * DecisionRecord.itemRating: the item rating the adaptive selector would
 * have served against `state` — the same deterministic knob path the drill
 * loop uses (selectKnobs → itemRatingFromKnobs).
 */
export function reconstructedItemRating(state: RatingState): number {
  return itemRatingFromKnobs(knobsForRating(targetItemRating(state)));
}

/**
 * Replay every rating-moving decision (GDD §5: only 'rated' and 'checkpoint')
 * through Glicko-2, oldest first. The opponent is the decision's PERSISTED
 * item rating (exactly what the live loop played against), so the replayed
 * trajectory equals the loop's own — displayed deltas can never contradict
 * the rating beside them. Cardinal errors carry 3× weight, exactly as the
 * loop applies them.
 */
export function replayRatings(decisions: DecisionRecord[]): RatingReplay {
  const states: Record<string, RatingState> = {};
  const history: Record<string, RatingPoint[]> = {};
  const sorted = decisions.slice().sort((a, b) => a.at - b.at);
  for (const d of sorted) {
    if (!movesRating(d.mode)) continue;
    const before = states[d.nodeId] ?? initialRating(d.nodeId);
    const after = updateRating(
      before,
      d.itemRating ?? reconstructedItemRating(before),
      ITEM_RD,
      d.verdict.score / 100,
      ratingWeight(d.verdict.cardinal),
    );
    states[d.nodeId] = after;
    (history[d.nodeId] ??= []).push({ at: d.at, rating: after.rating });
  }
  return { states, history };
}

/**
 * Display-rating movement across the last calendar day that moved the node's
 * rating (0 when the node has no rating history). "Today's climb" on the tree
 * chip and the dashboard skill list's delta column.
 */
export function deltaOverLastActiveDay(points: RatingPoint[]): number {
  if (points.length === 0) return 0;
  const lastDay = startOfDay(points[points.length - 1].at);
  let before = initialRating('x').rating;
  let last = before;
  for (const p of points) {
    if (startOfDay(p.at) < lastDay) before = p.rating;
    last = p.rating;
  }
  return Math.round(last) - Math.round(before);
}

/* ----------------------------------------------------------------------------
   Node progress derivation
   -------------------------------------------------------------------------- */

/** Everything the tree needs to render one node. */
export interface NodeProgress {
  id: DrillId;
  state: NodeState;
  /** Practice decisions on the node (boss mode excluded). */
  nDecisions: number;
  /** Rolling accuracy over the last GATE_WINDOW rated/checkpoint reps; null when unplayed. */
  accuracy: number | null;
  /** Median latency over the same window; null when unplayed. */
  medianLatencyMs: number | null;
  /** True when the in-drill gate criteria are met (GDD §3 arming). */
  gateArmed: boolean;
  /** Earliest epoch-ms the armed checkpoint may be taken; null unless armed. */
  checkpointAvailableAt: number | null;
  /** Spaced-review accuracy over the last REVIEW_WINDOW woodpecker/warmup reps. */
  reviewAccuracy: number | null;
  /** Epoch ms of the passing checkpoint; null when not mastered. */
  masteredAt: number | null;
  /**
   * Epoch ms of the latest FAILED checkpoint set (GDD §3 failure UX: "retry
   * available in 2 days"); null when no checkpoint has been failed.
   */
  failedCheckpointAt: number | null;
}

function median(xs: number[]): number {
  const s = xs.slice().sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Epoch ms of the latest passing checkpoint set, or null (GDD §3 mastery). */
export function masteredAtOf(drillId: DrillId, nodeDecisions: DecisionRecord[]): number | null {
  const cps = nodeDecisions
    .filter((d) => d.mode === 'checkpoint')
    .sort((a, b) => a.at - b.at);
  let masteredAt: number | null = null;
  for (let i = 0; i + CHECKPOINT_SIZE <= cps.length; i += CHECKPOINT_SIZE) {
    const set = cps.slice(i, i + CHECKPOINT_SIZE);
    const nCorrect = set.filter((d) => d.verdict.correct).length;
    const cardinals = set.filter((d) => d.verdict.cardinal).length;
    if (passesCheckpoint(drillId, nCorrect, CHECKPOINT_SIZE, cardinals)) {
      masteredAt = set[set.length - 1].at;
    }
  }
  return masteredAt;
}

/** Epoch ms of the latest FAILED checkpoint set, or null (GDD §3 retry UX). */
export function lastFailedCheckpointAt(
  drillId: DrillId,
  nodeDecisions: DecisionRecord[],
): number | null {
  const cps = nodeDecisions
    .filter((d) => d.mode === 'checkpoint')
    .sort((a, b) => a.at - b.at);
  let failedAt: number | null = null;
  for (let i = 0; i + CHECKPOINT_SIZE <= cps.length; i += CHECKPOINT_SIZE) {
    const set = cps.slice(i, i + CHECKPOINT_SIZE);
    const nCorrect = set.filter((d) => d.verdict.correct).length;
    const cardinals = set.filter((d) => d.verdict.cardinal).length;
    if (!passesCheckpoint(drillId, nCorrect, CHECKPOINT_SIZE, cardinals)) {
      failedAt = set[set.length - 1].at;
    }
  }
  return failedAt;
}

/** Rolling in-drill stats presented to the mastery gate (GDD §3). */
export function gateStatsOf(nodeDecisions: DecisionRecord[]): NodeStats | null {
  const rated = nodeDecisions
    .filter((d) => movesRating(d.mode))
    .sort((a, b) => a.at - b.at)
    .slice(-GATE_WINDOW);
  if (rated.length === 0) return null;
  const entries = brierLedger(nodeDecisions);
  return {
    accuracy: rated.filter((d) => d.verdict.correct).length / rated.length,
    medianLatencyMs: median(rated.map((d) => d.answer.latencyMs)),
    rollingBrier: rollingBrier(entries),
    cardinalErrors: rated.filter((d) => d.verdict.cardinal).length,
    // NO-TRADE compliance needs the item's ground truth, which the ledger does
    // not carry — until Drill J persists it, the setup-picker gate cannot arm.
    noTradeCompliance: 0,
  };
}

/** Spaced-review accuracy over the last REVIEW_WINDOW review reps (GDD §3 rusty). */
export function reviewAccuracyOf(nodeDecisions: DecisionRecord[]): number | null {
  const reviews = nodeDecisions
    .filter((d) => d.mode === 'woodpecker' || d.mode === 'warmup')
    .sort((a, b) => a.at - b.at)
    .slice(-REVIEW_WINDOW);
  if (reviews.length === 0) return null;
  return reviews.filter((d) => d.verdict.correct).length / reviews.length;
}

/** The derived tree: per-node progress plus the "tap next" active node. */
export interface TreeDerivation {
  nodes: Record<string, NodeProgress>;
  /** First unlocked, unmastered, unarmed node in tier order (the active row). */
  activeNodeId: DrillId | null;
}

/**
 * Derive every tree node's state from the ledger (pure measurement — see the
 * module header for the exact rules).
 */
export function deriveTree(decisions: DecisionRecord[], now: number): TreeDerivation {
  const byNode = new Map<string, DecisionRecord[]>();
  for (const d of decisions) {
    if (!PRACTICE_MODES.includes(d.mode)) continue;
    const arr = byNode.get(d.nodeId);
    if (arr) arr.push(d);
    else byNode.set(d.nodeId, [d]);
  }

  const nodes: Record<string, NodeProgress> = {};
  for (const id of TREE_ORDER) {
    const nd = (byNode.get(id) ?? []).slice().sort((a, b) => a.at - b.at);
    const masteredAt = masteredAtOf(id, nd);
    const failedCheckpointAt = lastFailedCheckpointAt(id, nd);
    const stats = gateStatsOf(nd);
    const gateArmed = stats !== null && isGateArmed(id, stats);
    const lastRatedAt = nd.filter((d) => movesRating(d.mode)).at(-1)?.at ?? null;
    const reviewAccuracy = reviewAccuracyOf(nd);

    // Unplayable v1.1 nodes never gate progression: unlock checks the nearest
    // PLAYABLE ancestor instead (effectivePrereq).
    const prereq = effectivePrereq(id);
    const prereqMastered =
      prereq === undefined || (nodes[prereq] !== undefined && nodes[prereq].masteredAt !== null);
    const unlocked = prereqMastered || nd.length > 0;

    let state: NodeState;
    if (masteredAt !== null) {
      state = reviewAccuracy !== null && isRusty(reviewAccuracy) ? 'rusty' : 'mastered';
    } else if (!unlocked || (!isPlayable(id) && nd.length === 0)) {
      // v1.1 nodes with no ledger history stay visually locked ("ships v1.1")
      // rather than presenting an unplayable 'learning' row.
      state = 'locked';
    } else if (gateArmed) {
      state = 'checkpoint-armed';
    } else {
      state = 'learning';
    }

    // Checkpoint availability: next calendar day after the last rated rep
    // (GDD §3), pushed out to local-midnight + 2 days after a failed attempt
    // (GDD §3 "checkpoint retry available in 2 days").
    const availableAt =
      state === 'checkpoint-armed' && lastRatedAt !== null
        ? Math.max(
            armCheckpoint(lastRatedAt),
            failedCheckpointAt !== null ? checkpointRetryAt(failedCheckpointAt) : 0,
          )
        : null;

    nodes[id] = {
      id,
      state,
      nDecisions: nd.length,
      accuracy: stats?.accuracy ?? null,
      medianLatencyMs: stats?.medianLatencyMs ?? null,
      gateArmed,
      checkpointAvailableAt: availableAt,
      reviewAccuracy,
      masteredAt,
      failedCheckpointAt,
    };
  }

  const activeNodeId =
    TREE_ORDER.find((id) => nodes[id].state === 'learning' && isPlayable(id)) ?? null;
  return { nodes, activeNodeId };
}

/** Gate accuracy criterion of a node, for "gate arms at 85%" copy. */
export function gateAccuracyOf(id: DrillId): number {
  return MASTERY_CRITERIA[id].minAccuracy;
}

/** Tier mastery meta line: "3/4 MASTERED" (rusty is demoted — not counted). */
export function tierMastery(tier: TreeTier, nodes: Record<string, NodeProgress>): {
  mastered: number;
  total: number;
} {
  const mastered = tier.nodes.filter((n) => nodes[n.id]?.state === 'mastered').length;
  return { mastered, total: tier.nodes.length };
}
