/* ============================================================================
   Player state store (runes-in-.svelte.ts).

   Holds the in-memory player state the UI binds to: per-node ratings, tree
   node states, streak meta, and prefs. Persistence flows through
   @schedule/persist (Dexie) — this store is the reactive projection, never
   the source of truth for the ledger.

   Teammates: extend via methods, keep fields serializable; hydrate() will be
   wired to Dexie by the schedule team.
   ========================================================================== */

import type { DrillId, NodeState, RatingState } from '../types';
import { initialRating } from '../schedule/glicko';

/** The v1 skill nodes shipped on the tree (T0–T3 partial, GDD §3/§11). */
export const V1_NODE_IDS: DrillId[] = [
  'poc-va-snap',
  'excess-or-poor',
  'hvn-lvn-marker',
  'shape-alphabet',
  'open-type-ladder',
  'one-timeframing-buzzer',
  'acceptance-clock',
  'regime-gate',
  'calibration-range',
  'setup-picker',
];

export interface StreakMeta {
  /** Consecutive completed Daily Warm-Ups (practice days, never accuracy). */
  days: number;
  /** Auto-granted freezes available. */
  freezes: number;
}

class PlayerStore {
  /** Per-node Glicko-2 rating states, keyed by node id. */
  ratings: Record<string, RatingState> = $state(
    Object.fromEntries(V1_NODE_IDS.map((id) => [id, initialRating(id)])),
  );

  /** Tree node states; only the first T0 node starts unlocked. */
  nodeStates: Record<string, NodeState> = $state(
    Object.fromEntries(
      V1_NODE_IDS.map((id, i) => [id, i === 0 ? 'learning' : 'locked'] as const),
    ),
  );

  /** Warm-Up streak meta (unfarmable by design, GDD §5). */
  streak: StreakMeta = $state({ days: 0, freezes: 2 });

  /** One-tap mute (persisted as a trivial pref). */
  muted: boolean = $state(false);

  /** Rating for a node, defaulting to a fresh 1500 state. */
  rating(nodeId: string): RatingState {
    return this.ratings[nodeId] ?? initialRating(nodeId);
  }

  /** Replace a node's rating state (called after a rated answer). */
  setRating(next: RatingState): void {
    this.ratings = { ...this.ratings, [next.nodeId]: next };
  }

  /** Move a node through the mastery lifecycle. */
  setNodeState(nodeId: string, state: NodeState): void {
    this.nodeStates = { ...this.nodeStates, [nodeId]: state };
  }
}

/** App-wide player store singleton. */
export const player = new PlayerStore();
