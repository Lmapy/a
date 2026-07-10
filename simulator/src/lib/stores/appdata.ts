/* ============================================================================
   App data loader — hydrates the Home tree and Stats dashboard from the
   persisted ledger (Dexie), or from the deterministic demo dataset when the
   ?demo=1 flag is present (review mode: seeded through the same LedgerStore
   interface, into an in-memory store so the player's real ledger is never
   touched).

   Everything returned here is either a raw persisted row or a derivation
   computed by the engine libs (schedule/tree, stats/model) — the screens
   bind, they never compute.
   ========================================================================== */

import type { DecisionRecord, QueueItem, RatingState } from '../types';
import {
  flush,
  getPref,
  store,
  useDefaultStore,
  useMemoryStore,
} from '../schedule/persist';
import { deriveTree, replayRatings } from '../schedule/tree';
import type { RatingReplay, TreeDerivation } from '../schedule/tree';
import { tradesOf } from '../stats/model';
import type { SimTrade } from '../stats/model';
import { buildDemoDataset } from '../demo/demo';
import { player, V1_NODE_IDS } from './player.svelte';

/** Everything the Home + Stats screens bind to. */
export interface AppData {
  demo: boolean;
  now: number;
  /** All persisted decisions, oldest first. */
  decisions: DecisionRecord[];
  queue: QueueItem[];
  /** Persisted Glicko states (the loop's own writes — the display source of truth). */
  ratings: RatingState[];
  /** Boss/sim trade fills extracted from the ledger. */
  trades: SimTrade[];
  /** Derived tree states (schedule/tree). */
  tree: TreeDerivation;
  /** Glicko replay for deltas / history / trend bars. */
  replay: RatingReplay;
}

/** True when the current URL carries the ?demo=1 review flag (hash or search). */
export function demoFlag(): boolean {
  if (typeof window === 'undefined') return false;
  const h = window.location.hash;
  const q = h.indexOf('?');
  const hashQuery = new URLSearchParams(q >= 0 ? h.slice(q + 1) : '');
  const search = new URLSearchParams(window.location.search);
  return hashQuery.get('demo') === '1' || search.get('demo') === '1';
}

let demoSeeded = false;
let demoCache: Promise<AppData> | null = null;

/**
 * Load (and derive) everything the dashboard screens need. Live data is
 * re-read on every call (the ledger may have grown since the last drill
 * block); the demo dataset is seeded + derived once per demo entry.
 */
export function loadAppData(): Promise<AppData> {
  if (demoFlag()) {
    if (!demoCache) demoCache = load(true);
    return demoCache;
  }
  // leaving demo drops the sandbox so re-entry re-seeds cleanly
  demoCache = null;
  return load(false);
}

async function load(demo: boolean): Promise<AppData> {
  const now = Date.now();

  if (demo) {
    const m = useMemoryStore(); // sandbox: never mixes into the real ledger
    const ds = buildDemoDataset(now);
    await m.putDecisions(ds.decisions);
    for (const r of ds.ratings) await m.putRating(r);
    await m.putQueueItems(ds.queue);
    player.streak = { ...ds.streak };
    demoSeeded = true;
  } else {
    if (demoSeeded) {
      useDefaultStore(); // leave the demo sandbox
      demoSeeded = false;
    }
    const days = Number(getPref('streak-days', '0'));
    const freezes = Number(getPref('streak-freezes', '2'));
    player.streak = {
      days: Number.isFinite(days) ? days : 0,
      freezes: Number.isFinite(freezes) ? freezes : 2,
    };
  }

  await flush(); // drain any write-behind rows before reading
  const s = store();
  const [decisions, ratings, queue] = await Promise.all([
    s.allDecisions(),
    s.allRatings(),
    s.allQueueItems(),
  ]);
  decisions.sort((a, b) => a.at - b.at);

  const tree = deriveTree(decisions, now);
  const replay = replayRatings(decisions);

  // Hydrate the player store: persisted rating rows are the display source
  // of truth (written by the loop itself); the replay covers rows the loop
  // has not flushed yet. Node states come from the tree derivation.
  for (const id of V1_NODE_IDS) {
    const persisted = ratings.find((r) => r.nodeId === id) ?? replay.states[id];
    if (persisted) player.setRating({ ...persisted });
    player.setNodeState(id, tree.nodes[id]?.state ?? 'locked');
  }

  return { demo, now, decisions, queue, ratings, trades: tradesOf(decisions), tree, replay };
}
