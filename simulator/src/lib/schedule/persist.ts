/* ============================================================================
   @schedule/persist — Dexie 4 / IndexedDB persistence of the decision ledger,
   rating states, and the spaced-repetition queue (GDD §9).

   Ledger contract: every scored decision carries (seed, paramsVersion) so any
   verdict is re-derivable forever. NEVER await a write inside the verdict
   path — queue and flush on idle (tools doc).

   Dexie is environment-agnostic (works in Node tests via fake-indexeddb if
   needed); this module stays DOM-free. localStorage is used ONLY for trivial
   prefs and only via the guarded helpers below.

   STATUS: schema defined and functional; higher-level query helpers pending.
   Owner: schedule team.
   ========================================================================== */

import Dexie, { type Table } from 'dexie';
import type { DecisionRecord, QueueItem, RatingState } from '../types';

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

/** Flush buffered decisions to IndexedDB. Call on idle / round end. */
export async function flush(): Promise<void> {
  if (pending.length === 0) return;
  const batch = pending.splice(0, pending.length);
  await db().decisions.bulkPut(batch);
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
