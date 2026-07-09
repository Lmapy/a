/* ============================================================================
   @schedule/queue — spaced-repetition queue: fixed expanding intervals
   (1d / 3d / 7d, GDD §3), same-session retry, leech rule (8 lapses),
   Warm-Up assembly, Woodpecker sibling re-serve.

   Pure TS — zero DOM/Svelte imports.
   STATUS: interval math + due-selection implemented; Warm-Up assembler is a
   stub. Owner: schedule team.
   ========================================================================== */

import type { DrillId, QueueItem } from '../types';

/** Expanding re-queue intervals in ms (index = QueueItem.intervalIndex). */
export const INTERVALS_MS = [
  1 * 24 * 60 * 60 * 1000, // 1d
  3 * 24 * 60 * 60 * 1000, // 3d
  7 * 24 * 60 * 60 * 1000, // 7d
] as const;

/** Lapse count at which drilling stops and a micro-lesson serves instead (GDD §3). */
export const LEECH_THRESHOLD = 8;

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

/** The assembled Daily Warm-Up plan (GDD §3 mode 8). */
export interface WarmUpPlan {
  /** Up to 2 due Woodpecker entries (sibling re-serve). */
  woodpecker: QueueItem[];
  /** Node id of the new-skill block, or null when no node is in 'learning'. */
  newSkillNodeId: string | null;
  /** Node ids of the interleaved mixed block. */
  mixedNodeIds: string[];
}

/**
 * Assemble the ~3-minute Daily Warm-Up: 2 due misses + 1 new-skill block +
 * 1 interleaved mixed block, always adaptive difficulty (unfarmable streak).
 * STUB: takes the first 2 due items and echoes the inputs. Owner: schedule team.
 */
export function assembleWarmUp(
  queue: QueueItem[],
  learningNodeIds: string[],
  masteredNodeIds: string[],
  now: number,
): WarmUpPlan {
  return {
    woodpecker: dueItems(queue, now).slice(0, 2),
    newSkillNodeId: learningNodeIds[0] ?? null,
    mixedNodeIds: masteredNodeIds.slice(0, 3),
  };
}
