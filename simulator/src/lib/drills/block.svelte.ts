/* ============================================================================
   @drills/block — the drill-loop runtime (runes class): a block of 10 reps,
   state machine ITEM → ARMED → VERDICT → NEXT (GDD §9 client spec).

   Responsibilities:
   · serve items seeded via the schedule queue: due Woodpecker entries for
     this drill are consumed first (SIBLING seeds via woodpeckerSeed — never
     the identical item) and marked mode 'woodpecker' (they never move
     rating, GDD §5); remaining reps draw fresh rated seeds;
   · grade locally (<100ms — synchronous, zero IO on the verdict path);
   · record every rep through schedule/persist.record() (write-behind), move
     Glicko rating per rated rep (cardinal ⇒ 3× weight), queue misses on the
     1d/3d/7d schedule, advance consumed queue entries — all IO deferred to
     the block end (flush/putRating/putQueueItems);
   · pre-generate the next item during the feedback window so the rhythm
     never stutters (GDD §2 t+400ms row).

   Checkpoint mode (GDD §3 mastery / mode 6): opts.checkpoint runs the block
   as the delayed gate exam — 10 items, the target skill mixed 40/60 with
   confusable items from MASTERED playable siblings (checkpointComposition;
   all-target when nothing else is mastered yet), every rep recorded with
   mode 'checkpoint' on the TARGET node (masteredAtOf chunks these in sets of
   CHECKPOINT_SIZE), pass = passesCheckpoint (≥8/10, zero-cardinal where the
   drill requires). Juice-free per the GDD: the UI shows no per-item verdict;
   this class still grades per rep so the debrief can reveal everything at
   the end. Checkpoint rows are buffered locally and only enter the ledger
   when the full set completes — an abandoned checkpoint must never leave a
   partial set that would corrupt masteredAtOf's chunking. Misses still queue
   into spaced repetition (GDD §3 failure UX), keyed to the missed ITEM's own
   drill.

   This module owns state + persistence; rendering lives in Drill.svelte.
   ========================================================================== */

import type { Answer, Confidence, DecisionRecord, QueueItem, Verdict } from '../types';
import { Prng, siblingSeed } from '../gen/prng';
import { LOOP_DRILLS, buildLoopItem } from './items';
import type { LoopDrillId, LoopItem } from './items';
import { gradeLoopItem } from './grade';
import { summarize } from './summary';
import type { BlockSummary, RepRecord } from './summary';
import { flush, record, store } from '../schedule/persist';
import {
  ITEM_RD,
  advance,
  checkpointComposition,
  checkpointRetryAt,
  dueItems,
  enqueueMiss,
  isLeech,
  movesRating,
  passesCheckpoint,
  selectKnobs,
  woodpeckerSeed,
} from '../schedule/queue';
import { masteredAtOf } from '../schedule/tree';
import type { DifficultyKnobs } from '../gen/scripts';
import { ratingWeight, updateRating } from '../schedule/glicko';
import { player } from '../stores/player.svelte';

export const BLOCK_SIZE = 10;

export type BlockPhase = 'loading' | 'armed' | 'verdict' | 'summary';

export interface BlockOptions {
  drillId: LoopDrillId;
  /** Block master seed (decimal string). Omit for a fresh random block. */
  seed?: string;
  /** Skip queue consumption (deterministic dev/e2e blocks). */
  noQueue?: boolean;
  /** Difficulty knobs; default = rating-matched adaptive selection (~85%). */
  knobs?: DifficultyKnobs;
  /** Run as the delayed mastery checkpoint (GDD §3 mode 6). */
  checkpoint?: boolean;
}

/** Debrief result of a checkpoint block (GDD §3 mastery / failure UX). */
export interface CheckpointResult {
  nCorrect: number;
  cardinals: number;
  passed: boolean;
  /** Earliest retry (local midnight + 2 days); null when passed. */
  retryAt: number | null;
}

function freshSeed(): string {
  // 53 random bits + time — decimal-string seed for a non-deterministic block
  return String(
    (BigInt(Date.now()) << 20n) ^ BigInt(Math.floor(Math.random() * 2 ** 40)),
  );
}

export class DrillBlock {
  readonly drillId: LoopDrillId;
  readonly blockSeed: string;
  readonly size = BLOCK_SIZE;

  phase = $state<BlockPhase>('loading');
  rep = $state(1);
  current = $state<LoopItem | null>(null);
  verdict = $state<Verdict | null>(null);
  lastAnswer = $state<Answer | null>(null);
  confidence = $state<Confidence>('lean');
  reps = $state<RepRecord[]>([]);
  /** Display-rating delta of the last rated rep (null before the first). */
  ratingDelta = $state<number | null>(null);
  summary = $state<BlockSummary | null>(null);
  /** Set at block end in checkpoint mode (null otherwise / until then). */
  checkpointResult = $state<CheckpointResult | null>(null);

  /** Wall-clock when the current item was armed (latency measurement). */
  armedAt = 0;

  /** True when this block is the delayed mastery checkpoint (GDD §3). */
  readonly isCheckpoint: boolean;

  private next: LoopItem | null = null;
  private readonly knobs: DifficultyKnobs;
  private readonly noQueue: boolean;
  /** Due Woodpecker entries consumed by this block, by rep index (1-based). */
  private queuePlan = new Map<number, QueueItem>();
  /** Checkpoint interleave: sibling drill served at a rep (else the target). */
  private itemPlan = new Map<number, LoopDrillId>();
  /** Checkpoint rows held back until the full set completes (see header). */
  private checkpointRows: DecisionRecord[] = [];
  private queueUpdates: QueueItem[] = [];
  private queueRemovals: string[] = [];
  private newMisses: QueueItem[] = [];

  constructor(opts: BlockOptions) {
    this.drillId = opts.drillId;
    this.blockSeed = opts.seed ?? freshSeed();
    this.isCheckpoint = opts.checkpoint ?? false;
    this.noQueue = (opts.noQueue ?? false) || this.isCheckpoint;
    this.knobs = opts.knobs ?? selectKnobs(player.rating(opts.drillId));
  }

  /** Item seed for a rep: due-queue sibling first, fresh rated seed after. */
  private seedFor(rep: number): string {
    const q = this.queuePlan.get(rep);
    if (q) return woodpeckerSeed(q);
    return siblingSeed(this.blockSeed, 100 + rep);
  }

  /** Mode of the current rep. */
  get mode(): DecisionRecord['mode'] {
    if (this.isCheckpoint) return 'checkpoint';
    return this.queuePlan.has(this.rep) ? 'woodpecker' : 'rated';
  }

  /** The drill served at a rep (checkpoint interleaves mastered siblings). */
  private drillFor(rep: number): LoopDrillId {
    return this.itemPlan.get(rep) ?? this.drillId;
  }

  private build(rep: number): LoopItem {
    const drill = this.drillFor(rep);
    // Sibling items are served at THEIR gate difficulty (rating-matched to
    // the player's rating on that node) — same selection path as their loop.
    const knobs = drill === this.drillId ? this.knobs : selectKnobs(player.rating(drill));
    return buildLoopItem(drill, this.seedFor(rep), knobs);
  }

  /**
   * Checkpoint interleave plan (GDD §3): 40% target / 60% confusable items
   * from MASTERED playable siblings. Which reps carry siblings — and which
   * sibling each carries — is drawn from the seeded PRNG's 'checkpoint-mix'
   * substream, so the same block seed always serves the identical exam.
   * With no mastered siblings yet the whole set is the target skill.
   */
  private planCheckpoint(masteredSiblings: LoopDrillId[]): void {
    if (masteredSiblings.length === 0) return;
    const { siblingItems } = checkpointComposition(this.size);
    const rng = new Prng(this.blockSeed).stream('checkpoint-mix');
    // choose siblingItems distinct rep slots out of 1..size (partial shuffle)
    const slots = Array.from({ length: this.size }, (_, i) => i + 1);
    for (let i = 0; i < siblingItems; i++) {
      const j = rng.nextInt(i, slots.length - 1);
      [slots[i], slots[j]] = [slots[j], slots[i]];
      this.itemPlan.set(slots[i], masteredSiblings[rng.nextInt(0, masteredSiblings.length - 1)]);
    }
  }

  /** Load queue/interleave plan + first item, then arm rep 1. */
  async init(): Promise<void> {
    if (this.isCheckpoint) {
      try {
        // Mastered playable siblings, measured from the ledger (GDD §3).
        const decisions = await store().allDecisions();
        const byNode = new Map<string, DecisionRecord[]>();
        for (const d of decisions) {
          const arr = byNode.get(d.nodeId);
          if (arr) arr.push(d);
          else byNode.set(d.nodeId, [d]);
        }
        const mastered = LOOP_DRILLS.filter(
          (d) => d !== this.drillId && masteredAtOf(d, byNode.get(d) ?? []) !== null,
        );
        this.planCheckpoint(mastered);
      } catch {
        /* store unavailable — serve an all-target checkpoint */
      }
    } else if (!this.noQueue) {
      try {
        const due = dueItems(await store().allQueueItems(), Date.now()).filter(
          (q) => q.drillId === this.drillId && !isLeech(q),
        );
        // reps 2..4 re-serve due misses (rep 1 stays fresh so the block
        // always opens on-rating; sibling seeds kill answer memorization)
        due.slice(0, 3).forEach((q, i) => this.queuePlan.set(i + 2, q));
      } catch {
        /* store unavailable (private mode) — serve a fresh block */
      }
    }
    this.current = this.build(1);
    this.arm();
  }

  private arm(): void {
    this.confidence = 'lean';
    this.verdict = null;
    this.lastAnswer = null;
    this.phase = 'armed';
    this.armedAt = performance.now();
  }

  /**
   * Commit an answer (call on pointerdown). Grading + state flip are fully
   * synchronous — the verdict is renderable in the same frame. Returns the
   * verdict. latencyOverride is dev-only (?state=verdict screenshots).
   */
  answer(choice: string | number, latencyOverride?: number): Verdict | null {
    if (this.phase !== 'armed' || !this.current) return null;
    const li = this.current;
    const isChipDrill = li.item.choices.length > 0;
    const ans: Answer = {
      itemId: li.item.id,
      choice,
      confidence: isChipDrill ? this.confidence : null,
      latencyMs: Math.max(0, Math.round(latencyOverride ?? performance.now() - this.armedAt)),
    };
    const verdict = gradeLoopItem(li, ans);
    this.lastAnswer = ans;
    this.verdict = verdict;
    this.phase = 'verdict';
    this.reps = [...this.reps, { li, answer: ans, verdict }];
    this.applyScoring(li, ans, verdict);
    // pre-generate the next item during the feedback window (never in the
    // verdict frame — the swap at NEXT is then instant)
    if (this.rep < this.size && this.next === null) {
      setTimeout(() => {
        if (this.next === null && this.rep < this.size) this.next = this.build(this.rep + 1);
      }, 30);
    }
    return verdict;
  }

  /** Rating movement + ledger row + queue bookkeeping (no awaited IO). */
  private applyScoring(li: LoopItem, ans: Answer, verdict: Verdict): void {
    const mode = this.mode;
    const now = Date.now();
    const row: DecisionRecord = {
      decisionId: `${this.drillId}:${li.item.seed}:${now}:${this.rep}`,
      // checkpoint rows attribute to the TARGET node (masteredAtOf chunks by
      // nodeId) while drillId keeps the served item's true drill (siblings)
      nodeId: this.drillId,
      drillId: li.item.drillId,
      seed: li.item.seed,
      paramsVersion: li.item.paramsVersion,
      answer: ans,
      verdict,
      at: now,
      mode,
      // exact Glicko opponent for the rating replay (schedule/tree): the
      // knobs are frozen at block start, so the replay must see the same
      // item rating the live update used — never re-derive it per rep
      itemRating: li.item.itemRating,
    };
    // checkpoint sets persist all-or-nothing (see class header); everything
    // else streams into the write-behind buffer immediately
    if (this.isCheckpoint) this.checkpointRows.push(row);
    else record(row);

    if (movesRating(mode)) {
      const before = player.rating(this.drillId);
      const after = updateRating(
        before,
        li.item.itemRating,
        ITEM_RD,
        verdict.score / 100,
        ratingWeight(verdict.cardinal),
      );
      player.setRating(after);
      this.ratingDelta = Math.round(after.rating) - Math.round(before.rating);
    }

    const consumed = this.queuePlan.get(this.rep);
    if (consumed) {
      const advanced = advance(consumed, verdict.correct, now);
      if (advanced === null) this.queueRemovals.push(consumed.id);
      else this.queueUpdates.push(advanced);
    } else if (!verdict.correct) {
      // fresh miss → spaced repetition (1d), sibling re-served by Woodpecker;
      // keyed to the served ITEM's own drill (a checkpoint miss on a sibling
      // item reviews on that sibling's node — GDD §3 failure UX)
      this.newMisses.push(
        enqueueMiss(
          `${li.item.drillId}:${li.item.seed}:${now}`,
          li.item.drillId,
          li.item.drillId,
          li.item.seed,
          li.item.paramsVersion,
          now,
        ),
      );
    }
  }

  /** Advance to the next rep, or close the block into the summary. */
  advanceRep(): void {
    if (this.phase !== 'verdict') return;
    if (this.rep >= this.size) {
      this.summary = summarize(this.drillId, this.reps);
      if (this.isCheckpoint) {
        const nCorrect = this.reps.filter((r) => r.verdict.correct).length;
        const cardinals = this.reps.filter((r) => r.verdict.cardinal).length;
        const passed = passesCheckpoint(this.drillId, nCorrect, this.size, cardinals);
        this.checkpointResult = {
          nCorrect,
          cardinals,
          passed,
          retryAt: passed ? null : checkpointRetryAt(Date.now()),
        };
      }
      this.phase = 'summary';
      void this.persistBlock();
      return;
    }
    this.rep += 1;
    this.current = this.next ?? this.build(this.rep);
    this.next = null;
    this.arm();
  }

  /** Flush the ledger + rating + queue changes (idle path, GDD §9). */
  private async persistBlock(): Promise<void> {
    try {
      // the complete checkpoint set enters the ledger only now (all-or-nothing)
      for (const row of this.checkpointRows) record(row);
      this.checkpointRows = [];
      await flush();
      const s = store();
      // plain snapshot: $state proxies fail IndexedDB's structured clone
      await s.putRating({ ...player.rating(this.drillId) });
      if (this.newMisses.length > 0 || this.queueUpdates.length > 0) {
        await s.putQueueItems([...this.newMisses, ...this.queueUpdates]);
      }
      for (const id of this.queueRemovals) await s.removeQueueItem(id);
    } catch {
      /* persistence is best-effort in v1 (no backend, private-mode safe) */
    }
  }
}
