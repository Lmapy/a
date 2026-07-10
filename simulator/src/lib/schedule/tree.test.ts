/* Tests for @schedule/tree — node-state derivation from the ledger, rating
   replay, tier meta, and the prerequisite DAG. */

import { describe, expect, it } from 'vitest';
import type { DecisionRecord, DrillId } from '../types';
import { initialRating, ratingWeight, updateRating } from './glicko';
import { ITEM_RD, armCheckpoint, checkpointRetryAt, dueItems } from './queue';
import {
  GATE_WINDOW,
  PLAYABLE_DRILLS,
  PREREQUISITE,
  TREE_ORDER,
  TREE_TIERS,
  deltaOverLastActiveDay,
  deriveTree,
  effectivePrereq,
  lastFailedCheckpointAt,
  masteredAtOf,
  reconstructedItemRating,
  replayRatings,
  reviewAccuracyOf,
  tierMastery,
} from './tree';
import { LOOP_DRILLS } from '../drills/items';

const DAY = 24 * 60 * 60 * 1000;
const NOW = new Date(2026, 6, 8, 15, 0, 0).getTime(); // local 3pm, mid-day safe

let seq = 0;
function dec(
  nodeId: DrillId,
  opts: Partial<{
    at: number;
    correct: boolean;
    mode: DecisionRecord['mode'];
    latencyMs: number;
    score: number;
    cardinal: boolean;
  }> = {},
): DecisionRecord {
  const correct = opts.correct ?? true;
  return {
    decisionId: `t:${seq++}`,
    nodeId,
    drillId: nodeId,
    seed: '1',
    paramsVersion: 't',
    answer: {
      itemId: 'i',
      choice: 'X',
      confidence: null,
      latencyMs: opts.latencyMs ?? 1500,
    },
    verdict: {
      correct,
      score: opts.score ?? (correct ? 100 : 0),
      explanation: 'x',
      explanationTemplateId: 'x',
      refs: [],
      cardinal: opts.cardinal ?? false,
      brier: null,
    },
    at: opts.at ?? NOW - 1000 * seq,
    mode: opts.mode ?? 'rated',
  };
}

/** n reps at a fixed accuracy pattern, 30s apart ending at `end`. */
function block(
  nodeId: DrillId,
  n: number,
  nWrong: number,
  end: number,
  mode: DecisionRecord['mode'] = 'rated',
  latencyMs = 1500,
): DecisionRecord[] {
  return Array.from({ length: n }, (_, i) =>
    dec(nodeId, {
      at: end - (n - i) * 30_000,
      correct: i >= nWrong,
      mode,
      latencyMs,
    }),
  );
}

/** A passing checkpoint (9/10) ending at `end`. */
function checkpoint(nodeId: DrillId, end: number, nCorrect = 9): DecisionRecord[] {
  return block(nodeId, 10, 10 - nCorrect, end, 'checkpoint');
}

describe('tree layout', () => {
  it('TREE_ORDER is topological w.r.t. the prerequisite DAG', () => {
    for (const [node, prereq] of Object.entries(PREREQUISITE)) {
      expect(TREE_ORDER.indexOf(prereq as DrillId)).toBeLessThan(
        TREE_ORDER.indexOf(node as DrillId),
      );
    }
  });

  it('every tier node is a distinct known drill id', () => {
    const ids = TREE_TIERS.flatMap((t) => t.nodes.map((n) => n.id));
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toEqual(TREE_ORDER);
  });

  it('PLAYABLE_DRILLS mirrors the wired loop roster (drills/items)', () => {
    expect([...PLAYABLE_DRILLS].sort()).toEqual([...LOOP_DRILLS].sort());
  });

  it('effectivePrereq skips unplayable v1.1 nodes', () => {
    // excess-or-poor's declared prereq is hvn-lvn-marker (v1.1) → poc-va-snap
    expect(effectivePrereq('excess-or-poor')).toBe('poc-va-snap');
    // regime-gate's chain walks past acceptance-clock + one-timeframing
    expect(effectivePrereq('regime-gate')).toBe('open-type-ladder');
    expect(effectivePrereq('poc-va-snap')).toBeUndefined();
    // playable prereqs are untouched
    expect(effectivePrereq('shape-alphabet')).toBe('excess-or-poor');
  });
});

describe('masteredAtOf', () => {
  it('a full checkpoint set with ≥8/10 masters at its last decision', () => {
    const cp = checkpoint('shape-alphabet', NOW - DAY, 8);
    expect(masteredAtOf('shape-alphabet', cp)).toBe(cp[cp.length - 1].at);
  });

  it('7/10 does not master; an incomplete second set is ignored', () => {
    const fail = checkpoint('shape-alphabet', NOW - DAY, 7);
    expect(masteredAtOf('shape-alphabet', fail)).toBeNull();
    const pass = checkpoint('shape-alphabet', NOW - DAY, 9);
    const partial = pass.concat(block('shape-alphabet', 4, 0, NOW, 'checkpoint'));
    expect(masteredAtOf('shape-alphabet', partial)).toBe(pass[pass.length - 1].at);
  });

  it('regime-gate: a cardinal error in the set fails the checkpoint', () => {
    const cp = checkpoint('regime-gate', NOW - DAY, 9);
    cp[0] = { ...cp[0], verdict: { ...cp[0].verdict, cardinal: true } };
    expect(masteredAtOf('regime-gate', cp)).toBeNull();
  });
});

describe('lastFailedCheckpointAt', () => {
  it('reports the latest failed set; passing sets do not count', () => {
    const fail = checkpoint('shape-alphabet', NOW - 3 * DAY, 6);
    expect(lastFailedCheckpointAt('shape-alphabet', fail)).toBe(fail[fail.length - 1].at);
    const pass = checkpoint('shape-alphabet', NOW - DAY, 9);
    expect(lastFailedCheckpointAt('shape-alphabet', fail.concat(pass))).toBe(
      fail[fail.length - 1].at,
    );
    expect(lastFailedCheckpointAt('shape-alphabet', pass)).toBeNull();
  });
});

describe('deriveTree', () => {
  it('empty ledger: root learning, everything else locked', () => {
    const t = deriveTree([], NOW);
    expect(t.nodes['poc-va-snap'].state).toBe('learning');
    for (const id of TREE_ORDER.slice(1)) expect(t.nodes[id].state).toBe('locked');
    expect(t.activeNodeId).toBe('poc-va-snap');
  });

  it('gate arms on accuracy AND latency; checkpoint waits for the next calendar day', () => {
    // excess-or-poor gate: ≥85% acc, median < 2000ms; unlock its prereqs first
    const ledger = [
      ...checkpoint('poc-va-snap', NOW - 10 * DAY),
      ...checkpoint('hvn-lvn-marker', NOW - 9 * DAY),
      ...block('excess-or-poor', GATE_WINDOW, 2, NOW - 3600_000, 'rated', 1500), // 90%
    ];
    const t = deriveTree(ledger, NOW);
    expect(t.nodes['excess-or-poor'].state).toBe('checkpoint-armed');
    expect(t.nodes['excess-or-poor'].checkpointAvailableAt).toBe(
      armCheckpoint(NOW - 3600_000 - 30_000),
    );

    // same accuracy, latency over the 2000ms criterion → still learning
    const slow = [
      ...checkpoint('poc-va-snap', NOW - 10 * DAY),
      ...checkpoint('hvn-lvn-marker', NOW - 9 * DAY),
      ...block('excess-or-poor', GATE_WINDOW, 2, NOW - 3600_000, 'rated', 2500),
    ];
    expect(deriveTree(slow, NOW).nodes['excess-or-poor'].state).toBe('learning');
  });

  it('mastered demotes to rusty when spaced-review accuracy decays below 70%', () => {
    const base = [
      ...checkpoint('poc-va-snap', NOW - 10 * DAY),
      ...block('poc-va-snap', 10, 5, NOW - DAY, 'woodpecker'), // 50% review
    ];
    expect(deriveTree(base, NOW).nodes['poc-va-snap'].state).toBe('rusty');
    const healthy = [
      ...checkpoint('poc-va-snap', NOW - 10 * DAY),
      ...block('poc-va-snap', 10, 1, NOW - DAY, 'woodpecker'), // 90% review
    ];
    expect(deriveTree(healthy, NOW).nodes['poc-va-snap'].state).toBe('mastered');
  });

  it('rusty still satisfies the prerequisite chain (unlocks the branch)', () => {
    const ledger = [
      ...checkpoint('poc-va-snap', NOW - 12 * DAY),
      ...checkpoint('hvn-lvn-marker', NOW - 11 * DAY),
      ...checkpoint('excess-or-poor', NOW - 10 * DAY),
      ...block('excess-or-poor', 10, 6, NOW - DAY, 'woodpecker'), // rusty
    ];
    const t = deriveTree(ledger, NOW);
    expect(t.nodes['excess-or-poor'].state).toBe('rusty');
    expect(t.nodes['shape-alphabet'].state).toBe('learning');
    expect(t.nodes['open-type-ladder'].state).toBe('learning'); // T1 branch
    expect(t.nodes['one-timeframing-buzzer'].state).toBe('locked');
  });

  it('boss-mode decisions are not practice evidence (never unlock a node)', () => {
    const boss = [dec('setup-picker', { mode: 'boss', at: NOW - DAY })];
    expect(deriveTree(boss, NOW).nodes['setup-picker'].state).toBe('locked');
  });

  it('progression never dead-ends on unplayable v1.1 nodes (open-type reachable)', () => {
    // Master ONLY the playable chain — never touch hvn-lvn-marker (v1.1).
    const ledger = [...checkpoint('poc-va-snap', NOW - 3 * DAY)];
    const t = deriveTree(ledger, NOW);
    // v1.1 node with no history renders locked but does not gate the line
    expect(t.nodes['hvn-lvn-marker'].state).toBe('locked');
    expect(t.nodes['excess-or-poor'].state).toBe('learning');
    // …and mastering excess-or-poor opens the T1 branch (the GDD DoD path)
    const t2 = deriveTree(
      ledger.concat(checkpoint('excess-or-poor', NOW - 2 * DAY)),
      NOW,
    );
    expect(t2.nodes['open-type-ladder'].state).toBe('learning');
    expect(t2.nodes['shape-alphabet'].state).toBe('learning');
    // the active row is always a playable node
    expect(PLAYABLE_DRILLS).toContain(t2.activeNodeId);
  });

  it('a failed checkpoint pushes availability to the +2-day retry (GDD §3)', () => {
    const failedCp = checkpoint('excess-or-poor', NOW - 26 * 3600_000, 7); // 7/10 — failed, gate still met
    const failedAt = failedCp[failedCp.length - 1].at;
    const ledger = [
      ...checkpoint('poc-va-snap', NOW - 10 * DAY),
      ...block('excess-or-poor', GATE_WINDOW, 2, NOW - 2 * DAY, 'rated', 1500),
      ...failedCp,
    ];
    const t = deriveTree(ledger, NOW);
    const n = t.nodes['excess-or-poor'];
    expect(n.state).toBe('checkpoint-armed');
    expect(n.failedCheckpointAt).toBe(failedAt);
    expect(n.checkpointAvailableAt).toBe(checkpointRetryAt(failedAt));
    expect(n.checkpointAvailableAt! > NOW).toBe(true);
  });

  it('tierMastery counts clean mastery only (rusty excluded)', () => {
    const ledger = [
      ...checkpoint('poc-va-snap', NOW - 12 * DAY),
      ...checkpoint('hvn-lvn-marker', NOW - 11 * DAY),
      ...checkpoint('excess-or-poor', NOW - 10 * DAY),
      ...block('excess-or-poor', 10, 6, NOW - DAY, 'woodpecker'), // rusty
    ];
    const t = deriveTree(ledger, NOW);
    expect(tierMastery(TREE_TIERS[0], t.nodes)).toEqual({ mastered: 2, total: 4 });
  });
});

describe('reviewAccuracyOf', () => {
  it('uses only woodpecker/warmup modes over the last window', () => {
    const nd = [
      ...block('excess-or-poor', 10, 0, NOW - 2 * DAY, 'rated'),
      ...block('excess-or-poor', 4, 2, NOW - DAY, 'woodpecker'),
      ...block('excess-or-poor', 2, 1, NOW - DAY / 2, 'warmup'),
    ];
    expect(reviewAccuracyOf(nd)).toBeCloseTo(3 / 6, 10);
    expect(reviewAccuracyOf(block('excess-or-poor', 5, 0, NOW, 'rated'))).toBeNull();
  });
});

describe('replayRatings', () => {
  it('matches the hand-computed Glicko chain (cardinal ⇒ 3× weight)', () => {
    const d1 = dec('regime-gate', { at: 1000, correct: true, mode: 'rated' });
    const d2 = dec('regime-gate', { at: 2000, correct: false, cardinal: true, mode: 'rated' });
    const d3 = dec('regime-gate', { at: 3000, correct: true, mode: 'woodpecker' }); // ignored
    const replay = replayRatings([d2, d3, d1]); // unsorted on purpose

    let s = initialRating('regime-gate');
    s = updateRating(s, reconstructedItemRating(s), ITEM_RD, 1, ratingWeight(false));
    s = updateRating(s, reconstructedItemRating(s), ITEM_RD, 0, ratingWeight(true));
    expect(replay.states['regime-gate'].rating).toBeCloseTo(s.rating, 10);
    expect(replay.states['regime-gate'].nAnswers).toBe(s.nAnswers);
    expect(replay.history['regime-gate']).toHaveLength(2);
  });

  it('replays the LIVE update path exactly when rows carry the persisted itemRating', () => {
    // The drill loop freezes knobs (⇒ one item rating) at block start; the
    // replay must reproduce its trajectory one-for-one from the persisted
    // opponent — displayed deltas may never contradict the rating shown.
    const frozenItemRating = 1234; // deliberately NOT what per-rep re-derivation would pick
    const pattern = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1];
    const rows = pattern.map((sc, i) => ({
      ...dec('excess-or-poor', { at: 1000 + i, correct: sc === 1, mode: 'rated' as const }),
      itemRating: frozenItemRating,
    }));
    const replay = replayRatings(rows);

    let live = initialRating('excess-or-poor');
    for (const sc of pattern) live = updateRating(live, frozenItemRating, ITEM_RD, sc, 1);
    expect(replay.states['excess-or-poor'].rating).toBeCloseTo(live.rating, 10);
    expect(replay.states['excess-or-poor'].rd).toBeCloseTo(live.rd, 10);
    // and it must NOT equal the legacy per-rep reconstruction (they diverge)
    let legacy = initialRating('excess-or-poor');
    for (const sc of pattern) legacy = updateRating(legacy, reconstructedItemRating(legacy), ITEM_RD, sc, 1);
    expect(replay.states['excess-or-poor'].rating).not.toBeCloseTo(legacy.rating, 1);
  });

  it('deltaOverLastActiveDay isolates the last calendar day of movement', () => {
    const day1 = new Date(2026, 6, 6, 12).getTime();
    const day2 = new Date(2026, 6, 7, 12).getTime();
    const points = [
      { at: day1, rating: 1520 },
      { at: day1 + 60_000, rating: 1540 },
      { at: day2, rating: 1550 },
      { at: day2 + 60_000, rating: 1532 },
    ];
    expect(deltaOverLastActiveDay(points)).toBe(1532 - 1540);
    expect(deltaOverLastActiveDay([])).toBe(0);
    // single-day history: measured from the 1500 baseline
    expect(deltaOverLastActiveDay([{ at: day1, rating: 1512 }])).toBe(12);
  });
});

describe('demo queue sanity (dueItems integration)', () => {
  it('due filtering matches queue.dueItems semantics', () => {
    const q = [
      { id: 'a', nodeId: 'x', drillId: 'excess-or-poor' as DrillId, seed: '1', paramsVersion: 't', dueAt: NOW - 1, intervalIndex: 0, lapses: 1 },
      { id: 'b', nodeId: 'x', drillId: 'excess-or-poor' as DrillId, seed: '1', paramsVersion: 't', dueAt: NOW + 1, intervalIndex: 0, lapses: 1 },
    ];
    expect(dueItems(q, NOW).map((i) => i.id)).toEqual(['a']);
  });
});
