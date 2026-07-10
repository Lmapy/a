/* ============================================================================
   @demo — deterministic review dataset behind the ?demo=1 flag.

   The demo seeds a plausible 5-week practice career as RAW LEDGER ROWS
   (DecisionRecords, queue entries, boss trade fills) and nothing else: every
   number the screens then show — ratings, deltas, node states, calibration
   buckets, Bot Points, Brier trend, expectancy cells — is computed from these
   rows by the same engine code (glicko / queue / tree / stats model) that
   processes real play. Only the simulated PLAYER BEHAVIOR (which rep was
   right, which confidence was tapped, the trade R draws) is scripted here;
   no displayed quantity is hand-placed.

   Determinism: all draws flow from Prng(DEMO_SEED) named substreams —
   buildDemoDataset(now) with equal `now` is byte-identical.

   Career arc (exercises every node state the tree renders):
   · poc-va-snap / hvn-lvn-marker — mastered (passed checkpoints)
   · excess-or-poor — mastered, then decayed spaced reviews → RUSTY, with two
     due Woodpecker queue entries (feeds the Warm-Up card)
   · shape-alphabet — gate accuracy+latency cleared today → CHECKPOINT-ARMED
   · open-type-ladder — in progress below the gate → LEARNING (active row)
   · everything after — LOCKED
   · ~240 boss trade fills across the playbook → expectancy ledger with both
     measured (n ≥ 30) and unmeasured cells + R histogram.

   Pure TS — zero DOM/Svelte imports.
   ========================================================================== */

import type {
  Confidence,
  DecisionRecord,
  DrillId,
  QueueItem,
  RatingState,
  Regime,
} from '../types';
import { Prng, siblingSeed } from '../gen/prng';
import type { RandomStream } from '../gen/prng';
import { brierFromConfidence } from '../schedule/persist';
import { INTERVALS_MS } from '../schedule/queue';
import { replayRatings } from '../schedule/tree';

export const DEMO_SEED = '426942694269';
export const DEMO_PARAMS_VERSION = 'demo-1';

const DAY = 24 * 60 * 60 * 1000;
const MIN = 60 * 1000;

/** The assembled demo dataset (persist via the LedgerStore, then hydrate). */
export interface DemoDataset {
  decisions: DecisionRecord[];
  queue: QueueItem[];
  /** Final Glicko states — the replay of `decisions`, exactly as live play persists them. */
  ratings: RatingState[];
  streak: { days: number; freezes: number };
}

/* ----------------------------------------------------------------------------
   Simulated player behavior
   -------------------------------------------------------------------------- */

/** Confidence-tap shares of the simulated player. */
const CONF_SHARE: [Confidence, number][] = [
  ['sure', 0.45],
  ['lean', 0.35],
  ['guess', 0.2],
];

/**
 * Hit rate per tapped confidence at the player's baseline (~0.72 pooled):
 * overconfident when saying "sure" (90% → ~80%), near-honest below — the
 * calibration story the dashboard measures and reports.
 */
const BUCKET_HIT: Record<Confidence, number> = { sure: 0.78, lean: 0.72, guess: 0.56 };
const BASELINE_ACC = 0.715; // Σ share·hit

function drawConfidence(s: RandomStream): Confidence {
  let u = s.nextFloat();
  for (const [c, w] of CONF_SHARE) {
    if (u < w) return c;
    u -= w;
  }
  return 'guess';
}

/** Par times per node (drill catalog §4) for latency draws. */
const PAR: Partial<Record<DrillId, number>> = {
  'poc-va-snap': 2500,
  'excess-or-poor': 2000,
  'hvn-lvn-marker': 2500,
  'shape-alphabet': 3000,
  'open-type-ladder': 6000,
};

/** Chip-drill flag: only chip drills carry confidence taps (loop contract). */
const CHIP: Partial<Record<DrillId, boolean>> = {
  'excess-or-poor': true,
  'shape-alphabet': true,
  'open-type-ladder': true,
  'regime-gate': true,
  'poc-va-snap': false,
  'hvn-lvn-marker': false,
};

/* ----------------------------------------------------------------------------
   Ledger-row factory
   -------------------------------------------------------------------------- */

interface RepSpec {
  nodeId: DrillId;
  mode: DecisionRecord['mode'];
  at: number;
  /** Block accuracy target: hit prob scales as BUCKET_HIT · acc/BASELINE_ACC. */
  acc: number;
  index: number;
}

function makeRep(spec: RepSpec, s: RandomStream): DecisionRecord {
  const chip = CHIP[spec.nodeId] ?? true;
  const conf = chip ? drawConfidence(s) : null;
  const pHit = Math.min(
    0.98,
    Math.max(0.05, (conf ? BUCKET_HIT[conf] : BASELINE_ACC) * (spec.acc / BASELINE_ACC)),
  );
  const correct = s.nextFloat() < pHit;
  // Snap drills score 60 on ±1-row near-misses about a third of the time.
  const score = correct ? 100 : !chip && s.nextFloat() < 0.33 ? 60 : 0;
  const par = PAR[spec.nodeId] ?? 4000;
  const latencyMs = Math.max(
    Math.round(par * 0.3),
    Math.round(par * (0.75 + 0.15 * s.nextGaussian())),
  );
  return {
    decisionId: `demo:${spec.nodeId}:${spec.index}`,
    nodeId: spec.nodeId,
    drillId: spec.nodeId,
    seed: siblingSeed(DEMO_SEED, spec.index),
    paramsVersion: DEMO_PARAMS_VERSION,
    answer: {
      itemId: `${spec.nodeId}:${siblingSeed(DEMO_SEED, spec.index)}:0`,
      choice: correct ? 'TRUTH' : 'MISS',
      confidence: conf,
      latencyMs,
    },
    verdict: {
      correct,
      score,
      explanation: correct ? '✓ demo ledger row (seeded)' : '✗ demo ledger row (seeded)',
      explanationTemplateId: 'demo',
      refs: [],
      cardinal: false,
      brier: conf ? brierFromConfidence(conf, correct) : null,
    },
    at: spec.at,
    mode: spec.mode,
  };
}

/* ----------------------------------------------------------------------------
   Career script
   -------------------------------------------------------------------------- */

/** One practice day of one node: `blocks` blocks of 10 at accuracy `acc`. */
interface DayScript {
  daysAgo: number;
  blocks: number;
  acc: number;
  mode?: DecisionRecord['mode'];
}

const CAREER: { nodeId: DrillId; days: DayScript[]; checkpointDaysAgo: number | null }[] = [
  {
    nodeId: 'poc-va-snap',
    days: [
      { daysAgo: 34, blocks: 2, acc: 0.74 },
      { daysAgo: 33, blocks: 2, acc: 0.85 },
      { daysAgo: 31, blocks: 2, acc: 0.92 },
      { daysAgo: 29, blocks: 2, acc: 0.96 },
      { daysAgo: 27, blocks: 2, acc: 0.97 },
    ],
    checkpointDaysAgo: 26,
  },
  {
    nodeId: 'hvn-lvn-marker',
    days: [
      { daysAgo: 25, blocks: 2, acc: 0.72 },
      { daysAgo: 23, blocks: 2, acc: 0.86 },
      { daysAgo: 21, blocks: 2, acc: 0.93 },
      { daysAgo: 19, blocks: 2, acc: 0.96 },
    ],
    checkpointDaysAgo: 18,
  },
  {
    nodeId: 'excess-or-poor',
    days: [
      { daysAgo: 17, blocks: 2, acc: 0.72 },
      { daysAgo: 15, blocks: 2, acc: 0.85 },
      { daysAgo: 13, blocks: 2, acc: 0.93 },
      { daysAgo: 12, blocks: 2, acc: 0.95 },
      // decayed spaced reviews → rusty + due Woodpecker entries
      { daysAgo: 2, blocks: 1, acc: 0.3, mode: 'woodpecker' },
      { daysAgo: 1, blocks: 1, acc: 0.35, mode: 'woodpecker' },
    ],
    checkpointDaysAgo: 11,
  },
  {
    nodeId: 'shape-alphabet',
    days: [
      { daysAgo: 8, blocks: 3, acc: 0.62 },
      { daysAgo: 6, blocks: 3, acc: 0.72 },
      { daysAgo: 4, blocks: 2, acc: 0.8 },
      { daysAgo: 2, blocks: 2, acc: 0.93 },
      { daysAgo: 0, blocks: 2, acc: 0.97 },
    ],
    checkpointDaysAgo: null, // armed, not yet taken
  },
  {
    nodeId: 'open-type-ladder',
    days: [
      { daysAgo: 4, blocks: 2, acc: 0.6 },
      { daysAgo: 2, blocks: 2, acc: 0.72 },
      { daysAgo: 0, blocks: 2, acc: 0.78 },
    ],
    checkpointDaysAgo: null, // learning — the active row
  },
];

/** Boss trade script: (setup, regime, n, mean R, sd R). */
const TRADES: [string, Regime, number, number, number][] = [
  ['VA-Edge Fade', 'balance', 61, 0.38, 1.0],
  ['80% Rule', 'balance', 47, 0.31, 0.9],
  ['Open-Drive Go', 'imbalance', 44, 0.71, 1.1],
  ['LVN Break', 'imbalance', 38, 0.52, 1.0],
  ['Look-Above & Fail', 'balance', 33, -0.08, 0.9],
  ['nPOC Magnet', 'imbalance', 12, 0.2, 0.8],
  ['POC Reversion', 'balance', 9, 0.05, 0.6],
];

/* ----------------------------------------------------------------------------
   Assembly
   -------------------------------------------------------------------------- */

/**
 * Build the deterministic demo dataset relative to `now`. Same (seed, now) ⇒
 * identical dataset; the ratings returned are the Glicko replay of the
 * decision rows (exactly what live play would have persisted).
 */
export function buildDemoDataset(now: number): DemoDataset {
  const prng = new Prng(DEMO_SEED);
  const reps = prng.stream('demo-reps');
  const trades = prng.stream('demo-trades');

  const decisions: DecisionRecord[] = [];
  let index = 0;

  // Anchor practice at `now − 90min` walking backwards per day so "today"
  // rows are today and every row is in the past.
  const anchor = (daysAgo: number, offsetMin: number): number =>
    now - 90 * MIN - daysAgo * DAY + offsetMin * MIN;

  for (const arc of CAREER) {
    for (const day of arc.days) {
      for (let b = 0; b < day.blocks; b++) {
        for (let r = 0; r < 10; r++) {
          decisions.push(
            makeRep(
              {
                nodeId: arc.nodeId,
                mode: day.mode ?? 'rated',
                at: anchor(day.daysAgo, b * 12 + r),
                acc: day.acc,
                index: index++,
              },
              reps,
            ),
          );
        }
      }
    }
    if (arc.checkpointDaysAgo !== null) {
      // The delayed checkpoint: 10 items, 9 correct (≥8/10 passes, GDD §3).
      for (let r = 0; r < 10; r++) {
        const d = makeRep(
          {
            nodeId: arc.nodeId,
            mode: 'checkpoint',
            at: anchor(arc.checkpointDaysAgo, 60 + r),
            acc: 0.9,
            index: index++,
          },
          reps,
        );
        d.verdict = { ...d.verdict, correct: r !== 3, score: r !== 3 ? 100 : 0 };
        decisions.push(d);
      }
    }
  }

  // Boss trade fills — the expectancy ledger + R histogram source (GDD §5).
  let t = 0;
  for (const [setup, regime, n, mean, sd] of TRADES) {
    for (let i = 0; i < n; i++) {
      const r = Math.round(Math.min(3.4, Math.max(-2.4, mean + sd * trades.nextGaussian())) * 100) / 100;
      const at = anchor(1 + (t % 20), 300 + t);
      decisions.push({
        decisionId: `demo:boss:${t}`,
        nodeId: 'setup-picker',
        drillId: 'setup-picker',
        seed: siblingSeed(DEMO_SEED, 100000 + t),
        paramsVersion: DEMO_PARAMS_VERSION,
        answer: { itemId: `boss:${t}`, choice: setup, confidence: null, latencyMs: 4000 },
        verdict: {
          correct: trades.nextFloat() < 0.72,
          score: 0,
          explanation: 'Boss read — demo ledger row (P&L shown, never scored)',
          explanationTemplateId: 'demo-boss',
          refs: [],
          cardinal: false,
          brier: null,
        },
        at,
        mode: 'boss',
        trade: { setup, regime, rMultiple: r },
      });
      t++;
    }
  }

  decisions.sort((a, b) => a.at - b.at);

  // Two due Woodpecker entries on the rusty node → the Warm-Up card's
  // "2 misses due · Excess vs Poor review".
  const queue: QueueItem[] = [
    {
      id: 'demo:q:0',
      nodeId: 'excess-or-poor',
      drillId: 'excess-or-poor',
      seed: siblingSeed(DEMO_SEED, 900001),
      paramsVersion: DEMO_PARAMS_VERSION,
      dueAt: now - 3 * 60 * MIN,
      intervalIndex: 0,
      lapses: 1,
    },
    {
      id: 'demo:q:1',
      nodeId: 'excess-or-poor',
      drillId: 'excess-or-poor',
      seed: siblingSeed(DEMO_SEED, 900002),
      paramsVersion: DEMO_PARAMS_VERSION,
      dueAt: now - 60 * MIN,
      intervalIndex: 1,
      lapses: 2,
    },
    // A not-yet-due entry proves dueItems() filtering (tomorrow's review).
    {
      id: 'demo:q:2',
      nodeId: 'shape-alphabet',
      drillId: 'shape-alphabet',
      seed: siblingSeed(DEMO_SEED, 900003),
      paramsVersion: DEMO_PARAMS_VERSION,
      dueAt: now + INTERVALS_MS[0],
      intervalIndex: 0,
      lapses: 1,
    },
  ];

  return {
    decisions,
    queue,
    ratings: Object.values(replayRatings(decisions).states),
    streak: { days: 12, freezes: 2 },
  };
}
