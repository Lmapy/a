import { beforeEach, describe, expect, it } from 'vitest';
import type { Confidence, DecisionRecord } from '../types';
import {
  AuctionDB,
  BLOCK_SIZE,
  BRIER_WINDOW,
  CONFIDENCE_PROB,
  DB_VERSION,
  MemoryStore,
  blockBotPoints,
  botPoints,
  brier,
  brierFromConfidence,
  brierLedger,
  calibrationBuckets,
  calibrationEntryOf,
  clampProb,
  confidenceProb,
  flush,
  getPref,
  pendingCount,
  record,
  rollingBrier,
  setPref,
  store,
  useMemoryStore,
} from './persist';

function decision(
  id: string,
  overrides: Partial<DecisionRecord> = {},
  confidence: Confidence | null = 'lean',
  brierVal: number | null = 0.5625,
): DecisionRecord {
  return {
    decisionId: id,
    nodeId: 'shape-alphabet',
    drillId: 'shape-alphabet',
    seed: '123',
    paramsVersion: 'v1.0.0',
    answer: { itemId: 'i1', choice: 'P', confidence, latencyMs: 1400 },
    verdict: {
      correct: false,
      score: 0,
      explanation: 'That is a b, not a P — the bulge sits low with a thin upper tail.',
      explanationTemplateId: 'shape-alphabet:miss:p-b-confusion',
      refs: ['guide#3.1'],
      cardinal: false,
      brier: brierVal,
    },
    at: Date.now(),
    mode: 'rated',
    ...overrides,
  };
}

describe('write-behind buffer (verdict path never awaits IO)', () => {
  it('record() is synchronous and buffers', () => {
    const before = pendingCount();
    record(decision(`d${Math.random()}`));
    expect(pendingCount()).toBe(before + 1);
  });

  it('flush() drains the buffer into the active store', async () => {
    const mem = useMemoryStore();
    await flush(); // clear anything buffered by earlier tests
    record(decision('flush-1'));
    record(decision('flush-2', { nodeId: 'regime-gate', drillId: 'regime-gate' }));
    expect(pendingCount()).toBe(2);
    await flush();
    expect(pendingCount()).toBe(0);
    expect((await mem.allDecisions()).map((d) => d.decisionId).sort()).toEqual(
      expect.arrayContaining(['flush-1', 'flush-2']),
    );
    expect((await mem.decisionsForNode('regime-gate')).map((d) => d.decisionId)).toEqual(['flush-2']);
  });
});

describe('store resolution', () => {
  it('falls back to MemoryStore in Node (no IndexedDB)', () => {
    useMemoryStore();
    expect(store()).toBeInstanceOf(MemoryStore);
  });
});

describe('MemoryStore CRUD (in-memory fallback used by tests)', () => {
  let mem: MemoryStore;
  beforeEach(() => {
    mem = useMemoryStore();
  });

  it('round-trips decisions, ratings, and queue items', async () => {
    await mem.putDecisions([decision('d1'), decision('d2', { nodeId: 'poc-va-snap' })]);
    expect(await mem.allDecisions()).toHaveLength(2);
    expect((await mem.decisionsForNode('poc-va-snap')).map((d) => d.decisionId)).toEqual(['d2']);

    const rating = { nodeId: 'poc-va-snap', rating: 1550, rd: 200, volatility: 0.06, nAnswers: 5 };
    await mem.putRating(rating);
    expect(await mem.getRating('poc-va-snap')).toEqual(rating);
    expect(await mem.getRating('missing')).toBeUndefined();
    expect(await mem.allRatings()).toHaveLength(1);

    const q = {
      id: 'q1',
      nodeId: 'n',
      drillId: 'shape-alphabet' as const,
      seed: '1',
      paramsVersion: 'v1',
      dueAt: 1,
      intervalIndex: 0,
      lapses: 1,
    };
    await mem.putQueueItems([q]);
    expect(await mem.allQueueItems()).toEqual([q]);
    await mem.removeQueueItem('q1');
    expect(await mem.allQueueItems()).toEqual([]);

    await mem.clear();
    expect(await mem.allDecisions()).toEqual([]);
    expect(await mem.allRatings()).toEqual([]);
  });

  it('putDecisions upserts by decisionId (idempotent flush)', async () => {
    await mem.putDecisions([decision('same')]);
    await mem.putDecisions([decision('same')]);
    expect(await mem.allDecisions()).toHaveLength(1);
  });
});

describe('AuctionDB schema', () => {
  it('declares the three tables at the current version without opening IDB', () => {
    // Constructing a Dexie subclass does not hit IndexedDB until open();
    // in Node there is no IDB, so we only assert schema declaration.
    const db = new AuctionDB('schema-smoke');
    expect(db.tables.map((t) => t.name).sort()).toEqual(['decisions', 'queue', 'ratings']);
    expect(DB_VERSION).toBeGreaterThanOrEqual(1);
  });
});

describe('confidence-tap mapping (GDD §4-B/§5: sure/lean/guess → 90/75/55%)', () => {
  it('pins the exact implicit probabilities', () => {
    expect(CONFIDENCE_PROB).toEqual({ sure: 0.9, lean: 0.75, guess: 0.55 });
    expect(confidenceProb('sure')).toBe(0.9);
    expect(confidenceProb('lean')).toBe(0.75);
    expect(confidenceProb('guess')).toBe(0.55);
  });

  it('confidence stays under the 95% cap and above chance', () => {
    for (const c of ['sure', 'lean', 'guess'] as const) {
      expect(CONFIDENCE_PROB[c]).toBeLessThanOrEqual(0.95);
      expect(CONFIDENCE_PROB[c]).toBeGreaterThan(0.5);
    }
  });
});

describe('Brier math', () => {
  it('brier = (p − o)²', () => {
    expect(brier(0.9, true)).toBeCloseTo(0.01, 12);
    expect(brier(0.9, false)).toBeCloseTo(0.81, 12);
    expect(brier(0.55, true)).toBeCloseTo(0.2025, 12);
  });

  it('brierFromConfidence scores the tap against correctness', () => {
    expect(brierFromConfidence('sure', true)).toBeCloseTo(0.01, 12);
    expect(brierFromConfidence('sure', false)).toBeCloseTo(0.81, 12);
    expect(brierFromConfidence('lean', false)).toBeCloseTo(0.5625, 12);
    expect(brierFromConfidence('guess', true)).toBeCloseTo(0.2025, 12);
  });

  it('clampProb enforces the taught 5–95% band', () => {
    expect(clampProb(0.99)).toBe(0.95);
    expect(clampProb(0.01)).toBe(0.05);
    expect(clampProb(0.6)).toBe(0.6);
  });
});

describe('calibrationEntryOf', () => {
  it('maps a confidence tap to its implicit p and recovers the outcome from Brier', () => {
    const missAtLean = decision('m', {}, 'lean', 0.5625); // (0.75 − 0)²
    expect(calibrationEntryOf(missAtLean)).toMatchObject({ p: 0.75, hit: false });
    const hitAtSure = decision('h', {}, 'sure', 0.01); // (0.9 − 1)²
    expect(calibrationEntryOf(hitAtSure)).toMatchObject({ p: 0.9, hit: true });
  });

  it('uses the slider value for calibration-range items', () => {
    const d = decision('s', {
      drillId: 'calibration-range',
      answer: { itemId: 'i', choice: 0.8, confidence: null, latencyMs: 900 },
    });
    d.verdict.brier = (0.8 - 1) ** 2; // hit
    expect(calibrationEntryOf(d)).toMatchObject({ p: 0.8, hit: true });
  });

  it('returns null when no probability was logged', () => {
    expect(calibrationEntryOf(decision('n', {}, null, null))).toBeNull();
    expect(calibrationEntryOf(decision('n2', {}, null, 0.25))).toBeNull(); // brier but no p source
  });

  it('brierLedger extracts entries oldest-first and skips non-probability reps', () => {
    const a = decision('a', { at: 200 }, 'sure', 0.01);
    const b = decision('b', { at: 100 }, 'guess', 0.3025);
    const c = decision('c', { at: 150 }, null, null);
    const ledger = brierLedger([a, b, c]);
    expect(ledger.map((e) => e.at)).toEqual([100, 200]);
    expect(ledger[0]).toMatchObject({ p: 0.55, hit: false });
  });
});

describe('rollingBrier (Drill E mastery input)', () => {
  it('averages the last `window` entries and is null when empty', () => {
    expect(rollingBrier([])).toBeNull();
    const entries = [
      { p: 0.9, hit: true, at: 1 }, // 0.01
      { p: 0.9, hit: false, at: 2 }, // 0.81
    ];
    expect(rollingBrier(entries)).toBeCloseTo(0.41, 12);
    // Window: only the newest entry counts with window = 1.
    expect(rollingBrier(entries, 1)).toBeCloseTo(0.81, 12);
    expect(BRIER_WINDOW).toBe(50);
  });
});

describe('calibrationBuckets (reliability curve)', () => {
  it('bins by stated probability with per-bucket meanP and hitRate', () => {
    const entries = [
      { p: 0.75, hit: true, at: 1 },
      { p: 0.75, hit: false, at: 2 },
      { p: 0.78, hit: true, at: 3 },
      { p: 0.9, hit: true, at: 4 },
    ];
    const buckets = calibrationBuckets(entries, 10);
    expect(buckets).toHaveLength(10);
    const b7 = buckets[7]; // [0.7, 0.8)
    expect(b7.n).toBe(3);
    expect(b7.meanP).toBeCloseTo((0.75 + 0.75 + 0.78) / 3, 12);
    expect(b7.hitRate).toBeCloseTo(2 / 3, 12);
    const b9 = buckets[9]; // [0.9, 1.0]
    expect(b9.n).toBe(1);
    expect(b9.hitRate).toBe(1);
    expect(buckets[0].n).toBe(0);
    expect(buckets[0].meanP).toBeNull();
    // Total preserved.
    expect(buckets.reduce((s, b) => s + b.n, 0)).toBe(entries.length);
  });
});

describe('Bot Points (GDD §5: AP = Σ (Brier_bot − Brier_you) × 100 per block of 25)', () => {
  it('positive when beating the base-rate bot, negative when losing to it', () => {
    // Event occurs; base rate 0.63. Saying 0.9 beats the bot; saying 0.2 loses.
    const win = botPoints([{ p: 0.9, hit: true, at: 1, baseRate: 0.63 }]);
    expect(win).toBeCloseTo(((0.63 - 1) ** 2 - (0.9 - 1) ** 2) * 100, 10);
    expect(win).toBeGreaterThan(0);
    expect(botPoints([{ p: 0.2, hit: true, at: 1, baseRate: 0.63 }])).toBeLessThan(0);
    // Answering exactly the base rate ties the bot.
    expect(botPoints([{ p: 0.63, hit: false, at: 1, baseRate: 0.63 }])).toBeCloseTo(0, 12);
  });

  it('blockBotPoints scores only completed blocks of 25', () => {
    expect(BLOCK_SIZE).toBe(25);
    const entry = { p: 0.9, hit: true, at: 1, baseRate: 0.63 };
    expect(blockBotPoints(Array.from({ length: 24 }, () => entry))).toEqual([]);
    const two = blockBotPoints(Array.from({ length: 60 }, () => entry));
    expect(two).toHaveLength(2);
    expect(two[0]).toBeCloseTo(botPoints(Array.from({ length: 25 }, () => entry)), 10);
  });
});

describe('prefs helpers degrade gracefully without localStorage', () => {
  it('getPref falls back and setPref is a no-op in Node', () => {
    expect(() => setPref('mute', '1')).not.toThrow();
    const v = getPref('definitely-unset-key', 'fallback');
    expect(typeof v).toBe('string');
  });
});
