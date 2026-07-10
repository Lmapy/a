/* Tests for @stats/model — calibration view + headline, bot points blocks,
   Brier trend, expectancy ledger gating (n ≥ 30), R histogram, trend bars. */

import { describe, expect, it } from 'vitest';
import type { Confidence, DecisionRecord, DrillId, Regime } from '../types';
import { brier, brierFromConfidence } from '../schedule/persist';
import {
  DRILL_CHOICES,
  MEASURED_MIN_N,
  MIN_BUCKET_N,
  PLAYBOOK_SETUPS,
  accuracyBars,
  beatingTheBot,
  bestMeasuredEdge,
  botBlockSeries,
  botEntries,
  brierBlockSeries,
  calibrationView,
  expectancyTable,
  rHistogram,
  scopeDecisions,
  tradesOf,
} from './model';
import type { SimTrade } from './model';

const NOW = new Date(2026, 6, 8, 15).getTime();
const DAY = 24 * 60 * 60 * 1000;

let seq = 0;
function dec(
  drillId: DrillId,
  opts: Partial<{
    at: number;
    correct: boolean;
    confidence: Confidence | null;
    mode: DecisionRecord['mode'];
    trade: { setup: string; regime: Regime; rMultiple: number };
  }> = {},
): DecisionRecord {
  const correct = opts.correct ?? true;
  const confidence = opts.confidence ?? null;
  return {
    decisionId: `m:${seq++}`,
    nodeId: drillId,
    drillId,
    seed: '1',
    paramsVersion: 't',
    answer: { itemId: 'i', choice: 'X', confidence, latencyMs: 1500 },
    verdict: {
      correct,
      score: correct ? 100 : 0,
      explanation: 'x',
      explanationTemplateId: 'x',
      refs: [],
      cardinal: false,
      brier: confidence ? brierFromConfidence(confidence, correct) : null,
    },
    at: opts.at ?? NOW - seq * 1000,
    mode: opts.mode ?? 'rated',
    trade: opts.trade,
  };
}

function trade(setup: string, regime: Regime, r: number, at = NOW - seq * 1000): SimTrade {
  return { setup, regime, rMultiple: r, at };
}

describe('scopeDecisions', () => {
  it('keeps the last 90 days, oldest first', () => {
    const inside = dec('shape-alphabet', { at: NOW - 89 * DAY });
    const outside = dec('shape-alphabet', { at: NOW - 91 * DAY });
    const recent = dec('shape-alphabet', { at: NOW - DAY });
    expect(scopeDecisions([recent, outside, inside], NOW)).toEqual([inside, recent]);
  });
});

describe('botEntries / botBlockSeries', () => {
  it('builds bot-comparable entries only for confidence-tapped chip drills', () => {
    const rows = [
      dec('shape-alphabet', { confidence: 'sure', correct: true, at: 1 }),
      dec('poc-va-snap', { at: 2 }), // no confidence → excluded
      dec('excess-or-poor', { confidence: 'guess', correct: false, at: 3 }),
    ];
    const es = botEntries(rows);
    expect(es).toHaveLength(2);
    expect(es[0].baseRate).toBeCloseTo(1 / DRILL_CHOICES['shape-alphabet'], 10);
    expect(es[1].baseRate).toBeCloseTo(1 / DRILL_CHOICES['excess-or-poor'], 10);
    expect(es[0].hit).toBe(true);
    expect(es[1].hit).toBe(false);
  });

  it('bot points per block of 25 match the hand formula', () => {
    // 25 sure-correct shape answers: you (0.9−1)² each, bot (0.2−1)² each
    const rows = Array.from({ length: 25 }, (_, i) =>
      dec('shape-alphabet', { confidence: 'sure', correct: true, at: i + 1 }),
    );
    const expected = Math.round((brier(0.2, true) - brier(0.9, true)) * 100 * 25);
    expect(botBlockSeries(rows)).toEqual([expected]);
    // a trailing partial block is not scored
    expect(botBlockSeries(rows.slice(0, 24))).toEqual([]);
  });

  it('beatingTheBot counts positive blocks over the tail', () => {
    expect(beatingTheBot([5, -3, 8], 12)).toEqual({ won: 2, of: 3 });
    expect(beatingTheBot([], 12)).toEqual({ won: 0, of: 0 });
  });
});

describe('brierBlockSeries', () => {
  it('samples the rolling Brier at each completed block of 25', () => {
    const rows = Array.from({ length: 60 }, (_, i) =>
      dec('excess-or-poor', { confidence: 'lean', correct: i % 2 === 0, at: i + 1 }),
    );
    const entries = rows
      .map((d) => ({ p: 0.75, hit: d.verdict.correct, at: d.at }))
      .sort((a, b) => a.at - b.at);
    const series = brierBlockSeries(entries);
    expect(series).toHaveLength(2); // at 25 and 50
    // window 50 over alternating hits: mean of (0.75−1)² and 0.75²
    const expected = (brier(0.75, true) + brier(0.75, false)) / 2;
    expect(series[1]).toBeCloseTo(expected, 3);
  });
});

describe('calibrationView', () => {
  it('finds the worst qualified bucket and writes the headline sentence', () => {
    const entries = [
      // 20 judgments at 90%, 14 hits → said 90, happened 70
      ...Array.from({ length: 20 }, (_, i) => ({ p: 0.9, hit: i < 14, at: i })),
      // 15 at 55%, 8 hits → said 55, happened 53 (honest guesses)
      ...Array.from({ length: 15 }, (_, i) => ({ p: 0.55, hit: i < 8, at: 100 + i })),
    ];
    const v = calibrationView(entries);
    expect(v.n).toBe(35);
    expect(v.worst).toEqual({ saidPct: 90, hapPct: 70, n: 20 });
    expect(v.headline?.said).toBe(90);
    expect(v.headline?.happened).toBe(70);
    expect(v.headline?.clause).toContain('overconfident');
    expect(v.headline?.clause).toContain('Guesses are honest');
  });

  it('produces no headline until a bucket reaches MIN_BUCKET_N', () => {
    const entries = Array.from({ length: MIN_BUCKET_N - 1 }, (_, i) => ({
      p: 0.9,
      hit: true,
      at: i,
    }));
    const v = calibrationView(entries);
    expect(v.headline).toBeNull();
    expect(v.dots).toHaveLength(1); // the dot still plots
  });
});

describe('expectancy ledger', () => {
  it('cells stay unmeasured until n ≥ 30 (unmeasured = folklore)', () => {
    const t29 = Array.from({ length: MEASURED_MIN_N - 1 }, () =>
      trade('VA-Edge Fade', 'balance', 1),
    );
    const rows29 = expectancyTable(t29);
    expect(rows29[0].setup).toBe('VA-Edge Fade');
    expect(rows29[0].balance?.measured).toBe(false);

    const t30 = t29.concat(trade('VA-Edge Fade', 'balance', -0.5));
    const rows30 = expectancyTable(t30);
    expect(rows30[0].balance?.measured).toBe(true);
    expect(rows30[0].balance?.n).toBe(30);
    expect(rows30[0].balance?.expectancy).toBeCloseTo((29 - 0.5) / 30, 10);
    expect(rows30[0].balance?.winRate).toBeCloseTo(29 / 30, 10);
    expect(rows30[0].imbalance).toBeNull(); // no trades on the other regime
  });

  it('folklore column carries only claims the guide records', () => {
    const byName = Object.fromEntries(PLAYBOOK_SETUPS.map((s) => [s.name, s.folklore]));
    expect(byName['80% Rule']).toContain('80%');
    expect(byName['nPOC Magnet']).toContain('80%');
    expect(byName['Look-Above & Fail']).toContain('70–75%');
    expect(byName['VA-Edge Fade']).toBeNull(); // no folklore number exists
    expect(byName['Open-Drive Go']).toBeNull();
    expect(PLAYBOOK_SETUPS).toHaveLength(9);
  });

  it('bestMeasuredEdge picks the highest measured expectancy', () => {
    const trades = [
      ...Array.from({ length: 30 }, () => trade('VA-Edge Fade', 'balance', 0.3)),
      ...Array.from({ length: 30 }, () => trade('Open-Drive Go', 'imbalance', 0.7)),
      ...Array.from({ length: 5 }, () => trade('LVN Break', 'imbalance', 5)), // unmeasured
    ];
    const best = bestMeasuredEdge(expectancyTable(trades));
    expect(best?.setup).toBe('Open-Drive Go');
    expect(best?.regime).toBe('imbalance');
    expect(bestMeasuredEdge(expectancyTable([]))).toBeNull();
  });
});

describe('tradesOf', () => {
  it('extracts only boss-mode decisions that carry a trade fill', () => {
    const rows = [
      dec('setup-picker', { mode: 'boss', at: 2, trade: { setup: 'LVN Break', regime: 'imbalance', rMultiple: 0.4 } }),
      dec('setup-picker', { mode: 'boss', at: 1, trade: { setup: '80% Rule', regime: 'balance', rMultiple: -1 } }),
      dec('setup-picker', { mode: 'boss', at: 3 }), // read without a fill
      dec('shape-alphabet', { mode: 'rated', at: 4 }),
    ];
    const ts = tradesOf(rows);
    expect(ts.map((t) => t.setup)).toEqual(['80% Rule', 'LVN Break']); // oldest first
    expect(ts[0].rMultiple).toBe(-1);
  });
});

describe('rHistogram', () => {
  it('bins on 0.5R from −2R, clamping outliers into the end bins', () => {
    const h = rHistogram([
      trade('x', 'balance', -3), // clamps into bin 0
      trade('x', 'balance', -2), // bin 0
      trade('x', 'balance', 0.6), // bin floor((0.6+2)/0.5)=5
      trade('x', 'balance', 9), // clamps into the last bin
    ]);
    expect(h.counts[0]).toBe(2);
    expect(h.counts[5]).toBe(1);
    expect(h.counts[h.counts.length - 1]).toBe(1);
    expect(h.n).toBe(4);
    expect(h.avg).toBeCloseTo((-3 - 2 + 0.6 + 9) / 4, 10);
  });

  it('is empty-safe', () => {
    const h = rHistogram([]);
    expect(h.n).toBe(0);
    expect(h.counts.every((c) => c === 0)).toBe(true);
    expect(h.avg).toBe(0);
  });
});

describe('accuracyBars', () => {
  it('splits rated history into k measured accuracy slices', () => {
    const rows = Array.from({ length: 8 }, (_, i) =>
      dec('shape-alphabet', { correct: i % 2 === 0, at: i + 1 }),
    );
    expect(accuracyBars(rows, 4)).toEqual([0.5, 0.5, 0.5, 0.5]);
    // improving run: last quarter perfect
    const run = Array.from({ length: 8 }, (_, i) =>
      dec('shape-alphabet', { correct: i >= 4, at: 100 + i }),
    );
    expect(accuracyBars(run, 4)).toEqual([0, 0, 1, 1]);
  });

  it('needs at least k rated reps; review modes are excluded', () => {
    expect(accuracyBars([dec('shape-alphabet', {})], 4)).toBeNull();
    const reviews = Array.from({ length: 8 }, (_, i) =>
      dec('shape-alphabet', { mode: 'woodpecker', at: i }),
    );
    expect(accuracyBars(reviews, 4)).toBeNull();
  });
});
