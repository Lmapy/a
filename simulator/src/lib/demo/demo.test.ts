/* Tests for the ?demo=1 dataset: determinism, the node states the tree
   renders, warm-up feed, expectancy coverage, and calibration story — all
   verified through the same engine derivations the screens use. */

import { describe, expect, it } from 'vitest';
import { buildDemoDataset } from './demo';
import { deriveTree, replayRatings } from '../schedule/tree';
import { dueItems } from '../schedule/queue';
import { brierLedger, useDefaultStore, useMemoryStore, store } from '../schedule/persist';
import {
  bestMeasuredEdge,
  botBlockSeries,
  calibrationView,
  expectancyTable,
  rHistogram,
  tradesOf,
} from '../stats/model';

const NOW = new Date(2026, 6, 8, 15, 0, 0).getTime(); // local mid-day

describe('buildDemoDataset', () => {
  const ds = buildDemoDataset(NOW);

  it('is deterministic for a fixed now', () => {
    expect(JSON.stringify(buildDemoDataset(NOW))).toBe(JSON.stringify(ds));
  });

  it('derives every tree state the Home screen must render', () => {
    const t = deriveTree(ds.decisions, NOW);
    expect(t.nodes['poc-va-snap'].state).toBe('mastered');
    expect(t.nodes['hvn-lvn-marker'].state).toBe('mastered');
    expect(t.nodes['excess-or-poor'].state).toBe('rusty');
    expect(t.nodes['shape-alphabet'].state).toBe('checkpoint-armed');
    expect(t.nodes['open-type-ladder'].state).toBe('learning');
    expect(t.activeNodeId).toBe('open-type-ladder');
    for (const id of ['one-timeframing-buzzer', 'acceptance-clock', 'regime-gate', 'calibration-range', 'setup-picker'] as const) {
      expect(t.nodes[id].state).toBe('locked');
    }
    // the armed checkpoint waits for the next calendar day
    expect(t.nodes['shape-alphabet'].checkpointAvailableAt).toBeGreaterThan(NOW);
  });

  it('ratings are exactly the Glicko replay of the decision rows', () => {
    const replay = replayRatings(ds.decisions).states;
    for (const r of ds.ratings) {
      expect(r.rating).toBeCloseTo(replay[r.nodeId].rating, 10);
      expect(r.nAnswers).toBe(replay[r.nodeId].nAnswers);
      expect(r.rating).toBeGreaterThan(800);
      expect(r.rating).toBeLessThan(2400);
    }
  });

  it('feeds the Warm-Up: two due Woodpecker entries on the rusty node', () => {
    const due = dueItems(ds.queue, NOW);
    expect(due).toHaveLength(2);
    expect(due.every((q) => q.nodeId === 'excess-or-poor')).toBe(true);
    expect(ds.queue).toHaveLength(3); // one entry not yet due
    expect(ds.streak).toEqual({ days: 12, freezes: 2 });
  });

  it('covers the expectancy ledger: measured AND unmeasured cells + histogram', () => {
    const trades = tradesOf(ds.decisions);
    expect(trades.length).toBeGreaterThan(200);
    const rows = expectancyTable(trades);
    const cells = rows.flatMap((r) => [r.balance, r.imbalance]).filter((c) => c !== null);
    expect(cells.some((c) => c.measured)).toBe(true);
    expect(cells.some((c) => !c.measured)).toBe(true);
    expect(bestMeasuredEdge(rows)).not.toBeNull();
    const h = rHistogram(trades);
    expect(h.n).toBe(trades.length);
    expect(h.counts.reduce((a, b) => a + b, 0)).toBe(trades.length);
  });

  it('tells a calibration story: headline present, bot blocks flowing', () => {
    const entries = brierLedger(ds.decisions);
    expect(entries.length).toBeGreaterThan(100);
    const v = calibrationView(entries);
    expect(v.headline).not.toBeNull();
    expect(v.headline?.clause).toContain('overconfident');
    expect(botBlockSeries(ds.decisions).length).toBeGreaterThanOrEqual(5);
  });
});

describe('persist.useDefaultStore', () => {
  it('drops a forced memory store so store() re-resolves fresh', async () => {
    const m = useMemoryStore();
    await m.putDecisions(buildDemoDataset(NOW).decisions.slice(0, 1));
    expect((await store().allDecisions())).toHaveLength(1);
    useDefaultStore();
    // In Node the default resolves to a NEW empty memory store.
    expect((await store().allDecisions())).toHaveLength(0);
  });
});
