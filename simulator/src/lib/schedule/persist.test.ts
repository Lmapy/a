import { beforeEach, describe, expect, it } from 'vitest';
import type { DecisionRecord } from '../types';
import { AuctionDB, DB_VERSION, getPref, pendingCount, record, setPref } from './persist';

function decision(id: string): DecisionRecord {
  return {
    decisionId: id,
    nodeId: 'shape-alphabet',
    drillId: 'shape-alphabet',
    seed: '123',
    paramsVersion: 'v1.0.0',
    answer: { itemId: 'i1', choice: 'P', confidence: 'lean', latencyMs: 1400 },
    verdict: {
      correct: false,
      score: 0,
      explanation: 'That is a b, not a P — the bulge sits low with a thin upper tail.',
      explanationTemplateId: 'shape-alphabet:miss:p-b-confusion',
      refs: ['guide#3.1'],
      cardinal: false,
      brier: 0.5625,
    },
    at: Date.now(),
    mode: 'rated',
  };
}

describe('write-behind buffer (verdict path never awaits IO)', () => {
  it('record() is synchronous and buffers', () => {
    const before = pendingCount();
    record(decision(`d${Math.random()}`));
    expect(pendingCount()).toBe(before + 1);
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

describe('prefs helpers degrade gracefully without localStorage', () => {
  beforeEach(() => {
    // Node has no localStorage unless something polyfilled it
  });
  it('getPref falls back and setPref is a no-op in Node', () => {
    expect(() => setPref('mute', '1')).not.toThrow();
    const v = getPref('definitely-unset-key', 'fallback');
    expect(typeof v).toBe('string');
  });
});
