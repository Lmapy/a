/* @boss/store tests — ledger row construction + best-attempt ordering. */

import { describe, expect, it } from 'vitest';
import { useMemoryStore } from '../schedule/persist';
import {
  SETUP_OF_ACTION,
  decisionRecordOf,
  isBetterAttempt,
  persistRun,
  queueOneThingToDrill,
  reflexRecordOf,
} from './store';
import type { BossRunSummary } from './store';
import { gradeDecision, gradeReflex } from './engine';
import { buildBossSession } from './engine';
import { DEMO_SEED } from './engine';

const session = buildBossSession(DEMO_SEED);

function summary(over: Partial<BossRunSummary>): BossRunSummary {
  return {
    at: 1,
    seed: '1',
    paramsVersion: 'v1.1.0',
    readScore: 50,
    pnlR: -1,
    passed: false,
    breach: null,
    actions: ['fade', 'skip', 'skip'],
    ...over,
  };
}

describe('decisionRecordOf', () => {
  it('attributes ticket decisions to regime-gate with the executed TradeFill', () => {
    const g = gradeDecision('fade', 'balance', session.gen.labels, session.decisionBars[0], null, 1);
    const rec = decisionRecordOf({
      seed: session.seed,
      paramsVersion: session.gen.script.paramsVersion,
      decisionIndex: 0,
      bar: session.decisionBars[0],
      action: 'fade',
      read: 'balance',
      latencyMs: 2100,
      verdict: g.verdict,
      rPnl: -1,
      at: 1234,
    });
    expect(rec.mode).toBe('boss');
    expect(rec.nodeId).toBe('regime-gate');
    expect(rec.seed).toBe(session.seed);
    expect(rec.trade).toEqual({ setup: SETUP_OF_ACTION.fade, regime: 'balance', rMultiple: -1 });
    expect(rec.verdict.cardinal).toBe(true);
  });

  it('stand-aside decisions carry no trade (nothing for the R-histogram)', () => {
    const g = gradeDecision('stand-aside', 'imbalance', session.gen.labels, session.decisionBars[0], null, 1);
    const rec = decisionRecordOf({
      seed: session.seed,
      paramsVersion: 'v1.1.0',
      decisionIndex: 1,
      bar: session.decisionBars[0],
      action: 'stand-aside',
      read: 'imbalance',
      latencyMs: 900,
      verdict: g.verdict,
      rPnl: null,
      at: 5,
    });
    expect(rec.trade).toBeUndefined();
  });
});

describe('reflexRecordOf', () => {
  it('attributes reflexes to one-timeframing-buzzer; slept answers are marked', () => {
    const g = gradeReflex('KILL', null, 5000);
    const rec = reflexRecordOf({
      seed: session.seed,
      paramsVersion: 'v1.1.0',
      bar: 135,
      answer: null,
      latencyMs: 5000,
      verdict: g.verdict,
      at: 7,
    });
    expect(rec.nodeId).toBe('one-timeframing-buzzer');
    expect(rec.answer.choice).toBe('SLEPT');
    expect(rec.mode).toBe('boss');
  });
});

describe('persistRun + queueOneThingToDrill', () => {
  it('writes ledger rows and a due Woodpecker sibling into the store', async () => {
    const mem = useMemoryStore();
    const g = gradeDecision('go-with', 'imbalance', session.gen.labels, session.decisionBars[1], null, 1);
    await persistRun([
      decisionRecordOf({
        seed: session.seed,
        paramsVersion: 'v1.1.0',
        decisionIndex: 0,
        bar: session.decisionBars[1],
        action: 'go-with',
        read: 'imbalance',
        latencyMs: 1500,
        verdict: g.verdict,
        rPnl: 2,
        at: 99,
      }),
    ]);
    const rows = await mem.allDecisions();
    expect(rows).toHaveLength(1);
    expect(rows[0].mode).toBe('boss');
    expect(rows[0].trade?.rMultiple).toBe(2);

    await queueOneThingToDrill(session.seed, 'v1.1.0', 1000);
    const queue = await mem.allQueueItems();
    expect(queue).toHaveLength(1);
    expect(queue[0].drillId).toBe('regime-gate');
    expect(queue[0].seed).toBe(session.seed);
  });
});

describe('isBetterAttempt', () => {
  it('passed beats not-passed, then Read Score, then P&L; null is always beaten', () => {
    expect(isBetterAttempt(summary({}), null)).toBe(true);
    expect(isBetterAttempt(summary({ passed: true, readScore: 10 }), summary({ readScore: 90 }))).toBe(true);
    expect(isBetterAttempt(summary({ readScore: 60 }), summary({ readScore: 61 }))).toBe(false);
    expect(isBetterAttempt(summary({ pnlR: -0.5 }), summary({ pnlR: -1 }))).toBe(true);
  });
});
