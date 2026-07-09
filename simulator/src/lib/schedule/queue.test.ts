import { describe, expect, it } from 'vitest';
import {
  INTERVALS_MS,
  LEECH_THRESHOLD,
  advance,
  assembleWarmUp,
  dueItems,
  enqueueMiss,
  isLeech,
} from './queue';

const NOW = 1_800_000_000_000;

describe('enqueueMiss', () => {
  it('creates an interval-0 entry due in 1 day carrying (seed, paramsVersion)', () => {
    const q = enqueueMiss('q1', 'shape-alphabet', 'shape-alphabet', '999', 'v1.0.0', NOW);
    expect(q.dueAt).toBe(NOW + INTERVALS_MS[0]);
    expect(q.intervalIndex).toBe(0);
    expect(q.lapses).toBe(1);
    expect(q.seed).toBe('999');
    expect(q.paramsVersion).toBe('v1.0.0');
  });
});

describe('advance', () => {
  const q = enqueueMiss('q1', 'n', 'shape-alphabet', '1', 'v1', NOW);

  it('correct review climbs the 1d/3d/7d ladder then graduates (null)', () => {
    const a = advance(q, true, NOW)!;
    expect(a.intervalIndex).toBe(1);
    expect(a.dueAt).toBe(NOW + INTERVALS_MS[1]);
    const b = advance(a, true, NOW)!;
    expect(b.intervalIndex).toBe(2);
    expect(advance(b, true, NOW)).toBeNull();
  });

  it('a miss resets to interval 0 and counts a lapse', () => {
    const climbed = advance(q, true, NOW)!;
    const missed = advance(climbed, false, NOW)!;
    expect(missed.intervalIndex).toBe(0);
    expect(missed.lapses).toBe(q.lapses + 1);
    expect(missed.dueAt).toBe(NOW + INTERVALS_MS[0]);
  });
});

describe('dueItems / isLeech', () => {
  it('returns only due entries, soonest first', () => {
    const early = { ...enqueueMiss('a', 'n', 'regime-gate', '1', 'v1', NOW), dueAt: NOW - 500 };
    const later = { ...enqueueMiss('b', 'n', 'regime-gate', '2', 'v1', NOW), dueAt: NOW - 100 };
    const future = enqueueMiss('c', 'n', 'regime-gate', '3', 'v1', NOW);
    expect(dueItems([future, later, early], NOW).map((q) => q.id)).toEqual(['a', 'b']);
  });

  it('leech rule fires at 8 lapses (GDD §3)', () => {
    const q = enqueueMiss('a', 'n', 'excess-or-poor', '1', 'v1', NOW);
    expect(isLeech(q)).toBe(false);
    expect(isLeech({ ...q, lapses: LEECH_THRESHOLD })).toBe(true);
  });
});

describe('assembleWarmUp (stub contract)', () => {
  it('takes at most 2 due misses + a learning node + a mixed block', () => {
    const queue = ['a', 'b', 'c'].map((id, i) => ({
      ...enqueueMiss(id, 'n', 'poc-va-snap' as const, String(i), 'v1', NOW),
      dueAt: NOW - 1,
    }));
    const plan = assembleWarmUp(queue, ['regime-gate'], ['poc-va-snap', 'shape-alphabet'], NOW);
    expect(plan.woodpecker).toHaveLength(2);
    expect(plan.newSkillNodeId).toBe('regime-gate');
    expect(plan.mixedNodeIds).toContain('shape-alphabet');
  });
});
