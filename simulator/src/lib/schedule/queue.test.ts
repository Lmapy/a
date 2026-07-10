import { describe, expect, it } from 'vitest';
import type { QueueItem } from '../types';
import { DEFAULT_KNOBS } from '../gen/scripts';
import { initialRating, targetItemRating } from './glicko';
import {
  CHECKPOINT_PASS_MIN,
  CHECKPOINT_SIZE,
  INTERVALS_MS,
  LEECH_THRESHOLD,
  MASTERY_CRITERIA,
  RUSTY_THRESHOLD,
  advance,
  armCheckpoint,
  assembleWarmUp,
  checkpointComposition,
  checkpointRetryAt,
  dueItems,
  enqueueMiss,
  expectedSuccessAgainst,
  isGateArmed,
  isLeech,
  isRusty,
  itemRatingFromKnobs,
  knobsForRating,
  movesRating,
  nextCalendarDayMs,
  passesCheckpoint,
  selectKnobs,
  startOfDay,
  woodpeckerSeed,
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

describe('dueItems / isLeech / isRusty', () => {
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

  it('mastered nodes demote to rusty below 70% spaced-review accuracy', () => {
    expect(RUSTY_THRESHOLD).toBe(0.7);
    expect(isRusty(0.69)).toBe(true);
    expect(isRusty(0.7)).toBe(false);
  });
});

describe('Woodpecker sibling re-serve', () => {
  const q = enqueueMiss('q1', 'shape-alphabet', 'shape-alphabet', '123456789', 'v1', NOW);

  it('never serves the identical seed and is deterministic', () => {
    expect(woodpeckerSeed(q)).not.toBe(q.seed);
    expect(woodpeckerSeed(q)).toBe(woodpeckerSeed({ ...q }));
  });

  it('every serve along any review path presents a distinct sibling', () => {
    const seeds = new Set<string>([q.seed]);
    // Path: miss, correct, miss, correct, correct — 5 serves.
    let item: QueueItem | null = q;
    const outcomes = [false, true, false, true, true];
    for (const ok of outcomes) {
      expect(item).not.toBeNull();
      const s = woodpeckerSeed(item!);
      expect(seeds.has(s)).toBe(false);
      seeds.add(s);
      item = advance(item!, ok, NOW);
    }
  });
});

describe('rating-matched difficulty knobs (GDD §5)', () => {
  it('item rating is monotone in every knob hardness and spans 800–2400', () => {
    expect(itemRatingFromKnobs(knobsForRating(0))).toBe(800);
    expect(itemRatingFromKnobs(knobsForRating(9999))).toBe(2400);
    const base = itemRatingFromKnobs(DEFAULT_KNOBS);
    expect(itemRatingFromKnobs({ ...DEFAULT_KNOBS, noiseGain: 0.9 })).toBeGreaterThan(base);
    expect(itemRatingFromKnobs({ ...DEFAULT_KNOBS, nu: 4 })).toBeGreaterThan(base);
    expect(itemRatingFromKnobs({ ...DEFAULT_KNOBS, ambiguity: 0.4 })).toBeGreaterThan(base);
    expect(itemRatingFromKnobs({ ...DEFAULT_KNOBS, decoys: 2 })).toBeGreaterThan(base);
    expect(itemRatingFromKnobs({ ...DEFAULT_KNOBS, lvnDepth: 0.2 })).toBeGreaterThan(base);
    expect(itemRatingFromKnobs({ ...DEFAULT_KNOBS, jumpIntensity: 1 })).toBeGreaterThan(base);
  });

  it('knobsForRating round-trips through itemRatingFromKnobs within decoy rounding (±60)', () => {
    for (const r of [900, 1100, 1350, 1600, 1900, 2200]) {
      const back = itemRatingFromKnobs(knobsForRating(r));
      expect(Math.abs(back - r)).toBeLessThanOrEqual(60);
    }
  });

  it('knob values stay inside their documented ranges', () => {
    for (const r of [0, 800, 1500, 2400, 5000]) {
      const k = knobsForRating(r);
      expect(k.noiseGain).toBeGreaterThanOrEqual(0);
      expect(k.noiseGain).toBeLessThanOrEqual(1);
      expect(k.nu).toBeGreaterThanOrEqual(4);
      expect(k.nu).toBeLessThanOrEqual(8);
      expect(k.ambiguity).toBeGreaterThanOrEqual(0);
      expect(k.ambiguity).toBeLessThanOrEqual(0.5);
      expect([0, 1, 2]).toContain(k.decoys);
      expect(k.lvnDepth).toBeGreaterThanOrEqual(0);
      expect(k.lvnDepth).toBeLessThanOrEqual(1);
    }
  });

  it('selectKnobs targets ~85% expected success for a fresh player', () => {
    const player = initialRating('shape-alphabet');
    const knobs = selectKnobs(player);
    expect(itemRatingFromKnobs(knobs)).toBeLessThan(player.rating); // easier than the player
    expect(Math.abs(expectedSuccessAgainst(player, knobs) - 0.85)).toBeLessThan(0.05);
  });

  it('a stronger player gets harder knobs', () => {
    const fresh = initialRating('shape-alphabet');
    const strong = { ...fresh, rating: 2100 };
    expect(selectKnobs(strong).noiseGain).toBeGreaterThan(selectKnobs(fresh).noiseGain);
    expect(targetItemRating(strong)).toBeGreaterThan(targetItemRating(fresh));
  });
});

describe('rating-moving modes (GDD §5)', () => {
  it('only rated and checkpoint move rating', () => {
    expect(movesRating('rated')).toBe(true);
    expect(movesRating('checkpoint')).toBe(true);
    for (const m of ['rush', 'streak', 'woodpecker', 'warmup', 'calibration', 'boss'] as const) {
      expect(movesRating(m)).toBe(false);
    }
  });
});

describe('mastery gates (accuracy AND latency, GDD §3–4)', () => {
  it('pins the GDD per-drill criteria', () => {
    expect(MASTERY_CRITERIA['poc-va-snap']).toMatchObject({ minAccuracy: 0.92, maxMedianLatencyMs: 2500 });
    expect(MASTERY_CRITERIA['excess-or-poor'].maxMedianLatencyMs).toBe(2000);
    expect(MASTERY_CRITERIA['hvn-lvn-marker'].minAccuracy).toBe(0.9);
    expect(MASTERY_CRITERIA['shape-alphabet'].maxMedianLatencyMs).toBe(3000);
    expect(MASTERY_CRITERIA['calibration-range'].maxBrier).toBe(0.18);
    expect(MASTERY_CRITERIA['regime-gate']).toMatchObject({ maxMedianLatencyMs: 5000, requireZeroCardinal: true });
    expect(MASTERY_CRITERIA['setup-picker'].minNoTradeCompliance).toBe(0.9);
  });

  it('requires BOTH accuracy and latency — either alone fails', () => {
    expect(isGateArmed('poc-va-snap', { accuracy: 0.93, medianLatencyMs: 2400 })).toBe(true);
    expect(isGateArmed('poc-va-snap', { accuracy: 0.93, medianLatencyMs: 2600 })).toBe(false);
    expect(isGateArmed('poc-va-snap', { accuracy: 0.9, medianLatencyMs: 2400 })).toBe(false);
  });

  it('calibration-range gates on rolling Brier, not raw accuracy', () => {
    expect(isGateArmed('calibration-range', { accuracy: 0, medianLatencyMs: 60_000, rollingBrier: 0.17 })).toBe(true);
    expect(isGateArmed('calibration-range', { accuracy: 1, medianLatencyMs: 100, rollingBrier: 0.19 })).toBe(false);
    expect(isGateArmed('calibration-range', { accuracy: 1, medianLatencyMs: 100 })).toBe(false); // unmeasured
  });

  it('regime-gate requires zero cardinal errors', () => {
    const good = { accuracy: 0.9, medianLatencyMs: 4000, cardinalErrors: 0 };
    expect(isGateArmed('regime-gate', good)).toBe(true);
    expect(isGateArmed('regime-gate', { ...good, cardinalErrors: 1 })).toBe(false);
  });

  it('setup-picker requires ≥90% NO-TRADE compliance', () => {
    const good = { accuracy: 0.9, medianLatencyMs: 8000, noTradeCompliance: 0.95 };
    expect(isGateArmed('setup-picker', good)).toBe(true);
    expect(isGateArmed('setup-picker', { ...good, noTradeCompliance: 0.85 })).toBe(false);
  });
});

describe('delayed interleaved checkpoint (GDD §3)', () => {
  it('arms no sooner than the next calendar day', () => {
    const armedAt = new Date(2026, 6, 9, 15, 30, 0).getTime(); // 3:30pm local
    const availableAt = armCheckpoint(armedAt);
    expect(availableAt).toBe(new Date(2026, 6, 10, 0, 0, 0).getTime());
    expect(availableAt).toBeGreaterThan(armedAt);
    // Arming a minute before midnight still pushes to the NEXT day.
    const late = new Date(2026, 6, 9, 23, 59, 0).getTime();
    expect(armCheckpoint(late)).toBe(new Date(2026, 6, 10, 0, 0, 0).getTime());
  });

  it('startOfDay/nextCalendarDayMs bracket a timestamp', () => {
    const t = new Date(2026, 0, 15, 12, 0, 0).getTime();
    expect(startOfDay(t)).toBe(new Date(2026, 0, 15, 0, 0, 0).getTime());
    expect(nextCalendarDayMs(t)).toBe(new Date(2026, 0, 16, 0, 0, 0).getTime());
  });

  it('failed checkpoint retries 2 calendar days out', () => {
    const failedAt = new Date(2026, 6, 9, 10, 0, 0).getTime();
    expect(checkpointRetryAt(failedAt)).toBe(new Date(2026, 6, 11, 0, 0, 0).getTime());
  });

  it('composition is 10 items at 40/60 target-vs-confusable-sibling mix', () => {
    expect(CHECKPOINT_SIZE).toBe(10);
    expect(checkpointComposition()).toEqual({ targetItems: 4, siblingItems: 6 });
  });

  it('pass requires ≥8/10; regime-gate additionally requires zero cardinal errors', () => {
    expect(CHECKPOINT_PASS_MIN).toBe(8);
    expect(passesCheckpoint('shape-alphabet', 8)).toBe(true);
    expect(passesCheckpoint('shape-alphabet', 7)).toBe(false);
    expect(passesCheckpoint('regime-gate', 9, 10, 0)).toBe(true);
    expect(passesCheckpoint('regime-gate', 9, 10, 1)).toBe(false);
  });
});

describe('assembleWarmUp (GDD §3 mode 8)', () => {
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

  it('excludes leeches from the Woodpecker block and surfaces them as micro-lessons', () => {
    const leech = {
      ...enqueueMiss('l', 'n', 'excess-or-poor' as const, '7', 'v1', NOW),
      dueAt: NOW - 10,
      lapses: LEECH_THRESHOLD,
    };
    const normal = { ...enqueueMiss('m', 'n', 'excess-or-poor' as const, '8', 'v1', NOW), dueAt: NOW - 5 };
    const plan = assembleWarmUp([leech, normal], [], [], NOW);
    expect(plan.woodpecker.map((q) => q.id)).toEqual(['m']);
    expect(plan.leeches.map((q) => q.id)).toEqual(['l']);
  });

  it('mixed block holds ≤3 nodes and rotates deterministically by calendar day', () => {
    const mastered = ['a', 'b', 'c', 'd', 'e'];
    const day1 = assembleWarmUp([], [], mastered, NOW);
    const day2 = assembleWarmUp([], [], mastered, NOW + 24 * 60 * 60 * 1000);
    expect(day1.mixedNodeIds).toHaveLength(3);
    expect(day1.mixedNodeIds).not.toEqual(day2.mixedNodeIds); // interleaving varies
    expect(assembleWarmUp([], [], mastered, NOW).mixedNodeIds).toEqual(day1.mixedNodeIds); // deterministic
  });

  it('handles an empty tree gracefully', () => {
    const plan = assembleWarmUp([], [], [], NOW);
    expect(plan.woodpecker).toEqual([]);
    expect(plan.newSkillNodeId).toBeNull();
    expect(plan.mixedNodeIds).toEqual([]);
    expect(plan.leeches).toEqual([]);
  });
});
