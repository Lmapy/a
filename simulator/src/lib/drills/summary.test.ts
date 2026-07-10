/* Block summary math: measured base-rate lines + Bot Points (GDD §2/§5). */
import { describe, expect, it } from 'vitest';
import { EASY_KNOBS } from '../gen/scripts';
import { siblingSeed } from '../gen/prng';
import { buildLoopItem } from './items';
import { gradeLoopItem } from './grade';
import { baseRateLine, blockBotPoints, calibrationLine, median, summarize } from './summary';
import type { RepRecord } from './summary';
import type { Answer } from '../types';

function playBlock(drill: 'excess-or-poor' | 'shape-alphabet', n: number): RepRecord[] {
  const reps: RepRecord[] = [];
  for (let i = 0; i < n; i++) {
    const li = buildLoopItem(drill, siblingSeed('606060', i), EASY_KNOBS);
    const answer: Answer = {
      itemId: li.item.id,
      // answer the first choice every time → a mix of hits and misses
      choice: li.item.choices[0],
      confidence: 'lean',
      latencyMs: 1000 + i * 100,
    };
    reps.push({ li, answer, verdict: gradeLoopItem(li, answer) });
  }
  return reps;
}

describe('median', () => {
  it('handles odd/even/empty', () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 2, 3])).toBe(2.5);
    expect(median([])).toBe(0);
  });
});

describe('baseRateLine — measured from the served block, never folklore', () => {
  it('excess-or-poor reports the block’s actual poor share', () => {
    const reps = playBlock('excess-or-poor', 10);
    const poor = reps.filter((r) => r.li.item.groundTruth === 'POOR').length;
    expect(baseRateLine('excess-or-poor', reps)).toBe(
      `In this set, highlighted extremes were poor ${Math.round((poor / 10) * 100)}% of the time.`,
    );
  });

  it('shape-alphabet reports the served letter mix', () => {
    const reps = playBlock('shape-alphabet', 6);
    const line = baseRateLine('shape-alphabet', reps);
    expect(line).toContain('of 6');
    expect(line.startsWith("This set's mix — ")).toBe(true);
  });
});

describe('blockBotPoints', () => {
  it('AP = Σ (Brier_bot − Brier_you) × 100 against the uniform-chooser bot', () => {
    const reps = playBlock('excess-or-poor', 4).filter((r) => r.verdict.brier !== null);
    const expected = reps.reduce((s, r) => {
      const p = 0.75; // lean
      const you = r.verdict.correct ? (p - 1) ** 2 : p ** 2;
      const bot = r.verdict.correct ? (0.5 - 1) ** 2 : 0.5 ** 2;
      return s + (bot - you) * 100;
    }, 0);
    expect(blockBotPoints(reps)).toBeCloseTo(expected, 8);
  });

  it('is null when no rep carried a confidence tap', () => {
    const reps = playBlock('excess-or-poor', 2).map((r) => ({
      ...r,
      answer: { ...r.answer, confidence: null },
      verdict: { ...r.verdict, brier: null },
    }));
    expect(blockBotPoints(reps)).toBeNull();
  });
});

describe('calibrationLine — "When you said sure, you were right N%" (GDD §8 screen 3)', () => {
  it('reports the most-confident tap used, its measured hit rate, and the honesty clause', () => {
    const reps = playBlock('excess-or-poor', 10).filter((r) => r.verdict.brier !== null);
    const line = calibrationLine(reps)!;
    const hitPct = Math.round((reps.filter((r) => r.verdict.correct).length / reps.length) * 100);
    expect(line).toBe(
      `When you said lean, you were right ${hitPct}% — ${
        Math.abs(hitPct - 75) <= 5
          ? 'well calibrated this round'
          : hitPct < 75
            ? 'overconfident this round'
            : 'underconfident this round'
      }.`,
    );
  });

  it('prefers sure over lean, measures only that tap, judges vs its 90%', () => {
    const base = playBlock('excess-or-poor', 6).filter((r) => r.verdict.brier !== null);
    const reps: RepRecord[] = base.map((r, i) => ({
      ...r,
      answer: { ...r.answer, confidence: i === 0 ? ('sure' as const) : ('lean' as const) },
      verdict: { ...r.verdict, correct: i === 0, brier: 0.1 },
    }));
    expect(calibrationLine(reps)).toBe('When you said sure, you were right 100% — underconfident this round.');
  });

  it('is null when nothing carried a confidence tap or Brier component', () => {
    const reps = playBlock('excess-or-poor', 3).map((r) => ({
      ...r,
      answer: { ...r.answer, confidence: null },
      verdict: { ...r.verdict, brier: null },
    }));
    expect(calibrationLine(reps)).toBeNull();
    expect(calibrationLine([])).toBeNull();
  });
});

describe('summarize', () => {
  it('accuracy, median latency, misses and cardinals add up', () => {
    const reps = playBlock('excess-or-poor', 10);
    const s = summarize('excess-or-poor', reps);
    expect(s.reps).toBe(10);
    expect(s.correct).toBe(reps.filter((r) => r.verdict.correct).length);
    expect(s.accuracy).toBeCloseTo(s.correct / 10, 10);
    expect(s.medianLatencyMs).toBe(median(reps.map((r) => r.answer.latencyMs)));
    expect(s.parMs).toBe(2000);
    expect(s.misses).toBe(10 - s.correct);
    expect(s.baseRateLine).toContain('%');
    expect(s.calibrationLine).toBe(calibrationLine(reps));
  });
});
