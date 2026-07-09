import { describe, expect, it } from 'vitest';
import {
  INITIAL_RATING,
  INITIAL_RD,
  expectedScore,
  fromMu,
  g,
  initialRating,
  targetItemRating,
  toMu,
  updateRating,
} from './glicko';

describe('scale conversion', () => {
  it('toMu/fromMu round-trip and pin the 1500 origin', () => {
    expect(toMu(INITIAL_RATING)).toBe(0);
    expect(fromMu(0)).toBe(INITIAL_RATING);
    expect(fromMu(toMu(1837))).toBeCloseTo(1837, 9);
  });
});

describe('g and expectedScore', () => {
  it('g(0) = 1 and decreases with phi', () => {
    expect(g(0)).toBe(1);
    expect(g(2)).toBeLessThan(g(1));
  });
  it('expected score is 0.5 vs an equal opponent and >0.5 vs a weaker one', () => {
    expect(expectedScore(0, 0, 0.5)).toBeCloseTo(0.5, 9);
    expect(expectedScore(1, 0, 0.5)).toBeGreaterThan(0.5);
  });
});

describe('updateRating (stub contract)', () => {
  const fresh = initialRating('shape-alphabet');

  it('win raises rating, loss lowers it, RD shrinks either way', () => {
    const win = updateRating(fresh, 1500, 60, 1);
    const loss = updateRating(fresh, 1500, 60, 0);
    expect(win.rating).toBeGreaterThan(fresh.rating);
    expect(loss.rating).toBeLessThan(fresh.rating);
    expect(win.rd).toBeLessThan(INITIAL_RD);
    expect(loss.rd).toBeLessThan(INITIAL_RD);
    expect(win.nAnswers).toBe(1);
  });

  it('cardinal errors (weight 3) move rating more than normal misses (GDD §5)', () => {
    const normal = updateRating(fresh, 1500, 60, 0, 1);
    const cardinal = updateRating(fresh, 1500, 60, 0, 3);
    expect(fresh.rating - cardinal.rating).toBeGreaterThan(fresh.rating - normal.rating);
  });

  it('does not mutate the input state', () => {
    const before = { ...fresh };
    updateRating(fresh, 1600, 60, 1);
    expect(fresh).toEqual(before);
  });
});

describe('targetItemRating (adaptive ~85% selection)', () => {
  it('yields an item the player is expected to beat ~85% of the time', () => {
    const player = initialRating('poc-va-snap');
    const itemR = targetItemRating(player, 0.85);
    expect(itemR).toBeLessThan(player.rating); // easier than the player
    const E = expectedScore(toMu(player.rating), toMu(itemR), 0);
    expect(E).toBeCloseTo(0.85, 2);
  });
});
