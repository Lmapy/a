import { describe, expect, it } from 'vitest';
import {
  CARDINAL_WEIGHT,
  GLICKO2_SCALE,
  INITIAL_RATING,
  INITIAL_RD,
  INITIAL_VOLATILITY,
  applyInactivity,
  expectedScore,
  fromMu,
  g,
  initialRating,
  ratingWeight,
  solveVolatility,
  targetItemRating,
  toMu,
  updateRating,
  updateRatingPeriod,
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

describe("Glickman's worked example (glicko2.pdf)", () => {
  // Player: r 1500, RD 200, σ 0.06, τ = 0.5.
  // Opponents: 1400/30 (win), 1550/100 (loss), 1700/300 (loss).
  const player = { nodeId: 'x', rating: 1500, rd: 200, volatility: 0.06, nAnswers: 0 };
  const results = [
    { itemRating: 1400, itemRd: 30, score: 1 },
    { itemRating: 1550, itemRd: 100, score: 0 },
    { itemRating: 1700, itemRd: 300, score: 0 },
  ];

  it('pins the intermediate g/E values of the paper (step 3)', () => {
    const mu = toMu(player.rating);
    expect(g(30 / GLICKO2_SCALE)).toBeCloseTo(0.9955, 4);
    expect(g(100 / GLICKO2_SCALE)).toBeCloseTo(0.9531, 4);
    expect(g(300 / GLICKO2_SCALE)).toBeCloseTo(0.7242, 4);
    expect(expectedScore(mu, toMu(1400), 30 / GLICKO2_SCALE)).toBeCloseTo(0.639, 3);
    expect(expectedScore(mu, toMu(1550), 100 / GLICKO2_SCALE)).toBeCloseTo(0.432, 3);
    expect(expectedScore(mu, toMu(1700), 300 / GLICKO2_SCALE)).toBeCloseTo(0.303, 3);
  });

  it("reproduces the paper's r' = 1464.06, RD' = 151.52, σ' = 0.05999", () => {
    const next = updateRatingPeriod(player, results);
    expect(next.rating).toBeCloseTo(1464.06, 1);
    expect(next.rd).toBeCloseTo(151.52, 1);
    expect(next.volatility).toBeCloseTo(0.05999, 4);
    expect(next.nAnswers).toBe(3);
  });

  it('the volatility solver converges to σ′ ≈ 0.05999 for the paper inputs', () => {
    // Paper intermediates: v = 1.7785, Δ = −0.4834, φ = 200/173.7178 = 1.1513.
    const sigma = solveVolatility(200 / GLICKO2_SCALE, 1.7785, -0.4834, 0.06, 0.5);
    expect(sigma).toBeCloseTo(0.05999, 4);
  });
});

describe('updateRating (single-item period)', () => {
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
    const cardinal = updateRating(fresh, 1500, 60, 0, CARDINAL_WEIGHT);
    expect(fresh.rating - cardinal.rating).toBeGreaterThan(fresh.rating - normal.rating);
  });

  it('ratingWeight maps cardinal → 3, normal → 1', () => {
    expect(ratingWeight(true)).toBe(3);
    expect(ratingWeight(false)).toBe(1);
  });

  it('does not mutate the input state', () => {
    const before = { ...fresh };
    updateRating(fresh, 1600, 60, 1);
    expect(fresh).toEqual(before);
  });

  it('a weight-3 result equals the same result listed three times', () => {
    const weighted = updateRating(fresh, 1500, 60, 0, 3);
    const listed = updateRatingPeriod(fresh, [
      { itemRating: 1500, itemRd: 60, score: 0 },
      { itemRating: 1500, itemRd: 60, score: 0 },
      { itemRating: 1500, itemRd: 60, score: 0 },
    ]);
    expect(weighted.rating).toBeCloseTo(listed.rating, 9);
    expect(weighted.rd).toBeCloseTo(listed.rd, 9);
    expect(weighted.volatility).toBeCloseTo(listed.volatility, 9);
  });

  it('volatility rises after a shocking result for a confident (low-RD) player', () => {
    const confident = { nodeId: 'x', rating: 1500, rd: 50, volatility: 0.06, nAnswers: 100 };
    // Losing repeatedly to a much weaker item is way outside expectation.
    const shocked = updateRatingPeriod(
      confident,
      Array.from({ length: 10 }, () => ({ itemRating: 900, itemRd: 60, score: 0 })),
    );
    expect(shocked.volatility).toBeGreaterThan(confident.volatility);
  });

  it('repeated expected wins converge rating upward and shrink RD toward a floor', () => {
    let s = initialRating('poc-va-snap');
    for (let i = 0; i < 50; i++) s = updateRating(s, s.rating + 100, 60, 1);
    expect(s.rating).toBeGreaterThan(1700);
    expect(s.rd).toBeLessThan(80);
  });
});

describe('applyInactivity', () => {
  it('grows RD (uncertainty returns with disuse) but never past the initial 350', () => {
    const seasoned = { nodeId: 'x', rating: 1600, rd: 80, volatility: 0.06, nAnswers: 40 };
    const idle = applyInactivity(seasoned);
    expect(idle.rd).toBeGreaterThan(seasoned.rd);
    expect(idle.rating).toBe(seasoned.rating);
    const maxed = applyInactivity({ ...seasoned, rd: INITIAL_RD });
    expect(maxed.rd).toBeLessThanOrEqual(INITIAL_RD);
  });

  it('an empty rating period is an inactive period', () => {
    const seasoned = { nodeId: 'x', rating: 1600, rd: 80, volatility: 0.06, nAnswers: 40 };
    expect(updateRatingPeriod(seasoned, [])).toEqual(applyInactivity(seasoned));
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

describe('constants', () => {
  it('pins the launch defaults', () => {
    expect(INITIAL_VOLATILITY).toBe(0.06);
    expect(CARDINAL_WEIGHT).toBe(3);
  });
});
