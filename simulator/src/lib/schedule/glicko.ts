/* ============================================================================
   @schedule/glicko — vendored Glicko-2 (GDD §5: per-skill rating).

   Full implementation of Glickman's algorithm ("Example of the Glicko-2
   system", glicko.net/glicko/glicko2.pdf): rating μ, deviation φ, and
   volatility σ, with the Illinois-algorithm volatility solver of Step 5.
   Unit tests pin the paper's worked example (1500/200/0.06 vs three
   opponents → r' 1464.06, RD' 151.52, σ' 0.05999).

   Extension for the drill case (GDD §5): each result carries a rating
   `weight` — 1 for normal items, 3 for cardinal errors (Drill I trend day
   called "balance"). A weight-w result enters the update sums as if the
   same game were played w times, which is exactly how Glicko-2 composes
   repeated results within one rating period.

   Pure TS — zero DOM/Svelte imports, zero deps.
   ========================================================================== */

import type { RatingState } from '../types';

/** Glicko-2 system constant τ (volatility change constraint; paper: 0.3–1.2). */
export const TAU = 0.5;
/** Convergence tolerance ε for the volatility solver (paper Step 5.1). */
export const CONVERGENCE_EPS = 0.000001;
/** Display-scale defaults for a fresh node. */
export const INITIAL_RATING = 1500;
/**
 * Initial rating deviation — TUNED for the per-answer update cadence, not the
 * paper's 350. The drill loop scores every rep as a one-result rating period
 * against a knob-derived item (RD 60) at ~85% expected success; with RD 350 a
 * single first-rep miss swings −439 points and a 7/10 opening block nets
 * −211, which whipsaws the adaptive selector and contradicts the GDD §8
 * result-screen scale ("Rating 1510 → 1518"). At 100 the same events measure
 * −49 and −82, while established-play behavior is unchanged: against ITEM_RD
 * 60 with σ = 0.06 the deviation converges to ≈75 regardless of its start,
 * so only the cold-start transient is tamed. Inactivity growth stays capped
 * at this value (applyInactivity).
 */
export const INITIAL_RD = 100;
export const INITIAL_VOLATILITY = 0.06;
/** Glicko display scale <-> internal (mu/phi) scale factor. */
export const GLICKO2_SCALE = 173.7178;

/** Rating weight applied to cardinal errors (GDD §5: Drill I, 3× rating). */
export const CARDINAL_WEIGHT = 3;

/** Rating weight for a verdict: 3 for cardinal errors, 1 otherwise (GDD §5). */
export function ratingWeight(cardinal: boolean): number {
  return cardinal ? CARDINAL_WEIGHT : 1;
}

/** Fresh rating state for a node. */
export function initialRating(nodeId: string): RatingState {
  return {
    nodeId,
    rating: INITIAL_RATING,
    rd: INITIAL_RD,
    volatility: INITIAL_VOLATILITY,
    nAnswers: 0,
  };
}

/** Display rating → internal μ. */
export function toMu(rating: number): number {
  return (rating - INITIAL_RATING) / GLICKO2_SCALE;
}

/** Internal μ → display rating. */
export function fromMu(mu: number): number {
  return mu * GLICKO2_SCALE + INITIAL_RATING;
}

/** Glicko g(φ) factor. */
export function g(phi: number): number {
  return 1 / Math.sqrt(1 + (3 * phi * phi) / (Math.PI * Math.PI));
}

/** Expected score of a player (μ, vs opponent μj with deviation φj). */
export function expectedScore(mu: number, muJ: number, phiJ: number): number {
  return 1 / (1 + Math.exp(-g(phiJ) * (mu - muJ)));
}

/** One opponent/item result within a rating period. */
export interface GameResult {
  /** Item (opponent) rating on the display scale. */
  itemRating: number;
  /** Item rating deviation (knob-derived items: use 60). */
  itemRd: number;
  /** Outcome in [0, 1]. */
  score: number;
  /** Rating weight (1 normal, 3 cardinal error). Default 1. */
  weight?: number;
}

/**
 * Step 5 of the paper: solve for the new volatility σ' with the Illinois
 * variant of regula falsi on f(x) = ln(σ'²) likelihood derivative.
 *
 * @param phi    current deviation on the internal scale
 * @param v      estimated variance of the rating from game outcomes (Step 3)
 * @param delta  estimated improvement Δ (Step 4)
 * @param sigma  current volatility
 * @param tau    system constant
 */
export function solveVolatility(
  phi: number,
  v: number,
  delta: number,
  sigma: number,
  tau = TAU,
): number {
  const a = Math.log(sigma * sigma);
  const phi2 = phi * phi;
  const d2 = delta * delta;

  const f = (x: number): number => {
    const ex = Math.exp(x);
    return (ex * (d2 - phi2 - v - ex)) / (2 * (phi2 + v + ex) * (phi2 + v + ex)) - (x - a) / (tau * tau);
  };

  // Step 5.2: initial bracket [A, B].
  let A = a;
  let B: number;
  if (d2 > phi2 + v) {
    B = Math.log(d2 - phi2 - v);
  } else {
    let k = 1;
    while (f(a - k * tau) < 0) k++;
    B = a - k * tau;
  }

  // Step 5.3–5.4: Illinois iteration until |B − A| ≤ ε.
  let fA = f(A);
  let fB = f(B);
  while (Math.abs(B - A) > CONVERGENCE_EPS) {
    const C = A + ((A - B) * fA) / (fB - fA);
    const fC = f(C);
    if (fC * fB <= 0) {
      A = B;
      fA = fB;
    } else {
      fA = fA / 2;
    }
    B = C;
    fB = fC;
  }

  // Step 5.5: σ' = e^(A/2).
  return Math.exp(A / 2);
}

/**
 * Full Glicko-2 rating-period update against a set of results (paper Steps
 * 2–8). Does not mutate the input. Empty result set = an inactive period:
 * only the deviation grows (Step 6 note), capped at INITIAL_RD.
 */
export function updateRatingPeriod(player: RatingState, results: GameResult[]): RatingState {
  if (results.length === 0) return applyInactivity(player);

  // Step 2: convert to the internal scale.
  const mu = toMu(player.rating);
  const phi = player.rd / GLICKO2_SCALE;

  // Step 3 (v) and Step 4 (Δ) sums; weight-w results count w times.
  let vInv = 0;
  let dSum = 0;
  for (const r of results) {
    const w = r.weight ?? 1;
    const muJ = toMu(r.itemRating);
    const phiJ = r.itemRd / GLICKO2_SCALE;
    const gj = g(phiJ);
    const E = expectedScore(mu, muJ, phiJ);
    vInv += w * gj * gj * E * (1 - E);
    dSum += w * gj * (r.score - E);
  }
  const v = 1 / vInv;
  const delta = v * dSum;

  // Step 5: new volatility σ'.
  const sigmaNew = solveVolatility(phi, v, delta, player.volatility);

  // Step 6: pre-period deviation φ*.
  const phiStar = Math.sqrt(phi * phi + sigmaNew * sigmaNew);

  // Step 7: new deviation φ' and rating μ'.
  const phiNew = 1 / Math.sqrt(1 / (phiStar * phiStar) + 1 / v);
  const muNew = mu + phiNew * phiNew * dSum;

  // Step 8: convert back to the display scale.
  return {
    nodeId: player.nodeId,
    rating: fromMu(muNew),
    rd: phiNew * GLICKO2_SCALE,
    volatility: sigmaNew,
    nAnswers: player.nAnswers + results.length,
  };
}

/**
 * Single-item update — the per-drill-answer case (player vs item, score in
 * [0,1]; cardinal errors pass weight = 3, GDD §5). Full Glicko-2 with a
 * one-result rating period.
 *
 * @param player  current state (not mutated)
 * @param itemRating  item rating on the display scale
 * @param itemRd      item rating deviation (knob-derived items: use 60)
 * @param score       outcome in [0,1]
 * @param weight      rating weight (1 normal, 3 cardinal errors)
 */
export function updateRating(
  player: RatingState,
  itemRating: number,
  itemRd: number,
  score: number,
  weight = 1,
): RatingState {
  return updateRatingPeriod(player, [{ itemRating, itemRd, score, weight }]);
}

/**
 * An inactive rating period: deviation grows toward INITIAL_RD (uncertainty
 * returns with disuse — the "rusty" mechanic's rating-side counterpart).
 */
export function applyInactivity(player: RatingState): RatingState {
  const phi = player.rd / GLICKO2_SCALE;
  const phiNew = Math.sqrt(phi * phi + player.volatility * player.volatility);
  return {
    ...player,
    rd: Math.min(phiNew * GLICKO2_SCALE, INITIAL_RD),
  };
}

/**
 * Adaptive selection target (GDD §5): pick items where the player's expected
 * score ≈ 0.85. Returns the item display-rating that satisfies that for the
 * given player state.
 */
export function targetItemRating(player: RatingState, targetE = 0.85): number {
  // Invert E = 1/(1+exp(-(mu - muJ))) with g≈1: muJ = mu - ln(E/(1-E))
  const mu = toMu(player.rating);
  const muJ = mu - Math.log(targetE / (1 - targetE));
  return fromMu(muJ);
}
