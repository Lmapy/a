/* ============================================================================
   @schedule/glicko — vendored Glicko-2 (GDD §5: per-skill rating; tools doc:
   "vendor ~200 lines, golden-tested against the glicko2.ts npm package").

   Pure TS — zero DOM/Svelte imports, zero deps.
   STATUS: constants + scale conversion implemented; the rating update is a
   simplified single-opponent stub pending the full Glicko-2 iteration
   (volatility solver). Owner: schedule team.
   ========================================================================== */

import type { RatingState } from '../types';

/** Glicko-2 system constant τ (volatility change constraint). */
export const TAU = 0.5;
/** Display-scale defaults for a fresh node. */
export const INITIAL_RATING = 1500;
export const INITIAL_RD = 350;
export const INITIAL_VOLATILITY = 0.06;
/** Glicko display scale <-> internal (mu/phi) scale factor. */
export const GLICKO2_SCALE = 173.7178;

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

/**
 * One rating-period update against a single item (the drill case: player vs
 * item, score ∈ [0,1]; cardinal errors pass weight = 3, GDD §5).
 *
 * STUB: implements the Glicko-2 variance/improvement step with volatility
 * held constant (no Illinois iteration yet). Monotone and bounded — safe for
 * scaffold integration; replace with the full algorithm + golden tests.
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
  const mu = toMu(player.rating);
  const phi = player.rd / GLICKO2_SCALE;
  const muJ = toMu(itemRating);
  const phiJ = itemRd / GLICKO2_SCALE;

  const E = expectedScore(mu, muJ, phiJ);
  const gj = g(phiJ);
  const v = 1 / (weight * gj * gj * E * (1 - E));
  const delta = v * weight * gj * (score - E);

  const phiStar = Math.sqrt(phi * phi + player.volatility * player.volatility);
  const phiNew = 1 / Math.sqrt(1 / (phiStar * phiStar) + 1 / v);
  const muNew = mu + phiNew * phiNew * weight * gj * (score - E);

  return {
    nodeId: player.nodeId,
    rating: fromMu(muNew),
    rd: phiNew * GLICKO2_SCALE,
    volatility: player.volatility, // STUB: full Glicko-2 updates σ via iteration
    nAnswers: player.nAnswers + 1,
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
