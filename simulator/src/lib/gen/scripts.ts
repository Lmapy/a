/* ============================================================================
   @gen/scripts — script compiler: (day type, open type, difficulty knobs)
   → SessionScript (segment skeleton with drift/vol/volume anchors).

   GDD §7: "Day type + open type compile to a segment script (per-segment
   drift, volatility, volume-intensity, anchors)". Trend-day contract:
   IB width 0.3–0.5× normal, pullbacks ≤40% of prior impulse, close in the
   extreme 15% of range, volume rising on impulses.

   Pure TS — zero DOM/Svelte imports. Randomness ONLY via the 'skeleton'
   substream of the passed Prng.

   STATUS: STUB — emits a minimal two-segment balance skeleton for any input.
   Owner: gen team.
   ========================================================================== */

import type { DayType, OpenLocation, OpenType, Segment, SessionScript } from '../types';
import { Prng } from './prng';

/** Continuous difficulty knobs (GDD §7), all mapped to adaptive selection. */
export interface DifficultyKnobs {
  /** Master signal-to-noise dial; 0 = clean, 1 = max noise. */
  noiseGain: number;
  /** Student-t degrees of freedom: 4 = hard/spiky, 8 = easy/tame. */
  nu: number;
  /** Day-type ambiguity blend weight in [0, 0.5]. */
  ambiguity: number;
  /** Number of decoy near-structures, 0–2. */
  decoys: number;
  /** News-jump intensity multiplier. */
  jumpIntensity: number;
  /** LVN depth: 1 = deep/obvious valley, 0 = shallow/hard. */
  lvnDepth: number;
}

/** Sensible mid-difficulty defaults. */
export const DEFAULT_KNOBS: DifficultyKnobs = {
  noiseGain: 0.5,
  nu: 6,
  ambiguity: 0,
  decoys: 1,
  jumpIntensity: 0,
  lvnDepth: 0.7,
};

/** RTH session length in 1-min bars (6.5 h). */
export const SESSION_BARS = 390;

/**
 * Compile a session script from taxonomy + knobs.
 * Deterministic: consumes only prng.stream('skeleton').
 *
 * STUB: returns a two-segment all-balance skeleton with zero drift; real
 * implementation branches per day type per GDD §7 kernel numbers.
 */
export function compileScript(
  prng: Prng,
  dayType: DayType,
  openType: OpenType,
  openLocation: OpenLocation,
  knobs: DifficultyKnobs = DEFAULT_KNOBS,
  paramsVersion = 'v1.0.0',
): SessionScript {
  const skeleton = prng.stream('skeleton');
  const splitBar = skeleton.nextInt(150, 240); // placeholder anchor
  const segments: Segment[] = [
    {
      startBar: 0,
      endBar: splitBar,
      regime: 'balance',
      drift: 0,
      volMult: 1,
      volumeBoost: 1,
      oneTimeframing: false,
    },
    {
      startBar: splitBar + 1,
      endBar: SESSION_BARS - 1,
      regime: 'balance',
      drift: 0,
      volMult: 1,
      volumeBoost: 1,
      oneTimeframing: false,
    },
  ];
  return {
    seed: prng.seedString,
    paramsVersion,
    dayType,
    openType,
    openLocation,
    segments,
    nBars: SESSION_BARS,
    rowStep: 0.5,
    ibWidthRatio: 1,
    oneTimeframingBreakBar: null,
  };
}
