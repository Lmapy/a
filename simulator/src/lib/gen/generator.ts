/* ============================================================================
   @gen/generator — SessionScript → bars + SessionLabels.

   GDD §7 pipeline: script skeleton → Brownian-bridge filler → λ engine
   (U-shaped seasonal × GARCH(1,1) × script boost) → structure planter →
   rejection-resampling verifier calling @core classifiers. Labels true by
   construction AND by measurement.

   Kernel numbers (GDD §7): GARCH α≈0.10 β≈0.85; standardized Student-t
   innovations (ν knob); seasonal vol open 2.5–3× midday, close 1.5–2×;
   morning-weighted jumps; profile mass = dwell × λ.

   Pure TS — zero DOM/Svelte imports. Randomness ONLY via named substreams:
   'skeleton' (in scripts.ts), 'noise', 'volume', 'prints', 'decoys'.

   STATUS: STUB — generateSession emits a deterministic random-walk session
   honoring the segment skeleton, with placeholder labels; NOT yet verified
   by classifiers. Owner: gen team.
   ========================================================================== */

import type { Bar, SessionLabels, SessionScript } from '../types';
import { Prng } from './prng';

/** A fully generated session: bars + the labels the grader consumes. */
export interface GeneratedSession {
  script: SessionScript;
  bars: Bar[];
  labels: SessionLabels;
}

/** U-shaped intraday seasonal volatility multiplier (open ~2.75×, close ~1.75× midday). */
export function seasonalVol(barIndex: number, nBars: number): number {
  const x = barIndex / Math.max(1, nBars - 1); // 0 at open, 1 at close
  // Smooth U: high at open, trough midday, partial recovery at close.
  const open = 2.75 * Math.exp(-x / 0.15);
  const close = 1.75 * Math.exp(-(1 - x) / 0.12);
  return 1 + open + close - Math.exp(-1 / 0.15) - Math.exp(-1 / 0.12);
}

/**
 * Generate a full session from a compiled script.
 * Deterministic: same script.seed => byte-identical output; consumes only the
 * 'noise', 'volume', and 'decoys' substreams.
 *
 * STUB: Gaussian random walk with seasonal vol + per-segment drift/volMult;
 * volume = seasonal λ × lognormal noise. No GARCH state, no bridge anchors,
 * no structure planting, no rejection loop yet.
 */
export function generateSession(
  script: SessionScript,
  basePrice = 5000,
): GeneratedSession {
  const prng = new Prng(script.seed);
  const noise = prng.stream('noise');
  const volume = prng.stream('volume');

  const bars: Bar[] = [];
  let price = basePrice;
  for (let t = 0; t < script.nBars; t++) {
    const seg =
      script.segments.find((s) => t >= s.startBar && t <= s.endBar) ??
      script.segments[script.segments.length - 1];
    const vol = 0.35 * seasonalVol(t, script.nBars) * seg.volMult;
    const o = price;
    const ret = seg.drift + vol * noise.nextGaussian();
    const c = o + ret;
    const wickUp = Math.abs(vol * 0.5 * noise.nextGaussian());
    const wickDn = Math.abs(vol * 0.5 * noise.nextGaussian());
    const h = Math.max(o, c) + wickUp;
    const l = Math.min(o, c) - wickDn;
    const lambda = seasonalVol(t, script.nBars) * seg.volumeBoost;
    const v = Math.max(1, Math.round(1000 * lambda * Math.exp(0.3 * volume.nextGaussian())));
    bars.push({ t, o, h, l, c, v });
    price = c;
  }

  const labels: SessionLabels = {
    dayType: script.dayType,
    blendWeight: 0,
    blendedToward: null,
    openType: script.openType,
    openLocation: script.openLocation,
    script: script.segments,
    ibWidthRatio: script.ibWidthRatio,
    oneTimeframingBreakBar: script.oneTimeframingBreakBar,
    extremes: [], // STUB: measured by classify.measureExcessTicks post-generation
    planted: [], // STUB: structure planter pending
    decoys: [], // STUB: decoy placement pending ('decoys' substream)
    acceptanceFlags: [], // STUB: per-bracket acceptance pending
    conditionalProbs: {}, // STUB: per-pool Monte-Carlo estimation pending
    ambiguityWeight: 0,
    poolBaseRates: {}, // STUB: declared by the item pool, never hard-coded
  };

  return { script, bars, labels };
}
