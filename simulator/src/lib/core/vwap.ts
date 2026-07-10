/* ============================================================================
   @core/vwap — session VWAP with volume-weighted standard-deviation bands.

   Reference: /home/user/a/scripts/figures/fig-12-vp-tpo-vwap.py.
   VWAP = Σ(p·v)/Σ(v) cumulatively from the session open (guide §2.5:
   "VWAP is the mean; POC is the mode").

   Pure TS — zero DOM/Svelte imports. FULLY IMPLEMENTED (simple math, but it
   must live here: no UI-layer reimplementation, per the single-library rule).
   ========================================================================== */

import type { Bar, VwapSeries } from '../types';

/** Typical price used as the bar's volume-weighted price sample (HLC/3). */
export function typicalPrice(bar: Bar): number {
  return (bar.h + bar.l + bar.c) / 3;
}

/**
 * Compute the cumulative session VWAP and ±1σ/±2σ volume-weighted bands.
 * Bands use the volume-weighted variance of typical price around the running
 * VWAP: σ²_i = Σ(v·p²)/Σ(v) − vwap_i² (clamped at 0 for float safety).
 *
 * @param bars bars from the session open (or an anchor), in time order
 */
export function computeVwap(bars: Bar[]): VwapSeries {
  const n = bars.length;
  const vwap = new Array<number>(n);
  const upper1 = new Array<number>(n);
  const lower1 = new Array<number>(n);
  const upper2 = new Array<number>(n);
  const lower2 = new Array<number>(n);

  let sumV = 0;
  let sumPV = 0;
  let sumPPV = 0;
  for (let i = 0; i < n; i++) {
    const p = typicalPrice(bars[i]);
    const v = bars[i].v;
    sumV += v;
    sumPV += p * v;
    sumPPV += p * p * v;
    const m = sumV > 0 ? sumPV / sumV : p;
    const variance = sumV > 0 ? Math.max(0, sumPPV / sumV - m * m) : 0;
    const sd = Math.sqrt(variance);
    vwap[i] = m;
    upper1[i] = m + sd;
    lower1[i] = m - sd;
    upper2[i] = m + 2 * sd;
    lower2[i] = m - 2 * sd;
  }
  return { vwap, upper1, lower1, upper2, lower2 };
}
