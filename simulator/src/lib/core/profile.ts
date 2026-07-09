/* ============================================================================
   @core/profile — volume-at-price histogram, POC, 70% value area, HVN/LVN.

   Reference implementations: /home/user/a/scripts/figures/fig-01-volume-profile-anatomy.py
   (POC + VA expansion), fig-11-lvn-behavior.py / fig-02-profile-shapes.py
   (HVN bulges, LVN necks). Golden-file parity is pinned by
   __fixtures__/fig01.ts and __fixtures__/fig12.ts (GDD §10-A).

   SINGLE-LIBRARY RULE (GDD §9): this module is used identically by
   generation-verification, rendering, and grading. No other code may compute
   POC/VA/HVN/LVN. Pure TS — zero DOM/Svelte imports.
   ========================================================================== */

import type { Bar, Profile, RowRange } from '../types';

/**
 * Build a volume profile from bars (tick-size aware via rowStep).
 *
 * Binning: each bar's volume is distributed uniformly across every row its
 * [l, h] range touches (the standard minute-bar dwell approximation — with
 * 1-min bars the intrabar path is unknown, so equal dwell per touched row is
 * the unbiased assignment). A bar with l === h puts all volume on one row,
 * which makes this binning exactly equal to the Python references that bin a
 * minute price path point-by-point (fig-12 `vap`).
 *
 * @param bars     session bars (any contiguous slice)
 * @param rowStep  price per row, in points (> 0)
 */
export function buildProfile(bars: Bar[], rowStep: number): Profile {
  if (bars.length === 0 || rowStep <= 0) {
    throw new Error('buildProfile: need at least one bar and rowStep > 0');
  }
  let lo = Infinity;
  let hi = -Infinity;
  for (const b of bars) {
    if (b.l < lo) lo = b.l;
    if (b.h > hi) hi = b.h;
  }
  const minPrice = Math.floor(lo / rowStep) * rowStep;
  const nRows = Math.max(1, Math.floor((hi - minPrice) / rowStep) + 1);
  const rows = new Array<number>(nRows).fill(0);
  let totalVolume = 0;
  for (const b of bars) {
    const rLo = Math.min(nRows - 1, Math.max(0, Math.floor((b.l - minPrice) / rowStep)));
    const rHi = Math.min(nRows - 1, Math.max(rLo, Math.floor((b.h - minPrice) / rowStep)));
    const share = b.v / (rHi - rLo + 1);
    for (let r = rLo; r <= rHi; r++) rows[r] += share;
    totalVolume += b.v;
  }
  return profileFromRows(rows, minPrice, rowStep, totalVolume);
}

/**
 * Assemble a Profile from an existing volume-at-price histogram (used by
 * buildProfile and by tests/verification code that already own the rows).
 */
export function profileFromRows(
  rows: number[],
  minPrice: number,
  rowStep: number,
  totalVolume?: number,
): Profile {
  const poc = findPoc(rows);
  const { vah, val } = expandValueArea(rows, poc, 0.7);
  return {
    minPrice,
    rowStep,
    rows,
    poc,
    vah,
    val,
    totalVolume: totalVolume ?? rows.reduce((a, b) => a + b, 0),
    hvnRanges: findHvnRanges(rows),
    lvnRanges: findLvnRanges(rows),
  };
}

/**
 * POC row = argmax volume. Tie-break: the row closest to the middle of the
 * profile (the documented CBOT convention — guide Part II §2.2 note), then
 * the lower row.
 */
export function findPoc(rows: number[]): number {
  if (rows.length === 0) throw new Error('findPoc: empty profile');
  const mid = (rows.length - 1) / 2;
  let best = 0;
  for (let i = 1; i < rows.length; i++) {
    if (
      rows[i] > rows[best] ||
      (rows[i] === rows[best] && Math.abs(i - mid) < Math.abs(best - mid))
    ) {
      best = i;
    }
  }
  return best;
}

/**
 * The canonical 70% value-area expansion (guide Part II §2.2, the
 * TradingView-documented variant, and fig-01 lines 34–47): start at the POC
 * row; repeatedly compare the single next untaken row above vs below the
 * current area; absorb the LARGER; stop when accumulated volume ≥
 * targetFraction × total.
 *
 * Tie rule: equal rows go to the row CLOSER to the POC (guide §2.2 step 4);
 * when both candidates are equidistant from the POC the tie goes UP, which is
 * exactly the Python reference's `if up >= dn` (expansion starts symmetric at
 * the POC, so every fig-01 tie is the equidistant case on the first divergent
 * step). Float fixtures never tie, so golden parity holds bit-exactly.
 *
 * NOTE for core team: GDD §4 Drill A calls this "the 70% two-row expansion",
 * but the fig-01 reference — which golden-file parity (GDD §10-A, blocking CI)
 * pins us to — expands one row at a time. This port matches the Python and
 * the guide. If the reference changes to two-row pairs, change it here and
 * regenerate golden files; never fork the algorithm elsewhere.
 *
 * Returns row indices { vah, val } (vah >= val; both inside [0, rows.length)).
 */
export function expandValueArea(
  rows: number[],
  poc: number,
  targetFraction = 0.7,
): { vah: number; val: number } {
  if (rows.length === 0) throw new Error('expandValueArea: empty profile');
  const total = rows.reduce((a, b) => a + b, 0);
  const target = total * targetFraction;
  let lo = poc;
  let hi = poc;
  let acc = rows[poc];
  while (acc < target && (lo > 0 || hi < rows.length - 1)) {
    const up = hi + 1 < rows.length ? rows[hi + 1] : -Infinity;
    const dn = lo - 1 >= 0 ? rows[lo - 1] : -Infinity;
    let goUp: boolean;
    if (up > dn) goUp = true;
    else if (dn > up) goUp = false;
    // tie: the row closer to the POC wins; equidistant -> up (fig-01 parity)
    else goUp = hi + 1 - poc <= poc - (lo - 1);
    if (goUp) {
      hi += 1;
      acc += up;
    } else {
      lo -= 1;
      acc += dn;
    }
  }
  return { vah: hi, val: lo };
}

/* ----------------------------------------------------------------------------
   HVN / LVN detection (fig-11 / fig-02 semantics)
   -------------------------------------------------------------------------- */

/** Tunable sensitivity for HVN/LVN detection. All fractions are of a peak height. */
export interface NodeDetectOptions {
  /** Passes of the [0.25, 0.5, 0.25] smoothing kernel before peak-finding. */
  smoothPasses?: number;
  /** Minimum peak height as a fraction of the global max to count as an HVN. */
  peakFrac?: number;
  /** Valleys SHALLOWER than this × the smaller flanking peak merge the peaks into one node. */
  mergeValleyFrac?: number;
  /** An HVN range extends from its peak while volume ≥ this × the peak height. */
  shoulderFrac?: number;
  /** Valleys DEEPER than this × the smaller flanking peak qualify as LVNs. */
  lvnFrac?: number;
}

const NODE_DEFAULTS: Required<NodeDetectOptions> = {
  smoothPasses: 2,
  peakFrac: 0.25,
  mergeValleyFrac: 0.65,
  shoulderFrac: 0.55,
  lvnFrac: 0.5,
};

/** One detected high-volume node: its peak row, smoothed height, and row range. */
export interface VolumeNode {
  /** Row index of the node's peak (argmax of raw volume inside the plateau). */
  peak: number;
  /** Smoothed volume at the peak (the height thresholds are measured against). */
  height: number;
  /** Rows belonging to the node (its bulge shoulders). */
  range: RowRange;
}

/** [0.25, 0.5, 0.25] kernel smoothing with edge renormalization. */
export function smoothRows(rows: number[], passes = 1): number[] {
  let s = rows.slice();
  for (let p = 0; p < passes; p++) {
    const next = new Array<number>(s.length);
    for (let i = 0; i < s.length; i++) {
      let acc = 0.5 * s[i];
      let w = 0.5;
      if (i > 0) {
        acc += 0.25 * s[i - 1];
        w += 0.25;
      }
      if (i < s.length - 1) {
        acc += 0.25 * s[i + 1];
        w += 0.25;
      }
      next[i] = acc / w;
    }
    s = next;
  }
  return s;
}

/**
 * Detect high-volume nodes: smooth, find local maxima (plateau-aware), drop
 * peaks under peakFrac × global max, merge peaks not separated by a real
 * valley (mergeValleyFrac), and expand each survivor into its bulge range
 * (shoulderFrac, clipped at the valley argmin between neighbors).
 * Returned in row order (low price first).
 */
export function detectVolumeNodes(
  rows: number[],
  opts: NodeDetectOptions = {},
): VolumeNode[] {
  const o = { ...NODE_DEFAULTS, ...opts };
  const n = rows.length;
  if (n === 0) return [];
  const s = smoothRows(rows, o.smoothPasses);
  const globalMax = Math.max(...s);
  if (globalMax <= 0) return [];

  // ---- plateau-aware local maxima -----------------------------------------
  let peaks: { peak: number; height: number }[] = [];
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && s[j + 1] === s[i]) j++;
    const leftOk = i === 0 || s[i - 1] < s[i];
    const rightOk = j === n - 1 || s[j + 1] < s[i];
    if (leftOk && rightOk) {
      // representative row = raw-volume argmax inside the plateau
      let rep = i;
      for (let k = i + 1; k <= j; k++) if (rows[k] > rows[rep]) rep = k;
      peaks.push({ peak: rep, height: s[rep] });
    }
    i = j + 1;
  }

  // ---- prominence filter ---------------------------------------------------
  peaks = peaks.filter((p) => p.height >= o.peakFrac * globalMax);

  // ---- merge peaks without a real valley between them ----------------------
  const valleyMin = (a: number, b: number): number => {
    let m = Infinity;
    for (let k = a + 1; k < b; k++) if (s[k] < m) m = s[k];
    return m === Infinity ? Math.min(s[a], s[b]) : m;
  };
  let merged = true;
  while (merged && peaks.length > 1) {
    merged = false;
    for (let k = 0; k + 1 < peaks.length; k++) {
      const a = peaks[k];
      const b = peaks[k + 1];
      if (valleyMin(a.peak, b.peak) > o.mergeValleyFrac * Math.min(a.height, b.height)) {
        peaks.splice(a.height >= b.height ? k + 1 : k, 1);
        merged = true;
        break;
      }
    }
  }

  // ---- expand each peak into its bulge range -------------------------------
  return peaks.map((p, k) => {
    let leftBound = 0;
    let rightBound = n - 1;
    if (k > 0) {
      let m = peaks[k - 1].peak + 1;
      for (let q = m; q < p.peak; q++) if (s[q] < s[m]) m = q;
      leftBound = m;
    }
    if (k + 1 < peaks.length) {
      let m = p.peak + 1;
      for (let q = m; q < peaks[k + 1].peak; q++) if (s[q] < s[m]) m = q;
      rightBound = m;
    }
    const cut = o.shoulderFrac * p.height;
    let lo = p.peak;
    let hi = p.peak;
    while (lo - 1 >= leftBound && s[lo - 1] >= cut) lo--;
    while (hi + 1 <= rightBound && s[hi + 1] >= cut) hi++;
    return { peak: p.peak, height: p.height, range: { lo, hi } };
  });
}

/**
 * High-volume-node bulge ranges (fig-11 semantics: HVN = acceptance bulge),
 * DOMINANT FIRST (tallest peak first).
 */
export function findHvnRanges(rows: number[], opts: NodeDetectOptions = {}): RowRange[] {
  return detectVolumeNodes(rows, opts)
    .slice()
    .sort((a, b) => b.height - a.height)
    .map((nd) => nd.range);
}

/**
 * Low-volume-node valley/neck ranges between HVN bulges (fig-11: LVN =
 * rejection air pocket; fig-02 B-shape neck), DEEPEST FIRST (lowest volume
 * relative to its flanking peaks first). A valley qualifies when its minimum
 * is ≤ lvnFrac × the smaller flanking peak.
 */
export function findLvnRanges(rows: number[], opts: NodeDetectOptions = {}): RowRange[] {
  const o = { ...NODE_DEFAULTS, ...opts };
  const nodes = detectVolumeNodes(rows, opts);
  if (nodes.length < 2) return [];
  const s = smoothRows(rows, o.smoothPasses);
  const lvns: { range: RowRange; depth: number }[] = [];
  for (let k = 0; k + 1 < nodes.length; k++) {
    const a = nodes[k];
    const b = nodes[k + 1];
    let m = a.peak + 1;
    for (let q = m; q < b.peak; q++) if (s[q] < s[m]) m = q;
    if (m >= b.peak) continue; // adjacent peaks, no interior valley
    const flank = Math.min(a.height, b.height);
    const cut = o.lvnFrac * flank;
    if (s[m] > cut) continue;
    let lo = m;
    let hi = m;
    while (lo - 1 > a.peak && s[lo - 1] <= cut) lo--;
    while (hi + 1 < b.peak && s[hi + 1] <= cut) hi++;
    lvns.push({ range: { lo, hi }, depth: s[m] / flank });
  }
  return lvns.sort((x, y) => x.depth - y.depth).map((l) => l.range);
}

/** Price of a row's center. Inverse of priceToRow at every zoom (GDD §10-A). */
export function rowToPrice(profile: Pick<Profile, 'minPrice' | 'rowStep'>, row: number): number {
  return profile.minPrice + row * profile.rowStep;
}

/** Nearest row index for a price (snap-to-row). */
export function priceToRow(profile: Pick<Profile, 'minPrice' | 'rowStep'>, price: number): number {
  return Math.round((price - profile.minPrice) / profile.rowStep);
}
