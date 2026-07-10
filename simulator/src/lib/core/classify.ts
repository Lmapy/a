/* ============================================================================
   @core/classify — day-type / open-type / shape classifiers, acceptance
   heuristics, excess-vs-poor measurement, value-relationship reading.

   References: fig-02-profile-shapes.py, fig-05-day-types.py,
   fig-06-open-types.py, fig-09-excess-vs-poor.py, fig-10-value-migration.py.
   Golden-file parity is pinned by the __fixtures__/*.ts modules (GDD §10-A).

   Role (GDD §7): the generator's rejection-resampling verifier runs THESE
   classifiers on generated sessions; labels are true by construction AND by
   measurement, and the same functions grade player answers. Post-rejection
   label recovery must be ≥99% (CI invariant).

   All thresholds are exposed as options with documented defaults tuned
   against the Python reference fixtures — the generator may tighten them.

   Pure TS — zero DOM/Svelte imports.
   STATUS: COMPLETE — all classifiers measurement-based, fixture-verified.
   Owner: core team.
   ========================================================================== */

import type {
  Bar,
  DayType,
  ExtremeSide,
  OpenType,
  Profile,
  ProfileShape,
  Regime,
  TpoProfile,
} from '../types';
import { buildProfile, detectVolumeNodes, rowToPrice, smoothRows } from './profile';
import { BARS_PER_BRACKET, bracketOf, findOneTimeframingBreak } from './tpo';

/* ----------------------------------------------------------------------------
   Shape alphabet (Drill D — fig-02)
   -------------------------------------------------------------------------- */

/** Tunables for classifyShape. Defaults verified against the fig-02 fixtures. */
export interface ShapeOptions {
  /** Flatness = (totalVolume / maxRow) / nRows at or above this ⇒ thin-trend. */
  trendFlatness?: number;
  /** Looser flatness accepted as thin-trend when the POC also sits at an extreme. */
  trendFlatnessLoose?: number;
  /** POC within this fraction of either end counts as "at an extreme". */
  pocExtremeFrac?: number;
  /** Secondary bulge must be ≥ this × the dominant bulge for a B call. */
  bimodalPeakFrac?: number;
  /** Neck between the bulges must be ≤ this × the smaller bulge for a B call. */
  neckFrac?: number;
  /** POC above this fraction of the range ⇒ P; below (1 − it) ⇒ b. */
  upperThirdFrac?: number;
  /**
   * ≥ this many COMPARABLE bulges (each ≥ bimodalPeakFrac × the dominant)
   * read as a one-timeframing ladder ⇒ thin-trend, never B. fig-02 defines B
   * as exactly TWO similar bulges split by one LVN neck; a staircase of 3+
   * beads is the trend-day footprint (each impulse leg leaves a small
   * consolidation distribution behind it), not a double distribution.
   */
  ladderMinPeaks?: number;
}

const SHAPE_DEFAULTS: Required<ShapeOptions> = {
  trendFlatness: 0.5,
  trendFlatnessLoose: 0.42,
  pocExtremeFrac: 0.2,
  bimodalPeakFrac: 0.4,
  neckFrac: 0.5,
  upperThirdFrac: 2 / 3,
  ladderMinPeaks: 3,
};

/**
 * Classify the profile shape into the drill-D alphabet (fig-02 semantics):
 *  · B — exactly TWO comparable HVN bulges separated by a thin LVN neck;
 *  · thin-trend — elongated, no dominant bulge: high "flatness"
 *    (volume spread evenly over many rows), POC typically near one extreme;
 *    OR a ladder of ≥ ladderMinPeaks comparable bulges — the staircase
 *    footprint an unbroken one-timeframing day prints (fig-02's B is two
 *    bulges; three-plus beads are impulse-leg consolidations, not a DD);
 *  · P — dominant bulge in the upper third (thin tail down);
 *  · b — mirror of P (bulge in the lower third, thin tail up);
 *  · D — symmetric balance bell (everything else).
 */
export function classifyShape(profile: Profile, opts: ShapeOptions = {}): ProfileShape {
  const o = { ...SHAPE_DEFAULTS, ...opts };
  const rows = profile.rows;
  const n = rows.length;
  if (n === 0) return 'D';
  let total = 0;
  let max = 0;
  for (const v of rows) {
    total += v;
    if (v > max) max = v;
  }
  if (max <= 0) return 'D';
  const flatness = total / max / n;
  const pocPos = n === 1 ? 0.5 : profile.poc / (n - 1);

  // --- B vs ladder: how many COMPARABLE distributions? ------------------------
  const nodes = detectVolumeNodes(rows);
  if (nodes.length >= 2) {
    const byHeight = nodes.slice().sort((a, b) => b.height - a.height);
    const [p0, p1] = byHeight;
    // one-timeframing ladder: ≥3 comparable beads ⇒ thin-trend, never B
    const comparable = byHeight.filter((nd) => nd.height >= o.bimodalPeakFrac * p0.height).length;
    if (comparable >= o.ladderMinPeaks) return 'thin-trend';
    if (p1.height >= o.bimodalPeakFrac * p0.height) {
      const s = smoothRows(rows, 2);
      const a = Math.min(p0.peak, p1.peak);
      const b = Math.max(p0.peak, p1.peak);
      let neck = Infinity;
      for (let k = a + 1; k < b; k++) if (s[k] < neck) neck = s[k];
      if (neck <= o.neckFrac * p1.height) return 'B';
    }
  }

  // --- thin-trend: elongation without a dominant bulge ------------------------
  const pocExtreme = pocPos <= o.pocExtremeFrac || pocPos >= 1 - o.pocExtremeFrac;
  if (flatness >= o.trendFlatness || (flatness >= o.trendFlatnessLoose && pocExtreme)) {
    return 'thin-trend';
  }

  // --- P / b / D by POC position ----------------------------------------------
  if (pocPos >= o.upperThirdFrac) return 'P';
  if (pocPos <= 1 - o.upperThirdFrac) return 'b';
  return 'D';
}

/* ----------------------------------------------------------------------------
   Day types (fig-05) — IB width, range-extension sides, close location, shape
   -------------------------------------------------------------------------- */

/** Tunables for classifyDayType. Defaults verified against the fig-05 fixtures. */
export interface DayTypeOptions {
  /** Range extension counts when beyond the IB by more than this × IB width. */
  extensionTolFrac?: number;
  /** IB/range fraction at or below this qualifies for a trend call. */
  trendIbFracMax?: number;
  /** IB/range fraction at or below this qualifies for a double-distribution call. */
  ddIbFracMax?: number;
  /** Close inside this outer fraction of the range = "close at the extreme". */
  closeExtremeFrac?: number;
  /**
   * Neck depth (fraction of the smaller bulge) demanded for a
   * double-distribution call. Much stricter than the Drill-D B-shape default:
   * a true DD neck is a single-print LVN (guide §3.2 "LVN/single-print
   * neck"), whereas trend-day rotation wobbles leave moderate valleys.
   */
  ddNeckFrac?: number;
  /**
   * Baseline session range (in points) for the nontrend call: a session with
   * no range extension AND range < nontrendRangeFrac × typicalRange is
   * nontrend. Omitted ⇒ nontrend is never called (a lone session cannot know
   * that its range is "tiny" without a norm — the generator supplies one).
   */
  typicalRange?: number;
  /** See typicalRange. */
  nontrendRangeFrac?: number;
}

const DAY_TYPE_DEFAULTS: Required<Omit<DayTypeOptions, 'typicalRange'>> = {
  extensionTolFrac: 0.1,
  trendIbFracMax: 0.4,
  ddIbFracMax: 0.55,
  closeExtremeFrac: 0.15,
  ddNeckFrac: 0.2,
  nontrendRangeFrac: 0.55,
};

/**
 * Classify the day type from the full session (fig-05 tells), in order:
 *  · nontrend — tiny IB AND range vs typicalRange (when given): nobody leads;
 *  · neutral — range extension BOTH sides of the IB;
 *  · double-distribution-trend — one-sided extension, narrow-ish IB, and the
 *    session profile prints a B with a genuinely thin neck (ddNeckFrac);
 *  · trend — narrow IB (≤ trendIbFracMax of range), one-sided extension,
 *    close in the extreme closeExtremeFrac of the range;
 *  · normal-variation — one-sided extension, range < ~2× IB;
 *  · normal — no meaningful extension (range ≈ IB).
 */
export function classifyDayType(
  bars: Bar[],
  tpo: TpoProfile,
  opts: DayTypeOptions = {},
): DayType {
  const o = { ...DAY_TYPE_DEFAULTS, ...opts };
  if (bars.length === 0) return 'normal';
  let high = -Infinity;
  let low = Infinity;
  for (const b of bars) {
    if (b.h > high) high = b.h;
    if (b.l < low) low = b.l;
  }
  const range = high - low;
  const ib = tpo.ibHigh - tpo.ibLow;
  if (range <= 0 || ib <= 0) return 'nontrend';
  const tol = o.extensionTolFrac * ib;
  const extUp = high > tpo.ibHigh + tol;
  const extDown = low < tpo.ibLow - tol;
  const ibFrac = ib / range;
  const close = bars[bars.length - 1].c;
  const closePos = (close - low) / range;

  // nontrend first: "tiny IB and range" (guide §3.2) — a squat session is
  // nontrend regardless of small wanders past its (tiny) IB
  if (o.typicalRange !== undefined && range < o.nontrendRangeFrac * o.typicalRange) {
    return 'nontrend';
  }

  if (extUp && extDown) return 'neutral';

  if (extUp || extDown) {
    if (ibFrac <= o.ddIbFracMax) {
      const shape = classifyShape(buildProfile(bars, tpo.rowStep), { neckFrac: o.ddNeckFrac });
      if (shape === 'B') return 'double-distribution-trend';
      const closeAtExtreme = extUp
        ? closePos >= 1 - o.closeExtremeFrac
        : closePos <= o.closeExtremeFrac;
      if (ibFrac <= o.trendIbFracMax && closeAtExtreme) return 'trend';
    }
    return 'normal-variation';
  }

  return 'normal';
}

/** IB width as a fraction of the session range — the fig-05 early tell. */
export function ibWidthFraction(bars: Bar[], ibBars = 2 * BARS_PER_BRACKET): number {
  if (bars.length === 0) throw new Error('ibWidthFraction: no bars');
  let high = -Infinity;
  let low = Infinity;
  let ibHigh = -Infinity;
  let ibLow = Infinity;
  for (const b of bars) {
    if (b.h > high) high = b.h;
    if (b.l < low) low = b.l;
    if (b.t < ibBars) {
      if (b.h > ibHigh) ibHigh = b.h;
      if (b.l < ibLow) ibLow = b.l;
    }
  }
  const range = high - low;
  return range > 0 ? (ibHigh - ibLow) / range : 1;
}

/* ----------------------------------------------------------------------------
   Open types (fig-06) — Dalton's conviction ladder
   -------------------------------------------------------------------------- */

/** Tunables for classifyOpenType. Defaults verified against the fig-06 fixtures. */
export interface OpenTypeOptions {
  /** This many above↔below transitions across the open ⇒ open-auction. */
  auctionMinCrossings?: number;
  /** The final one-sided hold must last at least this fraction of the window. */
  holdFrac?: number;
  /** The initial opposite-side probe must reach at least this fraction of the window range. */
  probeFrac?: number;
  /** Opposite extreme at or before this bar ⇒ test-drive; later ⇒ rejection-reverse. */
  earlyProbeBars?: number;
}

const OPEN_TYPE_DEFAULTS: Required<OpenTypeOptions> = {
  auctionMinCrossings: 3,
  holdFrac: 0.25,
  probeFrac: 0.1,
  earlyProbeBars: 20,
};

/**
 * Classify the open type from the opening bars (fig-06 semantics; callers pass
 * the window they want read — the drill uses the first 60–90 minutes):
 *  · open-drive — after the opening bar, price NEVER re-trades the open;
 *  · open-auction — price rotates across the open (≥ auctionMinCrossings
 *    side changes), or no directional resolution at all;
 *  · open-test-drive — an EARLY probe to one side (extreme by
 *    earlyProbeBars), then a reversal through the open that holds;
 *  · open-rejection-reverse — a LATER initial drive that fails and reverses
 *    through the open, holding the other side.
 */
export function classifyOpenType(
  bars: Bar[],
  openPrice: number,
  opts: OpenTypeOptions = {},
): OpenType {
  const o = { ...OPEN_TYPE_DEFAULTS, ...opts };
  const n = bars.length;
  if (n < 2) return 'open-auction';

  type Side = 'above' | 'below' | 'at';
  const sideOf = (b: Bar): Side =>
    b.l > openPrice ? 'above' : b.h < openPrice ? 'below' : 'at';
  const sides = bars.map(sideOf);

  // --- open-drive: never re-trades the open after the opening bar ------------
  const rest = sides.slice(1);
  if (rest.every((s) => s === 'above') || rest.every((s) => s === 'below')) {
    return 'open-drive';
  }

  // --- crossings: rotation across the open ⇒ auction --------------------------
  let transitions = 0;
  let prevStrict: Side | null = null;
  for (const s of sides) {
    if (s === 'at') continue;
    if (prevStrict !== null && s !== prevStrict) transitions++;
    prevStrict = s;
  }
  if (transitions >= o.auctionMinCrossings) return 'open-auction';

  // --- one reversal that holds: test-drive vs rejection-reverse ---------------
  const finalSide = prevStrict;
  if (finalSide === null) return 'open-auction';
  let lastTouch = -1; // last bar not strictly on the final side
  for (let i = 0; i < n; i++) if (sides[i] !== finalSide) lastTouch = i;
  const holdLen = n - 1 - lastTouch;
  if (holdLen < o.holdFrac * n) return 'open-auction';

  let high = -Infinity;
  let low = Infinity;
  for (const b of bars) {
    if (b.h > high) high = b.h;
    if (b.l < low) low = b.l;
  }
  const range = high - low;
  // opposite-side excursion before the final cross
  let extremeIdx = -1;
  let extremeDepth = 0;
  for (let i = 0; i <= lastTouch; i++) {
    const depth = finalSide === 'above' ? openPrice - bars[i].l : bars[i].h - openPrice;
    if (depth > extremeDepth) {
      extremeDepth = depth;
      extremeIdx = i;
    }
  }
  if (extremeIdx < 0 || extremeDepth < o.probeFrac * range) return 'open-auction';

  return extremeIdx <= o.earlyProbeBars ? 'open-test-drive' : 'open-rejection-reverse';
}

/* ----------------------------------------------------------------------------
   Excess vs poor extremes (Drill B — fig-09)
   -------------------------------------------------------------------------- */

/**
 * Measure the excess of a session extreme in ticks (rows): the length of the
 * consecutive SINGLE-PRINT run (TPO count = 1) at that extreme — the
 * rejection tail. A poor (flat, 2+-touch) extreme measures 0.
 * 0–1 = poor extreme, 4–8 = genuine excess, 2–3 = judgment band (GDD Drill B).
 */
export function measureExcessTicks(tpo: TpoProfile, side: ExtremeSide): number {
  const counts = tpo.rows.map((r) => r.length);
  const n = counts.length;
  let ticks = 0;
  if (side === 'high') {
    for (let r = n - 1; r >= 0 && counts[r] === 1; r--) ticks++;
  } else {
    for (let r = 0; r < n && counts[r] === 1; r++) ticks++;
  }
  return ticks;
}

/**
 * Volume-profile analog of the excess tail (fig-09 semantics): the number of
 * consecutive rows at the extreme whose volume is ≤ tailFrac × the profile's
 * max row — the thin taper. A poor extreme's flat, substantial top rows
 * measure 0.
 */
export function measureExcessRows(
  rows: number[],
  side: ExtremeSide,
  tailFrac = 0.25,
): number {
  const n = rows.length;
  let max = 0;
  for (const v of rows) if (v > max) max = v;
  if (max <= 0) return 0;
  const cut = tailFrac * max;
  let count = 0;
  if (side === 'high') {
    for (let r = n - 1; r >= 0 && rows[r] <= cut; r--) count++;
  } else {
    for (let r = 0; r < n && rows[r] <= cut; r++) count++;
  }
  return count;
}

/** Drill-B verdict classes: the 2–3 tick band is a judgment call (GDD §4-B). */
export type ExcessCall = 'excess' | 'poor' | 'judgment';

/**
 * Classify a volume-profile extreme per the GDD Drill B bands measured with
 * measureExcessRows: ≤1 row of taper = poor, ≥4 = excess, 2–3 = judgment.
 */
export function classifyExtreme(
  rows: number[],
  side: ExtremeSide,
  tailFrac = 0.25,
): ExcessCall {
  const t = measureExcessRows(rows, side, tailFrac);
  if (t <= 1) return 'poor';
  if (t >= 4) return 'excess';
  return 'judgment';
}

/* ----------------------------------------------------------------------------
   Acceptance outside value (Drill H — fig-10 / guide §3.4, §4.1)
   -------------------------------------------------------------------------- */

/** Tunables for isAcceptedOutsideValue. */
export interface AcceptanceOptions {
  /** Fraction of a bracket's closes that must sit outside prior value. */
  bracketOutsideFrac?: number;
  /** Row step used for the developing-POC migration check (points). */
  rowStep?: number;
}

/**
 * Acceptance-vs-rejection at a bracket boundary (guide: "two consecutive
 * 30-min periods building value outside + dPOC migration = acceptance").
 * Measured as: brackets atBracket−1 and atBracket BOTH spend ≥
 * bracketOutsideFrac of their bar closes outside prior value on the SAME
 * side, AND the developing POC through atBracket has migrated in that
 * direction versus two brackets earlier.
 *
 * @param bars        session bars so far (bars[i].t = minute index)
 * @param priorVah    prior session value area high (price)
 * @param priorVal    prior session value area low (price)
 * @param atBracket   bracket index to evaluate at (needs ≥ 2 brackets of data)
 */
export function isAcceptedOutsideValue(
  bars: Bar[],
  priorVah: number,
  priorVal: number,
  atBracket: number,
  opts: AcceptanceOptions = {},
): boolean {
  const outsideFrac = opts.bracketOutsideFrac ?? 0.5;
  if (atBracket < 1) return false;
  const upto = (br: number) => bars.filter((b) => bracketOf(b.t) <= br);
  const inBracket = (br: number) => bars.filter((b) => bracketOf(b.t) === br);

  const frac = (br: number, side: 'above' | 'below'): number => {
    const bb = inBracket(br);
    if (bb.length === 0) return 0;
    const out = bb.filter((b) => (side === 'above' ? b.c > priorVah : b.c < priorVal));
    return out.length / bb.length;
  };

  let side: 'above' | 'below' | null = null;
  for (const s of ['above', 'below'] as const) {
    if (frac(atBracket - 1, s) >= outsideFrac && frac(atBracket, s) >= outsideFrac) side = s;
  }
  if (side === null) return false;

  // developing POC migration in the acceptance direction
  const nowBars = upto(atBracket);
  const refBars = upto(Math.max(0, atBracket - 2));
  if (nowBars.length === 0 || refBars.length === 0) return false;
  let high = -Infinity;
  let low = Infinity;
  for (const b of nowBars) {
    if (b.h > high) high = b.h;
    if (b.l < low) low = b.l;
  }
  const rowStep = opts.rowStep ?? Math.max((high - low) / 50, 1e-9);
  const pNow = buildProfile(nowBars, rowStep);
  const pocNow = rowToPrice(pNow, pNow.poc);
  const pRef = buildProfile(refBars, rowStep);
  const pocRef = rowToPrice(pRef, pRef.poc);
  return side === 'above' ? pocNow > pocRef : pocNow < pocRef;
}

/* ----------------------------------------------------------------------------
   Regime (Drill I — guide §4.0)
   -------------------------------------------------------------------------- */

/** Tunables for classifyRegime. */
export interface RegimeOptions {
  /** dPOC migration beyond this fraction of the developing range ⇒ imbalance. */
  migrationFrac?: number;
  /** How many recent 30-min brackets the one-timeframing check inspects. */
  otfWindowBrackets?: number;
}

/**
 * Regime call at a snapshot bar (guide §4.0: "Overlapping value + D-shape =
 * balance → responsive trades. Value migrating + elongation + one-timeframing
 * = imbalance → initiative trades only."). Measured as: IMBALANCE when the
 * last otfWindowBrackets brackets are one-timeframing in their net direction
 * (unbroken — "one-timeframing since 10:00"), OR the developing POC has
 * migrated ≥ migrationFrac of the developing range between the first and
 * second half of the window; BALANCE otherwise.
 */
export function classifyRegime(
  bars: Bar[],
  atBar: number,
  opts: RegimeOptions = {},
): Regime {
  const migrationFrac = opts.migrationFrac ?? 0.25;
  const otfWindowBrackets = opts.otfWindowBrackets ?? 4;
  const slice = bars.filter((b) => b.t <= atBar);
  if (slice.length < 2) return 'balance';

  let high = -Infinity;
  let low = Infinity;
  for (const b of slice) {
    if (b.h > high) high = b.h;
    if (b.l < low) low = b.l;
  }
  const range = high - low;
  if (range <= 0) return 'balance';

  // --- recent one-timeframing in the window's net direction, unbroken ---------
  const curBracket = bracketOf(slice[slice.length - 1].t);
  const winStart = Math.max(0, curBracket - (otfWindowBrackets - 1));
  if (curBracket - winStart >= 2) {
    const win = slice.filter((b) => bracketOf(b.t) >= winStart);
    const net = win[win.length - 1].c - win[0].o;
    const dir: 'up' | 'down' = net >= 0 ? 'up' : 'down';
    if (findOneTimeframingBreak(win, dir) === null) return 'imbalance';
  }

  // --- developing POC migration, first half vs second half --------------------
  const mid = Math.floor(slice.length / 2);
  if (mid >= 1) {
    const rowStep = Math.max(range / 50, 1e-9);
    const pa = buildProfile(slice.slice(0, mid), rowStep);
    const pb = buildProfile(slice.slice(mid), rowStep);
    const shift = Math.abs(rowToPrice(pb, pb.poc) - rowToPrice(pa, pa.poc));
    if (shift >= migrationFrac * range) return 'imbalance';
  }
  return 'balance';
}

/* ----------------------------------------------------------------------------
   Value relationships (fig-10 / guide §3.4)
   -------------------------------------------------------------------------- */

/** Day-over-day value-area relationship (guide §3.4 taxonomy). */
export type ValueRelationship = 'higher' | 'lower' | 'overlapping' | 'inside' | 'outside';

/** A value area as prices (vah ≥ val). */
export interface ValueBand {
  vah: number;
  val: number;
}

/**
 * Classify today's value area against the prior session's (guide §3.4):
 *  · inside — today's value entirely within yesterday's (coiling);
 *  · outside — today's value engulfing yesterday's (expanding volatility);
 *  · overlapping — overlap ≥ overlapFrac of the narrower band (balance);
 *  · higher / lower — value migrated (little overlap): trend behavior.
 */
export function classifyValueRelationship(
  today: ValueBand,
  prior: ValueBand,
  overlapFrac = 0.8,
): ValueRelationship {
  if (today.val === prior.val && today.vah === prior.vah) return 'overlapping';
  if (today.val >= prior.val && today.vah <= prior.vah) return 'inside';
  if (today.val <= prior.val && today.vah >= prior.vah) return 'outside';
  const overlap = Math.max(0, Math.min(today.vah, prior.vah) - Math.max(today.val, prior.val));
  const narrower = Math.min(today.vah - today.val, prior.vah - prior.val);
  if (narrower > 0 && overlap / narrower >= overlapFrac) return 'overlapping';
  return today.val > prior.val ? 'higher' : 'lower';
}
