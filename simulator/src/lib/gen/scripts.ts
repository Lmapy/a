/* ============================================================================
   @gen/scripts — script compiler: (day type, open type, difficulty knobs)
   → CompiledScript (segment skeleton + anchor path + plant plans).

   GDD §7: "Day type + open type compile to a segment script (per-segment
   drift, volatility, volume-intensity, anchors)". Trend-day contract:
   IB width 0.3–0.5× normal, pullbacks ≤40% of prior impulse, close in the
   extreme 15% of range, volume rising on impulses.

   Design:
   · Everything is built in an "up frame" (imbalance direction = +1) with
     price OFFSETS from the open in points, then mirrored by `dir`.
   · The open type contributes the first ~55 bars of anchors (drive /
     test-drive / rejection-reverse / auction geometry per fig-06); the day
     type contributes the rest (fig-05 geometry).
   · Plant plans (extreme excess ticks, nPOC intent, LVN corridor) are
     decided here from the 'skeleton' substream; exact plant geometry
     (shoulder price, touch bars) is derived deterministically by the
     generator from the final (possibly ambiguity-blended) skeleton.
   · Incompatible (day type × open type) pairs are coerced to the nearest
     compatible open type (e.g. a nontrend day cannot host an open-drive
     without violating its tiny-range premise) — the coerced value is what
     lands in the script/labels, so labels stay true.

   Pure TS — zero DOM/Svelte imports. Randomness ONLY via the 'skeleton'
   substream of the passed Prng.

   STATUS: IMPLEMENTED (v1). Owner: gen team.
   ========================================================================== */

import type { DayType, OpenLocation, OpenType, Regime, Segment, SessionScript } from '../types';
import { Prng, RandomStream } from './prng';

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

/** Easy-difficulty knobs (used by label-recovery CI and early-tree items). */
export const EASY_KNOBS: DifficultyKnobs = {
  noiseGain: 0.25,
  nu: 8,
  ambiguity: 0,
  decoys: 1,
  jumpIntensity: 0,
  lvnDepth: 1,
};

/** RTH session length in 1-min bars (6.5 h). */
export const SESSION_BARS = 390;

/** "Normal" initial-balance width in points — the ibWidthRatio=1 reference. */
export const NORMAL_IB_POINTS = 12;

/** Profile row height in points (1 row = 1 "tick" for excess measurement). */
export const ROW_STEP = 0.5;

/** One skeleton anchor: the path must pass (bar, open + offset). */
export interface ScriptAnchor {
  /** Path index (bar open). The close of the last bar is index nBars. */
  bar: number;
  /** Price offset from the session open, in points (already dir-mirrored). */
  offset: number;
}

/** How a session extreme is scripted (Drill B ground truth). */
export type ExtremeMode =
  | 'poor'    // flat 2–3-touch shoulder, excessTicks 0–1 (or 2–3 in the judgment band)
  | 'excess'  // single rejection tail of excessTicks rows beyond the shoulder
  | 'open';   // extreme is the open itself (open-drive floor) — measured, not planted

/** Planted intent for one extreme; geometry is derived from the skeleton. */
export interface ExtremeIntent {
  mode: ExtremeMode;
  /** Planted tail length in ticks (rows). 0–1 poor, 4–8 excess, 2–3 judgment. */
  excessTicks: number;
}

/** Planted intent for the naked-POC structure (donor = fabricated prior 0). */
export interface NpocIntent {
  /** +1 = the nPOC sits above today's open, −1 below (already dir-mirrored). */
  side: 1 | -1;
  /** True = today's script touches it (trend/DD magnet); false = stays shy. */
  resolved: boolean;
  /** Desired |POC − open| distance for the donor prior session, in points. */
  targetOffset: number;
}

/** The compiled script plus everything the generator needs beyond the labels. */
export interface CompiledScript extends SessionScript {
  /** Imbalance direction: +1 up-frame, −1 mirrored. */
  dir: 1 | -1;
  /** Knobs the script was compiled with (echoed into difficulty labels). */
  knobs: DifficultyKnobs;
  /** Skeleton anchors, dir-mirrored, sorted by bar; first at 0, last at nBars. */
  anchors: ScriptAnchor[];
  /** Ambiguity blend: partner-day-type anchors to lerp toward (null if pure). */
  blendAnchors: ScriptAnchor[] | null;
  /** Mixture weight in [0, 0.5]; 0 = pure. */
  blendWeight: number;
  /** The day type blended toward; null when blendWeight = 0. */
  blendedToward: DayType | null;
  /** Extreme intents (post-mirror: high = session high). */
  extremeHigh: ExtremeIntent;
  extremeLow: ExtremeIntent;
  /** Naked-POC plant intent. */
  npoc: NpocIntent;
  /** LVN corridor [lo, hi] offsets from open (DD neck), or null. */
  lvnCorridor: [number, number] | null;
  /** Bar after which price must never re-trade the open (drive/test-drive). */
  noRetradeAfterBar: number | null;
}

/* ------------------------------------------------------------------------ */

interface Leg {
  from: number;
  to: number; // inclusive
  regime: Regime;
  oneTf: boolean;
  volMult: number;
  volumeBoost: number;
}

interface UpFrame {
  anchors: ScriptAnchor[];
  legs: Leg[];
  breakBar: number | null;
  hi: ExtremeIntent; // up-frame "far" extreme (session high when dir=+1)
  lo: ExtremeIntent;
  npocSide: 1 | -1;
  npocResolved: boolean;
  npocTarget: number;
  lvn: [number, number] | null;
}

/** Multiplicative jitter: x × (1 ± pct). */
function jit(sk: RandomStream, x: number, pct: number): number {
  return x * (1 + pct * (sk.nextFloat() * 2 - 1));
}

/** Piecewise-linear interpolation of anchors onto path indices 0..nBars. */
export function interpAnchors(anchors: ScriptAnchor[], nBars: number): number[] {
  const out = new Array<number>(nBars + 1).fill(0);
  const a = anchors;
  let j = 0;
  for (let i = 0; i <= nBars; i++) {
    while (j < a.length - 2 && a[j + 1].bar <= i) j++;
    const p = a[j];
    const q = a[Math.min(j + 1, a.length - 1)];
    const span = Math.max(1, q.bar - p.bar);
    const f = Math.min(1, Math.max(0, (i - p.bar) / span));
    out[i] = p.offset + f * (q.offset - p.offset);
  }
  return out;
}

/** Coerce open types that contradict the day type's range premise. */
export function coerceOpenType(dayType: DayType, openType: OpenType): OpenType {
  if (dayType === 'nontrend') return 'open-auction';
  if (dayType === 'neutral' && (openType === 'open-drive' || openType === 'open-test-drive')) {
    return 'open-rejection-reverse';
  }
  return openType;
}

/** IB width ratio band per day type (fig-05 tells; GDD §7 trend 0.3–0.5). */
function drawIbRatio(sk: RandomStream, dayType: DayType): number {
  switch (dayType) {
    case 'trend':
      return 0.3 + 0.2 * sk.nextFloat();
    case 'double-distribution-trend':
      return 0.42 + 0.16 * sk.nextFloat();
    case 'nontrend':
      return 0.24 + 0.08 * sk.nextFloat();
    case 'normal-variation':
      return 0.58 + 0.16 * sk.nextFloat();
    case 'neutral':
      return 0.6 + 0.18 * sk.nextFloat();
    case 'normal':
    default:
      return 0.9 + 0.2 * sk.nextFloat();
  }
}

/** Excess-ticks draw per extreme mode (GDD Drill B bands). */
function drawTicks(sk: RandomStream, mode: 'poor' | 'excess', ambiguity: number): number {
  if (ambiguity >= 0.3 && sk.nextFloat() < 0.35) return sk.nextInt(2, 3); // judgment band
  return mode === 'poor' ? sk.nextInt(0, 1) : sk.nextInt(4, 8);
}

/* ---------------------------------------------------------- open builders */

interface OpenPart {
  anchors: ScriptAnchor[];
  endBar: number;
  endOffset: number;
  /** Bar after which the path must stay on the +side of the open (up frame). */
  floorAfter: number | null;
  /** Up-frame extreme formed by the open structure itself, if any. */
  probeSide: 'lo' | 'hi' | null;
}

function buildOpen(sk: RandomStream, openType: OpenType, h: number): OpenPart {
  const a = (bar: number, offset: number): ScriptAnchor => ({ bar, offset });
  switch (openType) {
    case 'open-drive': {
      const top = jit(sk, 2.0 * h, 0.08);
      const anchors = [
        a(0, 0),
        a(5, jit(sk, 0.5 * h, 0.15)),
        a(15, jit(sk, 1.1 * h, 0.12)),
        a(30, jit(sk, 1.6 * h, 0.1)),
        a(48, top),
      ];
      return { anchors, endBar: 48, endOffset: top, floorAfter: 1, probeSide: null };
    }
    case 'open-test-drive': {
      const probe = -jit(sk, 0.9 * h, 0.1);
      const end = jit(sk, 1.15 * h, 0.1);
      const anchors = [
        a(0, 0),
        a(6, jit(sk, -0.55 * h, 0.15)),
        a(12, probe),
        a(20, jit(sk, 0.25 * h, 0.2)),
        a(32, jit(sk, 0.8 * h, 0.12)),
        a(50, end),
      ];
      return { anchors, endBar: 50, endOffset: end, floorAfter: 22, probeSide: 'lo' };
    }
    case 'open-rejection-reverse': {
      // probe extreme LATE (>20 bars in — the fig-06 test-drive/RR separator),
      // recross by ~bar 41 so the other side holds ≥25% of the first hour
      const probe = -jit(sk, 0.95 * h, 0.1);
      const end = jit(sk, 0.75 * h, 0.12);
      const anchors = [
        a(0, 0),
        a(6, jit(sk, -0.5 * h, 0.15)),
        a(28, probe),
        a(38, jit(sk, -0.2 * h, 0.25)),
        a(44, jit(sk, 0.4 * h, 0.15)),
        a(58, end),
      ];
      return { anchors, endBar: 58, endOffset: end, floorAfter: null, probeSide: 'lo' };
    }
    case 'open-auction':
    default: {
      const anchors: ScriptAnchor[] = [{ bar: 0, offset: 0 }];
      let sign = sk.nextFloat() < 0.5 ? 1 : -1;
      let last = 0;
      for (let bar = 9; bar <= 54; bar += 9) {
        last = sign * jit(sk, 0.8 * h, 0.15);
        anchors.push(a(bar, last));
        sign = -sign;
      }
      return { anchors, endBar: 54, endOffset: last, floorAfter: null, probeSide: null };
    }
  }
}

/* ----------------------------------------------------------- day builders */

/** Oscillation anchors: alternate around `center` ± amp, every ~period bars. */
function osc(
  sk: RandomStream,
  from: number,
  to: number,
  center: number,
  amp: number,
  period: number,
  floor: number | null,
): ScriptAnchor[] {
  const out: ScriptAnchor[] = [];
  let sign = sk.nextFloat() < 0.5 ? 1 : -1;
  for (let bar = from; bar <= to; bar += Math.max(6, Math.round(jit(sk, period, 0.2)))) {
    let off = center + sign * jit(sk, amp, 0.25);
    if (floor !== null) off = Math.max(off, floor);
    out.push({ bar, offset: off });
    sign = -sign;
  }
  return out;
}

/**
 * Build the up-frame skeleton (anchors + regime legs + plant intents) for a
 * (day type, open type) pair. All offsets in points from the open.
 */
function buildUpFrame(
  sk: RandomStream,
  dayType: DayType,
  openType: OpenType,
  h: number, // IB half-width in points
  knobs: DifficultyKnobs,
): UpFrame {
  const open = buildOpen(sk, openType, h);
  const floor = open.floorAfter !== null ? 0.15 * h : null;
  const ib = 2 * h;
  const R = NORMAL_IB_POINTS;
  const anchors = [...open.anchors];
  const start = open.endBar + 8;
  const legs: Leg[] = [{ from: 0, to: 59, regime: 'balance', oneTf: false, volMult: 1.05, volumeBoost: 1.15 }];
  let breakBar: number | null = null;
  let hiMode: 'poor' | 'excess' = sk.nextFloat() < 0.5 ? 'poor' : 'excess';
  let loMode: 'poor' | 'excess' = sk.nextFloat() < 0.5 ? 'poor' : 'excess';
  let lvn: [number, number] | null = null;
  let npocSide: 1 | -1 = 1;
  let npocResolved = false;
  let npocTarget = 0;

  // Open-structure probes are single visits → force excess on that side.
  if (open.probeSide === 'lo') loMode = 'excess';

  switch (dayType) {
    case 'normal': {
      const center = floor !== null ? Math.max(floor + 0.75 * h, open.endOffset * 0.55) : 0;
      anchors.push(...osc(sk, start, 384, center, 0.5 * h, 34, floor));
      anchors.push({ bar: SESSION_BARS, offset: center + jit(sk, 0.2 * h, 0.9) * (sk.nextFloat() < 0.5 ? 1 : -1) });
      legs.push({ from: 60, to: 389, regime: 'balance', oneTf: false, volMult: 0.92, volumeBoost: 0.95 });
      npocSide = sk.nextFloat() < 0.5 ? 1 : -1;
      npocTarget = h + 9.5; // beyond the day's range + tail + guard clearance
      break;
    }
    case 'nontrend': {
      anchors.push(...osc(sk, start, 384, 0, 0.38 * h, 30, floor));
      anchors.push({ bar: SESSION_BARS, offset: jit(sk, 0.15 * h, 0.9) * (sk.nextFloat() < 0.5 ? 1 : -1) });
      legs.push({ from: 60, to: 389, regime: 'balance', oneTf: false, volMult: 0.72, volumeBoost: 0.68 });
      // a nontrend day is TINY vs yesterday and carries no conviction —
      // both extremes print poor/flat (excess tails would inflate the range
      // and imply a rejection nobody had the initiative to cause)
      hiMode = 'poor';
      loMode = 'poor';
      npocSide = sk.nextFloat() < 0.5 ? 1 : -1;
      npocTarget = h + 9;
      break;
    }
    case 'normal-variation': {
      const ext = jit(sk, 0.65 * ib, 0.12);
      const preCenter = floor !== null ? floor + 0.45 * h : -0.1 * h;
      anchors.push(...osc(sk, start, 118, preCenter, 0.45 * h, 26, floor));
      const extEnd = 185 + sk.nextInt(-8, 8);
      anchors.push({ bar: 132, offset: preCenter + 0.3 * h });
      anchors.push({ bar: extEnd, offset: h + ext });
      anchors.push(...osc(sk, extEnd + 20, 380, h + 0.55 * ext, 0.3 * ext, 30, floor));
      anchors.push({ bar: SESSION_BARS, offset: h + jit(sk, 0.32 * ext, 0.25) });
      legs.push({ from: 60, to: 124, regime: 'balance', oneTf: false, volMult: 0.9, volumeBoost: 0.9 });
      legs.push({ from: 125, to: extEnd, regime: 'imbalance', oneTf: true, volMult: 1.15, volumeBoost: 1.35 });
      legs.push({ from: extEnd + 1, to: 389, regime: 'balance', oneTf: false, volMult: 0.85, volumeBoost: 0.9 });
      breakBar = extEnd + sk.nextInt(6, 18);
      hiMode = sk.nextFloat() < 0.6 ? 'excess' : 'poor'; // extension extreme
      npocSide = 1;
      npocTarget = h + ext + 8;
      break;
    }
    case 'trend': {
      // Staircase of impulses with ≤40% pullbacks; close in the extreme 15%.
      const l1 = jit(sk, 0.9 * R, 0.1);
      const l2 = jit(sk, 1.7 * R, 0.07);
      const l3 = jit(sk, 2.35 * R, 0.06);
      const top = jit(sk, 2.75 * R, 0.05);
      // pullback depths ≤ 40% of the prior impulse (GDD §7 trend contract)
      anchors.push({ bar: 105, offset: l1 });
      anchors.push({ bar: 125, offset: l1 - jit(sk, 0.3, 0.25) * (l1 - open.endOffset) });
      anchors.push({ bar: 195, offset: l2 });
      anchors.push({ bar: 215, offset: l2 - jit(sk, 0.3, 0.25) * (l2 - anchors[anchors.length - 2].offset) });
      anchors.push({ bar: 295, offset: l3 });
      anchors.push({ bar: 315, offset: l3 - jit(sk, 0.28, 0.25) * (l3 - anchors[anchors.length - 2].offset) });
      anchors.push({ bar: SESSION_BARS, offset: top });
      breakBar = sk.nextFloat() < 0.5 ? null : sk.nextInt(300, 358);
      const tfEnd = breakBar ?? 389;
      const trendLegs: Array<[number, number, boolean]> = [
        [60, 105, true],
        [106, 125, false],
        [126, 195, true],
        [196, 215, false],
        [216, 295, true],
        [296, 315, false],
        [316, 389, true],
      ];
      for (const [from, to, impulse] of trendLegs) {
        legs.push({
          from,
          to,
          regime: 'imbalance',
          oneTf: from <= tfEnd,
          volMult: impulse ? 1.15 : 0.9,
          volumeBoost: impulse ? 1.6 : 0.82, // volume expands with the move (GDD §7)
        });
      }
      hiMode = sk.nextFloat() < 0.55 ? 'excess' : 'poor'; // trend close extreme is often poor (unfinished)
      npocSide = 1;
      npocResolved = sk.nextFloat() < 0.7;
      npocTarget = npocResolved ? jit(sk, 2.1 * R, 0.08) : top + 7;
      break;
    }
    case 'double-distribution-trend': {
      const gap = jit(sk, 1.6 * R, 0.08);
      const traverseBars = Math.round(14 + (1 - knobs.lvnDepth) * 18);
      const t0 = 170 + sk.nextInt(-12, 12);
      const t1 = t0 + traverseBars;
      const box1C = floor !== null ? Math.max(floor + 0.55 * h, open.endOffset * 0.5) : 0.2 * h;
      const box2C = gap + 0.1 * h;
      anchors.push(...osc(sk, start, t0 - 6, box1C, 0.5 * h, 24, floor));
      anchors.push({ bar: t0, offset: box1C + 0.35 * h });
      anchors.push({ bar: t1, offset: box2C });
      anchors.push(...osc(sk, t1 + 14, 380, box2C, 0.5 * h, 26, null));
      anchors.push({ bar: SESSION_BARS, offset: box2C + jit(sk, 0.15 * h, 0.8) });
      legs.push({ from: 60, to: t0, regime: 'balance', oneTf: false, volMult: 0.9, volumeBoost: 0.95 });
      legs.push({ from: t0 + 1, to: t1, regime: 'imbalance', oneTf: true, volMult: 1.25, volumeBoost: 1.15 });
      legs.push({ from: t1 + 1, to: 389, regime: 'balance', oneTf: false, volMult: 0.9, volumeBoost: 1.0 });
      breakBar = t1 + sk.nextInt(6, 16);
      // corridor bounds exclude the boxes' rotation edges (anchor amp ≈
      // 0.5h × 1.25 jitter, plus bridge noise and wicks ≈ another ~0.8h)
      lvn = [box1C + 1.45 * h, box2C - 1.45 * h];
      npocSide = 1;
      npocResolved = sk.nextFloat() < 0.6;
      npocTarget = npocResolved ? box1C + 0.62 * h + jit(sk, 0.55, 0.4) * (box2C - box1C - 1.24 * h) : box2C + 0.65 * h + 7;
      break;
    }
    case 'neutral':
    default: {
      const extUp = jit(sk, 0.5 * ib, 0.15);
      const extDn = jit(sk, 0.55 * ib, 0.15);
      anchors.push(...osc(sk, start, 90, 0, 0.55 * h, 18, null));
      anchors.push({ bar: 138, offset: h + extUp });
      anchors.push({ bar: 158, offset: h + 0.55 * extUp });
      anchors.push({ bar: 245, offset: -(h + extDn) });
      anchors.push({ bar: 275, offset: -(h + 0.4 * extDn) });
      anchors.push(...osc(sk, 300, 375, -0.1 * h, 0.4 * h, 26, null));
      anchors.push({ bar: SESSION_BARS, offset: jit(sk, 0.18 * h, 0.9) * (sk.nextFloat() < 0.5 ? 1 : -1) });
      legs.push({ from: 60, to: 94, regime: 'balance', oneTf: false, volMult: 0.95, volumeBoost: 0.95 });
      legs.push({ from: 95, to: 158, regime: 'imbalance', oneTf: true, volMult: 1.1, volumeBoost: 1.2 });
      legs.push({ from: 159, to: 275, regime: 'imbalance', oneTf: false, volMult: 1.1, volumeBoost: 1.15 });
      legs.push({ from: 276, to: 389, regime: 'balance', oneTf: false, volMult: 0.9, volumeBoost: 0.95 });
      breakBar = 138 + sk.nextInt(4, 14);
      npocSide = sk.nextFloat() < 0.5 ? 1 : -1;
      npocTarget = h + Math.max(extUp, extDn) + 8;
      break;
    }
  }

  // Open-drive floor makes the open itself the near extreme — measured, not planted.
  // A nontrend day caps its tails so the tails cannot inflate its tiny range.
  const tickCap = dayType === 'nontrend' ? 5 : 8;
  const loIntent: ExtremeIntent =
    open.floorAfter !== null
      ? { mode: 'open', excessTicks: 0 }
      : { mode: loMode, excessTicks: Math.min(tickCap, drawTicks(sk, loMode, knobs.ambiguity)) };
  const hiIntent: ExtremeIntent = {
    mode: hiMode,
    excessTicks: Math.min(tickCap, drawTicks(sk, hiMode, knobs.ambiguity)),
  };

  anchors.sort((a, b) => a.bar - b.bar);
  return {
    anchors,
    legs,
    breakBar,
    hi: hiIntent,
    lo: loIntent,
    npocSide,
    npocResolved,
    npocTarget,
    lvn,
  };
}

/** Ambiguity confusion partners (Drill D/I designed confusions). */
export const BLEND_PARTNER: Record<DayType, DayType> = {
  normal: 'normal-variation',
  'normal-variation': 'trend',
  trend: 'double-distribution-trend',
  'double-distribution-trend': 'trend',
  nontrend: 'normal',
  neutral: 'normal',
};

/* ---------------------------------------------------------------- compile */

/**
 * Compile a session script from taxonomy + knobs.
 * Deterministic: consumes only prng.stream('skeleton'); same
 * (seed, dayType, openType, openLocation, knobs) → identical script.
 */
export function compileScript(
  prng: Prng,
  dayType: DayType,
  openType: OpenType,
  openLocation: OpenLocation,
  knobs: DifficultyKnobs = DEFAULT_KNOBS,
  paramsVersion = 'v1.1.0',
): CompiledScript {
  const sk = prng.stream('skeleton');
  const oType = coerceOpenType(dayType, openType);
  const dir: 1 | -1 = sk.nextFloat() < 0.5 ? 1 : -1;
  const ratio = drawIbRatio(sk, dayType);
  const h = (ratio * NORMAL_IB_POINTS) / 2;

  const frame = buildUpFrame(sk, dayType, oType, h, knobs);

  // Ambiguity blend: compile the confusion partner's skeleton with the same
  // stream (deterministic) and let the generator lerp the dense paths.
  const blendWeight = Math.min(0.5, Math.max(0, knobs.ambiguity));
  let blendAnchors: ScriptAnchor[] | null = null;
  let blendedToward: DayType | null = null;
  if (blendWeight > 0) {
    blendedToward = BLEND_PARTNER[dayType];
    const pRatio = drawIbRatio(sk, blendedToward);
    const pFrame = buildUpFrame(sk, blendedToward, oType, (pRatio * NORMAL_IB_POINTS) / 2, knobs);
    blendAnchors = pFrame.anchors.map((a) => ({ bar: a.bar, offset: dir * a.offset }));
  }

  // Mirror the up frame by dir.
  const anchors = frame.anchors.map((a) => ({ bar: a.bar, offset: dir * a.offset }));
  const [extremeHigh, extremeLow] = dir === 1 ? [frame.hi, frame.lo] : [frame.lo, frame.hi];
  const lvnCorridor: [number, number] | null = frame.lvn
    ? dir === 1
      ? [frame.lvn[0], frame.lvn[1]]
      : [-frame.lvn[1], -frame.lvn[0]]
    : null;
  const npoc: NpocIntent = {
    side: (dir * frame.npocSide) as 1 | -1,
    resolved: frame.npocResolved,
    targetOffset: frame.npocTarget,
  };

  // Legs → Segments; drift derived from the dense (mirrored) skeleton.
  const dense = interpAnchors(anchors, SESSION_BARS);
  const segments: Segment[] = frame.legs.map((leg) => ({
    startBar: leg.from,
    endBar: leg.to,
    regime: leg.regime,
    drift: (dense[Math.min(SESSION_BARS, leg.to + 1)] - dense[leg.from]) / (leg.to - leg.from + 1),
    volMult: leg.volMult,
    volumeBoost: leg.volumeBoost,
    oneTimeframing: leg.oneTf,
  }));

  const noRetradeAfterBar = oType === 'open-drive' ? 0 : oType === 'open-test-drive' ? 22 : null;

  return {
    seed: prng.seedString,
    paramsVersion,
    dayType,
    openType: oType,
    openLocation,
    segments,
    nBars: SESSION_BARS,
    rowStep: ROW_STEP,
    ibWidthRatio: ratio,
    oneTimeframingBreakBar: frame.breakBar,
    dir,
    knobs,
    anchors,
    blendAnchors,
    blendWeight,
    blendedToward,
    extremeHigh,
    extremeLow,
    npoc,
    lvnCorridor,
    noRetradeAfterBar,
  };
}
