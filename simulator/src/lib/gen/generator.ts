/* ============================================================================
   @gen/generator — CompiledScript → bars + fabricated priors + SessionLabels.

   GDD §7 pipeline, implemented:
   · One latent intensity drives everything (MDH): λ_t = U-shaped seasonal ×
     GARCH(1,1) variance state × script boost. Returns use it, bar volume
     uses it. Profile mass = dwell × λ, so D/P/b/B shapes, HVN bulges and
     LVN necks EMERGE from the script skeleton — no special-case profile code.
   · Kernel: 1-min bars; GARCH α=0.10 β=0.85; standardized Student-t
     innovations (ν knob); seasonal open ≈2.75× midday, close ≈1.75×;
     morning-weighted jumps; Brownian-bridge noise pinned at script anchors.
   · Planted structures via CONTEXT, not paint: three prior sessions are
     fabricated from the same generator (seeds derived via siblingSeed), so
     the naked POC and prior-VA-edge references exist with ground-truth
     ancestry. The donor prior's POC is MEASURED (buildProfile) — never
     hand-placed — and today's path is scripted around that measured price.
   · Open-drive "never re-trades the open" enforced by reflection at a floor
     above the open (the fig-06 reference's own technique) + wick floor.
   · Verification loop: the game's own classifiers (core/classify) must
     recover dayType and openType; failing attempts are re-noised (named
     substreams noise#k / volume#k) up to MAX_NOISE_ATTEMPTS. Labels are true
     by construction AND by measurement.

   Determinism: everything flows from script.seed through named PRNG
   substreams ('noise' = innovations/jumps, 'volume' = wick sizes + volume
   draws, 'place' = open/prior placement, 'decoys', retry variants
   'noise#k'/'volume#k'); same seed ⇒ byte-identical {bars, priors, labels}.

   Pure TS — zero DOM/Svelte imports.
   STATUS: IMPLEMENTED (v1). Owner: gen team.
   ========================================================================== */

import type {
  AcceptanceFlag,
  Bar,
  DecoyStructure,
  ExtremeLabel,
  PlantedStructure,
  Profile,
  SessionLabels,
  SessionScript,
} from '../types';
import { Prng, siblingSeed } from './prng';
import type { RandomStream } from './prng';
import {
  DEFAULT_KNOBS,
  EASY_KNOBS,
  NORMAL_IB_POINTS,
  ROW_STEP,
  compileScript,
  interpAnchors,
} from './scripts';
import type { CompiledScript } from './scripts';
import { buildProfile, rowToPrice } from '../core/profile';
import { bracketOf, buildTpoProfile } from '../core/tpo';
import {
  classifyDayType,
  classifyOpenType,
  isAcceptedOutsideValue,
  measureExcessTicks,
} from '../core/classify';
import type { DayTypeOptions } from '../core/classify';

/**
 * The classifier options the generator's verification loop runs with — THE
 * verification contract. ddNeckFrac is tightened from the drill-side default
 * (0.2): a scripted DD neck is a near-single-print traverse (measured
 * neck/peak ≈ 0.03), while a trend day's thinnest impulse stretch measures
 * ≈ 0.13–0.18 — 0.1 splits the two populations.
 */
export const GEN_DAYTYPE_OPTS: DayTypeOptions = { ddNeckFrac: 0.1 };

/** One fabricated prior session (sim-first steal, GDD §7). */
export interface PriorSession {
  bars: Bar[];
  profile: Profile;
}

/** A fully generated session: bars + priors + the labels the grader consumes. */
export interface GeneratedSession {
  script: SessionScript;
  bars: Bar[];
  labels: SessionLabels;
  /** Fabricated prior sessions, oldest first (priors[2] = yesterday). */
  priors: PriorSession[];
}

/** U-shaped intraday seasonal volatility multiplier (open ~2.75×, close ~1.75× midday). */
export function seasonalVol(barIndex: number, nBars: number): number {
  const x = barIndex / Math.max(1, nBars - 1); // 0 at open, 1 at close
  // Smooth U: high at open, trough midday, partial recovery at close.
  const open = 2.75 * Math.exp(-x / 0.15);
  const close = 1.75 * Math.exp(-(1 - x) / 0.12);
  return 1 + open + close - Math.exp(-1 / 0.15) - Math.exp(-1 / 0.12);
}

/* ------------------------------------------------------------------ helpers */

const GARCH_ALPHA = 0.1;
const GARCH_BETA = 0.85;
/** How many re-noise attempts the verification loop may take. */
const MAX_NOISE_ATTEMPTS = 6;
/** How many donor-prior placements the nPOC plant may try. */
const MAX_PRIOR_ATTEMPTS = 6;
/** Guard band (points) the path must keep from an untouched nPOC. */
const NPOC_GUARD = 1.0;
/** Shoulder pad beyond the skeleton's extreme (points). */
const SHOULDER_PAD = 0.45;

function segAt(cs: CompiledScript, t: number) {
  for (const s of cs.segments) if (t >= s.startBar && t <= s.endBar) return s;
  return cs.segments[cs.segments.length - 1];
}

/** Reflect x back below a ceiling (mirror, then hard-clamp overshoot). */
function reflectHi(x: number, ceil: number): number {
  return x > ceil ? Math.max(ceil - (x - ceil), ceil - 1.5) : x;
}
function reflectLo(x: number, floor: number): number {
  return x < floor ? Math.min(floor + (floor - x), floor + 1.5) : x;
}

interface ExtremeGeometry {
  shoulder: number; // offset from open
  touchBars: number[];
  spikeBar: number | null;
  tail: number; // points beyond the shoulder (excessTicks × ROW_STEP)
}

/**
 * Derive extreme plant geometry from the dense skeleton: the shoulder sits
 * SHOULDER_PAD beyond the skeleton's extreme; touch bars are local extrema
 * of the skeleton near it (spread across ≥2 TPO brackets when possible).
 */
function planExtreme(
  dense: number[],
  side: 1 | -1, // +1 = session high
  mode: 'poor' | 'excess' | 'open',
  excessTicks: number,
  minBar: number,
  nBars: number,
): ExtremeGeometry {
  let ext = -Infinity;
  let peakBar = 0;
  for (let i = 0; i <= nBars; i++) {
    const v = side * dense[i];
    if (v > ext) {
      ext = v;
      peakBar = i;
    }
  }
  const shoulder = side * (ext + SHOULDER_PAD);
  if (mode === 'open') return { shoulder, touchBars: [], spikeBar: null, tail: 0 };

  const barOf = (i: number) => Math.max(minBar, Math.min(nBars - 1, i));
  if (mode === 'excess') {
    return { shoulder, touchBars: [], spikeBar: barOf(peakBar), tail: excessTicks * ROW_STEP };
  }
  // poor: pick 2–3 skeleton local extrema close to the shoulder, bracket-diverse
  const candidates: number[] = [];
  for (let i = Math.max(1, minBar); i < nBars; i++) {
    const v = side * dense[i];
    if (v >= ext - 1.8 && v >= side * dense[i - 1] && v >= side * dense[i + 1]) {
      if (candidates.length === 0 || i - candidates[candidates.length - 1] >= 12) candidates.push(i);
    }
  }
  let touchBars: number[] = [];
  for (const c of candidates) {
    if (touchBars.length === 0) touchBars.push(c);
    else if (bracketOf(c) !== bracketOf(touchBars[0]) || c - touchBars[touchBars.length - 1] >= 25) {
      touchBars.push(c);
    }
    if (touchBars.length >= 3) break;
  }
  if (touchBars.length < 2) {
    const p = barOf(peakBar);
    touchBars = [barOf(p - 35), p];
  }
  // poor-with-1-tick tail: a single 1-row poke at the last touch bar
  const spikeBar = excessTicks > 0 ? touchBars[touchBars.length - 1] : null;
  return { shoulder, touchBars, spikeBar, tail: excessTicks * ROW_STEP };
}

/* --------------------------------------------------------- path synthesis */

interface SynthCtx {
  /** Absolute open price. */
  open: number;
  /** nPOC constraints (absolute price), or null when not enforced. */
  npocPrice: number | null;
  npocSide: 1 | -1;
  /** Bar index of the scripted touch; Infinity = never touched (unresolved). */
  npocTouchBar: number;
  hi: ExtremeGeometry;
  lo: ExtremeGeometry;
}

/** Synthesize the OHLCV bars for one attempt. Consumes the given streams. */
function synthesizeBars(
  cs: CompiledScript,
  dense: number[],
  pins: number[],
  ctx: SynthCtx,
  noise: RandomStream,
  volume: RandomStream,
): Bar[] {
  const n = cs.nBars;
  const k = cs.knobs;
  const sigMid = 0.3 * (0.55 + 0.9 * k.noiseGain);

  // --- GARCH(1,1) innovation sequence over the seasonal λ (MDH kernel)
  const eps = new Array<number>(n).fill(0);
  const sig = new Array<number>(n).fill(sigMid);
  const omega = sigMid * sigMid * (1 - GARCH_ALPHA - GARCH_BETA);
  let hVar = sigMid * sigMid;
  for (let t = 0; t < n; t++) {
    const seg = segAt(cs, t);
    const s = Math.sqrt(hVar) * seasonalVol(t, n) * seg.volMult;
    sig[t] = s;
    let e = s * noise.nextStudentT(k.nu);
    // morning-weighted news jumps (difficulty knob; 0 at easy)
    if (k.jumpIntensity > 0) {
      const p = k.jumpIntensity * (t < 120 ? 3 : 1) * 0.002;
      if (noise.nextFloat() < p) e += (noise.nextFloat() < 0.5 ? -1 : 1) * s * (2 + 2 * noise.nextFloat());
    }
    eps[t] = e;
    const z = e / (seasonalVol(t, n) * seg.volMult);
    hVar = omega + GARCH_ALPHA * z * z + GARCH_BETA * hVar;
  }

  // --- Brownian-bridge the noise between script anchors (pins)
  const off = dense.slice(); // offsets from open, length n+1
  for (let p = 0; p < pins.length - 1; p++) {
    const a = pins[p];
    const b = pins[p + 1];
    let cum = 0;
    const cums = new Array<number>(b - a + 1).fill(0);
    for (let i = a + 1; i <= b; i++) {
      cum += eps[i - 1];
      cums[i - a] = cum;
    }
    for (let i = a + 1; i <= b; i++) {
      off[i] += cums[i - a] - ((i - a) / (b - a)) * cum;
    }
  }

  // --- structural enforcement (deterministic reflections)
  const hiCeil = ctx.hi.shoulder;
  const loFloor = ctx.lo.shoulder;
  const npocOff = ctx.npocPrice !== null ? ctx.npocPrice - ctx.open : null;
  const openFloor = cs.dir * 0.4; // "never re-trade the open" band (dir frame)
  for (let i = 1; i <= n; i++) {
    // open-type floor (drive / post-recross test-drive)
    if (cs.noRetradeAfterBar !== null && i > cs.noRetradeAfterBar) {
      off[i] = cs.dir === 1 ? reflectLo(off[i], openFloor) : reflectHi(off[i], openFloor);
    }
    // nPOC guard: stay a band away until the scripted touch. Path index
    // touchBar is bar touchBar−1's CLOSE, so it must still be guarded (≤);
    // the touch happens inside bar touchBar via its close/high.
    if (npocOff !== null && i <= ctx.npocTouchBar) {
      off[i] =
        ctx.npocSide === 1
          ? reflectHi(off[i], npocOff - NPOC_GUARD)
          : reflectLo(off[i], npocOff + NPOC_GUARD);
    }
    // extreme shoulders
    off[i] = reflectHi(off[i], hiCeil);
    off[i] = reflectLo(off[i], loFloor);
    // hard clamp (reflection overshoot safety)
    off[i] = Math.min(hiCeil, Math.max(loFloor, off[i]));
  }

  // --- bars with wicks, wick-level caps, and forced plant touches
  const bars: Bar[] = [];
  const touchHi = new Set(ctx.hi.touchBars);
  const touchLo = new Set(ctx.lo.touchBars);
  for (let t = 0; t < n; t++) {
    const o = ctx.open + off[t];
    const c = ctx.open + off[t + 1];
    let h = Math.max(o, c) + Math.abs(volume.nextGaussian()) * 0.3 * sig[t];
    let l = Math.min(o, c) - Math.abs(volume.nextGaussian()) * 0.3 * sig[t];

    const hiAbs = ctx.open + hiCeil;
    const loAbs = ctx.open + loFloor;
    if (t !== ctx.hi.spikeBar) h = Math.min(h, hiAbs);
    if (t !== ctx.lo.spikeBar) l = Math.max(l, loAbs);
    if (touchHi.has(t)) h = hiAbs;
    if (touchLo.has(t)) l = loAbs;
    if (t === ctx.hi.spikeBar) h = hiAbs + ctx.hi.tail;
    if (t === ctx.lo.spikeBar) l = loAbs - ctx.lo.tail;

    // never re-trade the open: wick floor above/below the open price
    if (cs.noRetradeAfterBar !== null && t > cs.noRetradeAfterBar) {
      if (cs.dir === 1) l = Math.max(l, ctx.open + 0.25);
      else h = Math.min(h, ctx.open - 0.25);
    }
    // nPOC honesty at the wick level
    if (ctx.npocPrice !== null && t < ctx.npocTouchBar) {
      if (ctx.npocSide === 1) h = Math.min(h, ctx.npocPrice - 0.5);
      else l = Math.max(l, ctx.npocPrice + 0.5);
    }
    if (ctx.npocPrice !== null && t === ctx.npocTouchBar) {
      if (ctx.npocSide === 1) h = Math.max(h, ctx.npocPrice + 0.1);
      else l = Math.min(l, ctx.npocPrice - 0.1);
    }
    h = Math.max(h, Math.max(o, c));
    l = Math.min(l, Math.min(o, c));

    // MDH volume: seasonal λ × script boost × |innovation| coupling × lognoise
    const seg = segAt(cs, t);
    const move = Math.abs(off[t + 1] - off[t]);
    const moveF = 0.6 + 0.9 * Math.min(2.5, move / (sigMid * seasonalVol(t, n)));
    const v = Math.max(
      1,
      Math.round(160 * seasonalVol(t, n) * seg.volumeBoost * moveF * Math.exp(0.32 * volume.nextGaussian())),
    );
    bars.push({ t, o, h, l, c, v });
  }
  return bars;
}

/* ------------------------------------------------------------- priors */

const PRIOR_KNOBS = { ...EASY_KNOBS, noiseGain: 0.45, decoys: 0 };

/** Fabricate one prior session from the same generator (no recursion). */
function genPrior(seed: string, basePrice: number): PriorSession {
  const prng = new Prng(seed);
  const cs = compileScript(prng, 'normal', 'open-auction', 'in-value', PRIOR_KNOBS);
  const dense = interpAnchors(cs.anchors, cs.nBars);
  const pins = cs.anchors.map((a) => a.bar);
  const hi = planExtreme(dense, 1, cs.extremeHigh.mode, cs.extremeHigh.excessTicks, 0, cs.nBars);
  const lo = planExtreme(dense, -1, cs.extremeLow.mode, cs.extremeLow.excessTicks, 0, cs.nBars);
  const ctx: SynthCtx = {
    open: basePrice,
    npocPrice: null,
    npocSide: 1,
    npocTouchBar: Infinity,
    hi,
    lo,
  };
  const bars = synthesizeBars(cs, dense, pins, ctx, prng.stream('noise'), prng.stream('volume'));
  return { bars, profile: buildProfile(bars, ROW_STEP) };
}

function priceTouched(bars: Bar[], price: number): boolean {
  for (const b of bars) if (b.l <= price && price <= b.h) return true;
  return false;
}

/** High − low of a bar series. */
export function barsRange(bars: Bar[]): number {
  let hi = -Infinity;
  let lo = Infinity;
  for (const b of bars) {
    if (b.h > hi) hi = b.h;
    if (b.l < lo) lo = b.l;
  }
  return hi - lo;
}

/**
 * The day-over-day scale the verification loop (and graders) hand to
 * classifyDayType as `typicalRange`: the MEAN of the fabricated priors'
 * ranges (a single prior day is too noisy a norm — the guide's "tiny day"
 * read is always against recent typical ranges).
 */
export function typicalPriorRange(priors: PriorSession[]): number {
  return priors.reduce((a, p) => a + barsRange(p.bars), 0) / Math.max(1, priors.length);
}

/* ------------------------------------------------------------ generate */

/**
 * Generate a full session from a compiled script: fabricated priors →
 * open placement → nPOC donor placement (measured POC) → path synthesis →
 * classifier verification loop → measured labels.
 *
 * Deterministic: same script.seed ⇒ byte-identical output. Substreams:
 * 'place' (open/prior placement), 'noise'/'noise#k' (innovations + wicks are
 * drawn from 'volume' stream alongside volume), 'decoys'.
 */
export function generateSession(script: SessionScript, basePrice = 5000): GeneratedSession {
  const cs: CompiledScript =
    'anchors' in script
      ? (script as CompiledScript)
      : compileScript(new Prng(script.seed), script.dayType, script.openType, script.openLocation, DEFAULT_KNOBS, script.paramsVersion);

  const prng = new Prng(cs.seed);
  const place = prng.stream('place');
  const n = cs.nBars;

  // --- dense skeleton (with ambiguity blend) + bridge pins
  let dense = interpAnchors(cs.anchors, n);
  if (cs.blendAnchors && cs.blendWeight > 0) {
    const partner = interpAnchors(cs.blendAnchors, n);
    const w = cs.blendWeight;
    dense = dense.map((x, i) => (1 - w) * x + w * partner[i]);
  }
  const pins = [...new Set(cs.anchors.map((a) => Math.min(a.bar, n)))].sort((a, b) => a - b);
  if (pins[0] !== 0) pins.unshift(0);
  if (pins[pins.length - 1] !== n) pins.push(n);

  // --- prior 2 (yesterday) → today's open per openLocation
  const prior2 = genPrior(siblingSeed(cs.seed, 3002), basePrice);
  const p2 = prior2.profile;
  const p2vah = rowToPrice(p2, p2.vah);
  const p2val = rowToPrice(p2, p2.val);
  const p2poc = rowToPrice(p2, p2.poc);
  let p2hi = -Infinity;
  let p2lo = Infinity;
  for (const b of prior2.bars) {
    if (b.h > p2hi) p2hi = b.h;
    if (b.l < p2lo) p2lo = b.l;
  }
  let open: number;
  switch (cs.openLocation) {
    case 'in-value':
      open = p2poc + (place.nextFloat() * 2 - 1) * 0.3 * Math.max(0.5, p2vah - p2val);
      break;
    case 'out-of-value-in-range':
      open =
        cs.dir === 1
          ? p2vah + Math.max(0.5, 0.5 * (p2hi - p2vah))
          : p2val - Math.max(0.5, 0.5 * (p2val - p2lo));
      break;
    case 'out-of-range':
    default:
      open =
        cs.dir === 1
          ? p2hi + (0.4 + 0.5 * place.nextFloat()) * NORMAL_IB_POINTS
          : p2lo - (0.4 + 0.5 * place.nextFloat()) * NORMAL_IB_POINTS;
      break;
  }

  // --- extreme plant geometry from the (blended) skeleton
  const maxDense = Math.max(...dense);
  const minDense = Math.min(...dense);

  // --- nPOC donor prior: measured POC, verified naked, path scripted around it
  const side = cs.npoc.side;
  let npocPrice: number | null = null;
  let npocTouchBar = Infinity;
  let npocResolvedPlan = false;
  let prior0: PriorSession | null = null;
  let prior1: PriorSession | null = null;
  const hiTail = cs.extremeHigh.excessTicks * ROW_STEP;
  const loTail = cs.extremeLow.excessTicks * ROW_STEP;
  for (let attempt = 0; attempt < MAX_PRIOR_ATTEMPTS; attempt++) {
    const push = cs.npoc.resolved ? 0 : attempt * 3;
    const b0 = open + side * (cs.npoc.targetOffset + push);
    const cand0 = genPrior(siblingSeed(cs.seed, 3000 + attempt * 16), b0);
    const cand1 = genPrior(siblingSeed(cs.seed, 3001 + attempt * 16), open - side * (3 + attempt));
    const P = rowToPrice(cand0.profile, cand0.profile.poc);
    const pOff = P - open;
    // nakedness: no later session (prior1, prior2) may have traded through it
    if (priceTouched(cand1.bars, P) || priceTouched(prior2.bars, P)) continue;
    if (cs.npoc.resolved) {
      // scripted touch = the skeleton's FIRST crossing of the measured POC.
      // Bars before it cannot trade the price: the path is reflected off
      // P − guard and wicks are capped until the touch bar (honesty by
      // construction — the approach itself is the only thing that gets close).
      let touch = -1;
      for (let i = 60; i <= n; i++) {
        if (side * (dense[i] - pOff) >= -0.15) {
          touch = Math.min(i, n - 1);
          break;
        }
      }
      if (touch < 70) continue;
      npocPrice = P;
      npocTouchBar = touch;
      npocResolvedPlan = true;
      prior0 = cand0;
      prior1 = cand1;
      break;
    } else {
      // unresolved: must clear the day's extreme (shoulder + tail) by a margin
      const clearance =
        side === 1
          ? pOff - (maxDense + SHOULDER_PAD + hiTail)
          : (minDense - SHOULDER_PAD - loTail) - pOff;
      if (clearance < NPOC_GUARD + 0.75) continue;
      npocPrice = P;
      npocTouchBar = Infinity;
      npocResolvedPlan = false;
      prior0 = cand0;
      prior1 = cand1;
      break;
    }
  }
  if (prior0 === null || prior1 === null) {
    // fall back: keep the priors from a fixed variant, omit the nPOC plant
    prior0 = genPrior(siblingSeed(cs.seed, 3000), open + side * (cs.npoc.targetOffset + 15));
    prior1 = genPrior(siblingSeed(cs.seed, 3001), open - side * 3);
    npocPrice = null;
  }

  // hi-extreme touches must not violate the untouched nPOC band
  const hiMinBar = npocPrice !== null && side === 1 && npocResolvedPlan ? npocTouchBar : 0;
  const loMinBar = npocPrice !== null && side === -1 && npocResolvedPlan ? npocTouchBar : 0;
  const hi = planExtreme(dense, 1, cs.extremeHigh.mode, cs.extremeHigh.excessTicks, hiMinBar, n);
  const lo = planExtreme(dense, -1, cs.extremeLow.mode, cs.extremeLow.excessTicks, loMinBar, n);
  const ctx: SynthCtx = {
    open,
    npocPrice,
    npocSide: side,
    npocTouchBar,
    hi,
    lo,
  };

  // --- synthesis + classifier verification loop (rejection resampling)
  const refRange = typicalPriorRange([prior0, prior1, prior2]);
  let bars: Bar[] = [];
  for (let k = 0; k < MAX_NOISE_ATTEMPTS; k++) {
    const noise = prng.stream(k === 0 ? 'noise' : `noise#${k}`);
    const volume = prng.stream(k === 0 ? 'volume' : `volume#${k}`);
    bars = synthesizeBars(cs, dense, pins, ctx, noise, volume);
    const tpo = buildTpoProfile(bars, cs.rowStep);
    const dayOk = classifyDayType(bars, tpo, { ...GEN_DAYTYPE_OPTS, typicalRange: refRange }) === cs.dayType;
    const openOk = classifyOpenType(bars.slice(0, 60), bars[0].o) === cs.openType;
    if (dayOk && openOk) break;
  }

  // --- measured labels
  const tpo = buildTpoProfile(bars, cs.rowStep);
  let hiPrice = -Infinity;
  let loPrice = Infinity;
  for (const b of bars) {
    if (b.h > hiPrice) hiPrice = b.h;
    if (b.l < loPrice) loPrice = b.l;
  }
  const extremes: ExtremeLabel[] = [
    {
      side: 'high',
      price: hiPrice,
      excessTicks: cs.extremeHigh.mode === 'open' ? measureExcessTicks(tpo, 'high') : cs.extremeHigh.excessTicks,
    },
    {
      side: 'low',
      price: loPrice,
      excessTicks: cs.extremeLow.mode === 'open' ? measureExcessTicks(tpo, 'low') : cs.extremeLow.excessTicks,
    },
  ];

  const planted: PlantedStructure[] = [];
  if (npocPrice !== null) {
    planted.push({
      kind: 'nPOC',
      price: npocPrice,
      sourceSessionIndex: 0,
      resolved: priceTouched(bars, npocPrice),
      ...(npocResolvedPlan ? { touchBar: npocTouchBar } : {}),
    });
  }
  if (cs.extremeHigh.mode === 'poor' && cs.extremeHigh.excessTicks <= 1) {
    planted.push({ kind: 'poorHigh', price: hiPrice, sourceSessionIndex: -1, resolved: false });
  }
  if (cs.extremeLow.mode === 'poor' && cs.extremeLow.excessTicks <= 1) {
    planted.push({ kind: 'poorLow', price: loPrice, sourceSessionIndex: -1, resolved: false });
  }
  if (cs.lvnCorridor) {
    const [a, b] = cs.lvnCorridor;
    planted.push({
      kind: 'lvnCorridor',
      price: open + (a + b) / 2,
      priceRange: [open + a, open + b],
      sourceSessionIndex: -1,
      resolved: false,
    });
  }
  const vaEdge = Math.abs(p2vah - open) <= Math.abs(p2val - open) ? p2vah : p2val;
  planted.push({
    kind: 'priorVAEdge',
    price: vaEdge,
    sourceSessionIndex: 2,
    resolved: priceTouched(bars, vaEdge),
  });

  const decoyStream = prng.stream('decoys');
  const decoys: DecoyStructure[] = [];
  const nDecoys = Math.max(0, Math.min(2, Math.round(cs.knobs.decoys)));
  for (let i = 0; i < nDecoys; i++) {
    if (i === 0) {
      // a COVERED prior POC — looks like an nPOC, isn't naked
      decoys.push({ mimics: 'nPOC', price: rowToPrice(prior1.profile, prior1.profile.poc) });
    } else {
      const off = (2 + 2.5 * decoyStream.nextFloat()) * (decoyStream.nextFloat() < 0.5 ? -1 : 1);
      decoys.push({ mimics: 'lvnCorridor', price: open + (maxDense + minDense) / 2 + off });
    }
  }

  const nBrackets = bracketOf(n - 1) + 1;
  const acceptanceFlags: AcceptanceFlag[] = [];
  for (let br = 0; br < nBrackets; br++) {
    acceptanceFlags.push({ bracket: br, accepted: isAcceptedOutsideValue(bars, p2vah, p2val, br) });
  }

  const labels: SessionLabels = {
    dayType: cs.dayType,
    blendWeight: cs.blendWeight,
    blendedToward: cs.blendWeight > 0 ? cs.blendedToward : null,
    openType: cs.openType,
    openLocation: cs.openLocation,
    script: cs.segments,
    ibWidthRatio: cs.ibWidthRatio,
    oneTimeframingBreakBar: cs.oneTimeframingBreakBar,
    extremes,
    planted,
    decoys,
    acceptanceFlags,
    // Per-question conditional probabilities are declared by the item POOL
    // (Drill E builder runs a Monte-Carlo over the pool's regime mixture) —
    // never hard-coded folklore. Empty at the single-session level.
    conditionalProbs: {},
    ambiguityWeight: cs.knobs.ambiguity,
    poolBaseRates: {},
  };

  return { script: cs, bars, labels, priors: [prior0, prior1, prior2] };
}
