/* ============================================================================
   @drills/items — item builders for the five wired drills (GDD §4 A/B/D/F/I).

   Every quantity in an item is computed by the engine libs from the seed:
   profiles/rows via @core/profile, labels via @gen/generator (verified by the
   classifier rejection loop), item ratings via @schedule knob mapping. The UI
   layer never invents a number (GDD §9 accuracy rules).

   GROUND-TRUTH SOURCES per drill:
   · poc-va-snap  — profile.poc / .vah / .val of the rendered profile (the
     same @core/profile output the canvas draws: disagreement impossible).
   · excess-or-poor — the planted excessTicks label (gen-verified: poor
     extremes measure ≤1 tick, excess bands agree — see gen/labels.test.ts);
     the 2–3-tick band is served as a "judgment call" (GDD Drill B).
   · shape-alphabet — EVERY letter is verified by the measured classifyShape
     of the rendered profile (GDD Drill D: true by construction AND by
     measurement); D/B/TREND targets additionally ride a scripted day type
     verified by classifyDayType inside generateSession's rejection loop.
     The builder rejects sessions whose measurement disagrees with the
     served letter — the same classifier grades, renders and rejects.
   · open-type-ladder — labels.openType (gen-verified at 100% recovery).
   · regime-gate — the script segment active at the snapshot bar (+ the
     nontrend day type ⇒ STAND ASIDE), per GDD Drill I.

   Determinism: everything flows from the item seed through named substreams
   ('drill' for pool/side/snapshot picks) and siblingSeed for shape-target
   retries. Same seed ⇒ byte-identical item.

   Pure TS — zero DOM/Svelte imports.
   ========================================================================== */

import type {
  Bar,
  DrillId,
  DrillItem,
  ExtremeSide,
  OpenLocation,
  OpenType,
  DayType,
  ProfileShape,
  SessionLabels,
  Stimulus,
} from '../types';
import { Prng, siblingSeed } from '../gen/prng';
import type { RandomStream } from '../gen/prng';
import { compileScript, DEFAULT_KNOBS } from '../gen/scripts';
import type { DifficultyKnobs } from '../gen/scripts';
import { generateSession, typicalPriorRange, barsRange } from '../gen/generator';
import type { GeneratedSession } from '../gen/generator';
import { buildProfile, priceToBin } from '../core/profile';
import { classifyShape } from '../core/classify';
import { itemRatingFromKnobs } from '../schedule/queue';

/** The five drills wired into the v1 loop. */
export type LoopDrillId = Extract<
  DrillId,
  'poc-va-snap' | 'excess-or-poor' | 'shape-alphabet' | 'open-type-ladder' | 'regime-gate'
>;

export const LOOP_DRILLS: LoopDrillId[] = [
  'poc-va-snap',
  'excess-or-poor',
  'shape-alphabet',
  'open-type-ladder',
  'regime-gate',
];

/** Display titles (mode line: "Rated · Shape Alphabet"). */
export const DRILL_TITLE: Record<LoopDrillId, string> = {
  'poc-va-snap': 'POC/VA Snap',
  'excess-or-poor': 'Excess or Poor',
  'shape-alphabet': 'Shape Alphabet',
  'open-type-ladder': 'Open-Type Ladder',
  'regime-gate': 'Regime Gate',
};

/** Par times in ms (informational in Rated; GDD §4 latency criteria). */
export const PAR_MS: Record<LoopDrillId, number> = {
  'poc-va-snap': 2500,
  'excess-or-poor': 2000,
  'shape-alphabet': 3000,
  'open-type-ladder': 6000,
  'regime-gate': 5000,
};

/** Chip label ↔ engine-value maps. */
export const SHAPE_CHIP: Record<ProfileShape, string> = {
  D: 'D',
  P: 'P',
  b: 'b',
  B: 'B',
  'thin-trend': 'TREND',
};
export const OPEN_CHIP: Record<OpenType, string> = {
  'open-drive': 'DRIVE',
  'open-test-drive': 'TEST-DRIVE',
  'open-rejection-reverse': 'REJECTION-REVERSE',
  'open-auction': 'AUCTION',
};
export const REGIME_CHIPS = ['FADE', 'FOLLOW', 'NO TRADE'] as const;
export type RegimeChip = (typeof REGIME_CHIPS)[number];

/** Drill B verdict classes for a planted tick count (GDD Drill B bands). */
export function excessClassOf(ticks: number): 'poor' | 'excess' | 'judgment' {
  if (ticks <= 1) return 'poor';
  if (ticks >= 4) return 'excess';
  return 'judgment';
}

/** Session clock label for a bar index (RTH open = 09:30). */
export function barClock(bar: number): string {
  const m = 9 * 60 + 30 + bar;
  return `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`;
}

/* ----------------------------------------------------------------------------
   Context the grader needs beyond DrillItem (engine-computed at build time)
   -------------------------------------------------------------------------- */

export interface DrillContext {
  /** poc-va-snap: which landmark is asked. */
  target?: 'POC' | 'VAH' | 'VAL';
  /** poc-va-snap: expansion rounds (vah − val). */
  expansionRounds?: number;
  /** excess-or-poor: highlighted side + its planted/measured ticks. */
  extremeSide?: ExtremeSide;
  extremeTicks?: number;
  extremeClass?: 'poor' | 'excess' | 'judgment';
  /** shape-alphabet: truth chip. */
  shapeTruth?: string;
  /** regime-gate: truth chip + slots for the templates. */
  regimeTruth?: RegimeChip;
  dayType?: DayType;
  /** Clock label of the bar one-timeframing started (cardinal template). */
  otfSince?: string | null;
  /** 'up' | 'down' — stair-step direction at the snapshot. */
  stairDir?: string;
  /** Session range as % of typical prior range (nontrend line). */
  rangePct?: number;
  /** open-type-ladder: open price (template slot). */
  openPrice?: number;
}

/** A built item: the DrillItem contract + grading context + full session. */
export interface LoopItem {
  item: DrillItem;
  ctx: DrillContext;
  /** The generating session (bars for replay, priors for context). */
  gen: GeneratedSession;
}

/* ------------------------------------------------------------------ pools */

interface PoolEntry {
  dayType: DayType;
  openType: OpenType;
  openLocation: OpenLocation;
}

const P = (dayType: DayType, openType: OpenType, openLocation: OpenLocation): PoolEntry => ({
  dayType,
  openType,
  openLocation,
});

/** Structure-rich profile pools for the snap drill. */
const POCVA_POOL: PoolEntry[] = [
  P('normal', 'open-auction', 'in-value'),
  P('normal-variation', 'open-test-drive', 'in-value'),
  P('double-distribution-trend', 'open-drive', 'out-of-value-in-range'),
];

/** Excess/poor pools (plant-verified extremes). */
const EXCESS_POOL: PoolEntry[] = [
  P('normal', 'open-auction', 'in-value'),
  P('normal-variation', 'open-test-drive', 'in-value'),
  P('neutral', 'open-rejection-reverse', 'in-value'),
];

/** Shape targets → generating pool. */
const SHAPE_POOL: Record<ProfileShape, PoolEntry> = {
  D: P('normal', 'open-auction', 'in-value'),
  P: P('normal-variation', 'open-test-drive', 'in-value'),
  b: P('normal-variation', 'open-test-drive', 'in-value'),
  B: P('double-distribution-trend', 'open-drive', 'out-of-value-in-range'),
  'thin-trend': P('trend', 'open-drive', 'out-of-range'),
};
const SHAPE_TARGETS: ProfileShape[] = ['D', 'P', 'b', 'B', 'thin-trend'];

/** Open-type pool: one entry per truth class (uniform mix ⇒ base rate 0.25). */
const OPEN_POOL: PoolEntry[] = [
  P('trend', 'open-drive', 'out-of-range'),
  P('normal-variation', 'open-test-drive', 'in-value'),
  P('neutral', 'open-rejection-reverse', 'in-value'),
  P('normal', 'open-auction', 'in-value'),
];

/** Regime pool: trend (FOLLOW, cardinal risk) / normal (FADE) / nontrend (NO TRADE). */
const REGIME_POOL: PoolEntry[] = [
  P('trend', 'open-drive', 'out-of-range'),
  P('normal', 'open-auction', 'in-value'),
  P('nontrend', 'open-auction', 'in-value'),
];
const REGIME_WEIGHTS = [0.4, 0.4, 0.2];

/* ------------------------------------------------------------------ helpers */

function makeStimulus(bars: Bar[], rowStep: number, revealBar: number): Stimulus {
  return { bars, profile: buildProfile(bars, rowStep), revealBar };
}

function baseItem(
  drillId: LoopDrillId,
  seed: string,
  variant: number,
  gen: GeneratedSession,
  stimulus: Stimulus,
  question: string,
  choices: string[],
  groundTruth: string | number,
  templateId: string,
  knobs: DifficultyKnobs,
  labels: SessionLabels = gen.labels,
): DrillItem {
  return {
    id: `${drillId}:${seed}:${variant}`,
    drillId,
    seed,
    paramsVersion: gen.script.paramsVersion,
    stimulus,
    question,
    choices,
    groundTruth,
    explanationTemplateId: templateId,
    labels,
    itemRating: itemRatingFromKnobs(knobs),
    parMs: PAR_MS[drillId],
  };
}

function pick<T>(ds: RandomStream, arr: readonly T[]): T {
  return arr[ds.nextInt(0, arr.length - 1)];
}

function weightedPick<T>(ds: RandomStream, arr: readonly T[], weights: readonly number[]): T {
  const total = weights.reduce((a, b) => a + b, 0);
  let x = ds.nextFloat() * total;
  for (let i = 0; i < arr.length; i++) {
    x -= weights[i];
    if (x <= 0) return arr[i];
  }
  return arr[arr.length - 1];
}

function generate(seed: string, entry: PoolEntry, knobs: DifficultyKnobs): GeneratedSession {
  const script = compileScript(
    new Prng(seed),
    entry.dayType,
    entry.openType,
    entry.openLocation,
    knobs,
  );
  return generateSession(script);
}

/* ------------------------------------------------------------------ builders */

/** How many sibling regenerations the shape verifier may take. */
const SHAPE_ATTEMPTS = 12;

function buildPocVa(seed: string, knobs: DifficultyKnobs, ds: RandomStream): LoopItem {
  const target = pick(ds, ['POC', 'VAH', 'VAL'] as const);
  const entry = pick(ds, POCVA_POOL);
  const gen = generate(seed, entry, knobs);
  const stimulus = makeStimulus(gen.bars, gen.script.rowStep, gen.bars.length - 1);
  const p = stimulus.profile;
  const truth = target === 'POC' ? p.poc : target === 'VAH' ? p.vah : p.val;
  const question = target === 'POC' ? 'Tap the POC' : `Place ${target}`;
  const item = baseItem(
    'poc-va-snap',
    seed,
    0,
    gen,
    stimulus,
    question,
    [],
    truth,
    target === 'POC' ? 'pocva.hit.exact.poc' : 'pocva.hit.exact.va',
    knobs,
  );
  return { item, ctx: { target, expansionRounds: p.vah - p.val }, gen };
}

function buildExcess(seed: string, knobs: DifficultyKnobs, ds: RandomStream): LoopItem {
  const targetClass: 'excess' | 'poor' = ds.nextFloat() < 0.5 ? 'excess' : 'poor';
  const entry = pick(ds, EXCESS_POOL);
  const gen = generate(seed, entry, knobs);
  const classified = gen.labels.extremes.map((e) => ({ e, cls: excessClassOf(e.excessTicks) }));
  const matching = classified.filter((c) => c.cls === targetClass);
  const unambiguous = classified.filter((c) => c.cls !== 'judgment');
  const chosen =
    matching.length > 0 ? pick(ds, matching) : unambiguous.length > 0 ? pick(ds, unambiguous) : classified[0];
  const stimulus = makeStimulus(gen.bars, gen.script.rowStep, gen.bars.length - 1);
  // Bin containment, not rounding: the highlight must land ON the row the
  // extreme's volume printed into (the profile's own bottom/top row), never
  // the empty neighbor priceToRow's snap can name when the extreme sits in
  // the outer half of its bin.
  const row = Math.min(
    stimulus.profile.rows.length - 1,
    Math.max(0, priceToBin(stimulus.profile, chosen.e.price)),
  );
  stimulus.highlightRow = row;
  const truth = chosen.cls === 'excess' ? 'EXCESS' : chosen.cls === 'poor' ? 'POOR' : chosen.e.excessTicks >= 3 ? 'EXCESS' : 'POOR';
  // The pool declares its own mixture — the bot's answer and the interstitial
  // reveal (GDD §5/§7): targets are drawn 50/50 by construction.
  const labels: SessionLabels = {
    ...gen.labels,
    poolBaseRates: { ...gen.labels.poolBaseRates, 'excess-or-poor': 0.5 },
  };
  const item = baseItem(
    'excess-or-poor',
    seed,
    0,
    gen,
    stimulus,
    `Excess or poor — the highlighted ${chosen.e.side}?`,
    ['EXCESS', 'POOR'],
    truth,
    chosen.cls === 'excess' ? 'excess.hit.excess' : 'excess.hit.poor',
    knobs,
    labels,
  );
  return {
    item,
    ctx: { extremeSide: chosen.e.side, extremeTicks: chosen.e.excessTicks, extremeClass: chosen.cls },
    gen,
  };
}

function buildShape(seed: string, knobs: DifficultyKnobs, ds: RandomStream): LoopItem {
  const target = pick(ds, SHAPE_TARGETS);
  const entry = SHAPE_POOL[target];
  let gen = generate(seed, entry, knobs);
  let variant = 0;
  let truthShape: ProfileShape | null = null;
  for (let a = 0; a < SHAPE_ATTEMPTS; a++) {
    const sessSeed = a === 0 ? seed : siblingSeed(seed, 7000 + a);
    if (a > 0) gen = generate(sessSeed, entry, knobs);
    // EVERY target is verified by measurement (GDD Drill D: the label is true
    // by construction AND by measurement): the same classifyShape that the
    // alphabet is defined by must agree with the served letter, or the
    // session is rejected. D/B/TREND additionally carry the scripted day
    // type verified by classifyDayType inside generateSession's rejection
    // loop; P/b have no scripted day type — the measurement IS the label.
    // trend↔B stays the designed confusion (GDD §4-D confusion matrix): a
    // trend day's ≥3-bead staircase measures thin-trend, a true DD's two
    // bulges measure B (core classifyShape ladder rule).
    const measured = classifyShape(buildProfile(gen.bars, gen.script.rowStep));
    if (measured === target) {
      truthShape = target;
      variant = a;
      break;
    }
  }
  if (truthShape === null) {
    // Rejection budget exhausted: fall back to the measured shape of the last
    // attempt — the label stays honest (measurement-backed) either way.
    truthShape = classifyShape(buildProfile(gen.bars, gen.script.rowStep));
    variant = SHAPE_ATTEMPTS - 1;
  }
  const stimulus = makeStimulus(gen.bars, gen.script.rowStep, gen.bars.length - 1);
  const labels: SessionLabels = {
    ...gen.labels,
    poolBaseRates: {
      ...gen.labels.poolBaseRates,
      // uniform target draw over the five letters, by construction
      'shape-alphabet': 1 / SHAPE_TARGETS.length,
    },
  };
  const item = baseItem(
    'shape-alphabet',
    seed,
    variant,
    gen,
    stimulus,
    'What shape is this session?',
    SHAPE_TARGETS.map((s) => SHAPE_CHIP[s]),
    SHAPE_CHIP[truthShape],
    'shape.hit',
    knobs,
    labels,
  );
  return { item, ctx: { shapeTruth: SHAPE_CHIP[truthShape] }, gen };
}

/** Minutes of tape the open-type stimulus reveals (first 90 minutes). */
export const OPEN_REVEAL_BARS = 90;

function buildOpenType(seed: string, knobs: DifficultyKnobs, ds: RandomStream): LoopItem {
  const entry = pick(ds, OPEN_POOL);
  const gen = generate(seed, entry, knobs);
  const slice = gen.bars.slice(0, OPEN_REVEAL_BARS);
  const stimulus = makeStimulus(slice, gen.script.rowStep, OPEN_REVEAL_BARS - 1);
  stimulus.replaySpeed = 8;
  const labels: SessionLabels = {
    ...gen.labels,
    poolBaseRates: { ...gen.labels.poolBaseRates, 'open-type-ladder': 1 / OPEN_POOL.length },
  };
  const item = baseItem(
    'open-type-ladder',
    seed,
    0,
    gen,
    stimulus,
    'How did this session open?',
    OPEN_POOL.map((e) => OPEN_CHIP[e.openType]),
    OPEN_CHIP[gen.labels.openType],
    `open.truth.${gen.labels.openType}`,
    knobs,
    labels,
  );
  return { item, ctx: { openPrice: gen.bars[0].o }, gen };
}

function buildRegime(seed: string, knobs: DifficultyKnobs, ds: RandomStream): LoopItem {
  const entry = weightedPick(ds, REGIME_POOL, REGIME_WEIGHTS);
  const gen = generate(seed, entry, knobs);
  let snapshotBar = ds.nextInt(210, 330);
  // keep trend snapshots INSIDE the unbroken one-timeframing run so the
  // cardinal template's "one-timeframing since {time}" claim is true
  const breakBar = gen.labels.oneTimeframingBreakBar;
  if (entry.dayType === 'trend' && breakBar !== null && snapshotBar >= breakBar) {
    snapshotBar = Math.max(210, breakBar - 5);
  }
  const slice = gen.bars.slice(0, snapshotBar + 1);
  const stimulus = makeStimulus(slice, gen.script.rowStep, snapshotBar);

  const seg = gen.labels.script.find((s) => snapshotBar >= s.startBar && snapshotBar <= s.endBar);
  const truth: RegimeChip =
    gen.labels.dayType === 'nontrend' ? 'NO TRADE' : seg?.regime === 'imbalance' ? 'FOLLOW' : 'FADE';

  // one-timeframing start: the first bar of the contiguous imbalance run
  // covering the snapshot (trend days: the IB's end)
  let otfSince: string | null = null;
  if (seg?.regime === 'imbalance') {
    let start = seg.startBar;
    for (let i = gen.labels.script.indexOf(seg) - 1; i >= 0; i--) {
      const s = gen.labels.script[i];
      if (s.regime !== 'imbalance') break;
      start = s.startBar;
    }
    otfSince = barClock(start);
  }
  const stairDir = slice[slice.length - 1].c >= slice[0].o ? 'up' : 'down';
  const rangePct = Math.round((barsRange(gen.bars) / typicalPriorRange(gen.priors)) * 100);

  const labels: SessionLabels = {
    ...gen.labels,
    poolBaseRates: {
      ...gen.labels.poolBaseRates,
      'regime-gate:follow': REGIME_WEIGHTS[0],
      'regime-gate:fade': REGIME_WEIGHTS[1],
      'regime-gate:no-trade': REGIME_WEIGHTS[2],
    },
  };
  const item = baseItem(
    'regime-gate',
    seed,
    0,
    gen,
    stimulus,
    "Balance or imbalance — what's the playbook?",
    [...REGIME_CHIPS],
    truth,
    truth === 'FADE' ? 'regime.hit.balance' : truth === 'FOLLOW' ? 'regime.hit.imbalance' : 'regime.hit.nontrend',
    knobs,
    labels,
  );
  return {
    item,
    ctx: { regimeTruth: truth, dayType: gen.labels.dayType, otfSince, stairDir, rangePct },
    gen,
  };
}

/* ------------------------------------------------------------------ dispatch */

/**
 * Build one drill item deterministically from (drillId, seed, knobs).
 * Same inputs ⇒ byte-identical item (unit-tested).
 */
export function buildLoopItem(
  drillId: LoopDrillId,
  seed: string,
  knobs: DifficultyKnobs = DEFAULT_KNOBS,
): LoopItem {
  const ds = new Prng(seed).stream('drill');
  switch (drillId) {
    case 'poc-va-snap':
      return buildPocVa(seed, knobs, ds);
    case 'excess-or-poor':
      return buildExcess(seed, knobs, ds);
    case 'shape-alphabet':
      return buildShape(seed, knobs, ds);
    case 'open-type-ladder':
      return buildOpenType(seed, knobs, ds);
    case 'regime-gate':
      return buildRegime(seed, knobs, ds);
  }
}

/** Render flags per item: Drill A must never reveal its own answer. */
export function renderFlags(item: DrillItem): { showPoc: boolean; showVa: boolean } {
  if (item.drillId === 'poc-va-snap') {
    // "Tap the POC" hides everything; VAH/VAL items show the revealed POC
    // but not the VA band (GDD Drill A input spec).
    return { showPoc: item.question !== 'Tap the POC', showVa: false };
  }
  return { showPoc: true, showVa: true };
}
