/* ============================================================================
   THE AUCTION — domain contract (src/lib/types.ts)

   Single source of truth for the shapes passed between the engine libs
   (core / gen / schedule), the renderer, and the UI. Derived from:
     · GDD §7 engine spec + ItemLabels schema
     · GDD §4 drill catalog (Drills A–J)
     · GDD §5 scoring math (Glicko-2, Brier, confidence taps)
     · domain guide /docs/volume-profiles-and-auction-market-theory.md

   RULES FOR TEAMMATES:
   · EXTEND, don't break. Add optional fields or new types; never repurpose
     existing fields or change their meaning.
   · Engine libs (core/gen/schedule) import ONLY from this file and each
     other — zero Svelte/DOM imports.
   · Every displayed quantity is computed from these structures by core;
     the UI layer never invents numbers.
   ========================================================================== */

/* ----------------------------------------------------------------------------
   Market data primitives
   -------------------------------------------------------------------------- */

/** One 1-minute OHLCV bar. All prices in instrument points (not ticks). */
export interface Bar {
  /** Bar index within the session, 0-based (minute 0 = session open). */
  t: number;
  /** Open price. */
  o: number;
  /** High price (h >= max(o, c)). */
  h: number;
  /** Low price (l <= min(o, c)). */
  l: number;
  /** Close price. */
  c: number;
  /** Volume traded during the bar (arbitrary units; relative mass is what matters). */
  v: number;
}

/** Inclusive range of row indices into Profile.rows (e.g. an HVN bulge). */
export interface RowRange {
  /** First row index of the range (lowest price row). */
  lo: number;
  /** Last row index of the range, inclusive. */
  hi: number;
}

/* ----------------------------------------------------------------------------
   Taxonomies (GDD §7 "6-type taxonomy", §4 Drill D/F/I, domain guide Part III)
   -------------------------------------------------------------------------- */

/** Steidlmayer/Dalton six day types (guide §3.2). */
export type DayType =
  | 'normal'                    // wide IB, range ≈ IB, symmetric D
  | 'normal-variation'          // IB extended one side ~ up to double
  | 'trend'                     // narrow IB, one-timeframing, close at extreme
  | 'double-distribution-trend' // two bulges joined by an LVN neck
  | 'nontrend'                  // tiny IB and range, squat profile, nobody leads
  | 'neutral';                  // range extension BOTH sides, close mid-range

/** Dalton open types (guide §3.4, fig-06). Order = conviction descending. */
export type OpenType =
  | 'open-drive'            // never re-trades the open; highest conviction
  | 'open-test-drive'       // probes one side, rejects, then drives
  | 'open-rejection-reverse'// initial move fails and reverses through the open
  | 'open-auction';         // rotational, low conviction

/** Where the session opened relative to prior value (guide §3.4). */
export type OpenLocation = 'in-value' | 'out-of-value-in-range' | 'out-of-range';

/** Profile shape alphabet (Drill D; guide §3.1, fig-02). */
export type ProfileShape = 'D' | 'P' | 'b' | 'B' | 'thin-trend';

/** Auction regime for a script segment / regime-gate answer (Drill I). */
export type Regime = 'balance' | 'imbalance';

/** Which extreme of the session a structure refers to. */
export type ExtremeSide = 'high' | 'low';

/* ----------------------------------------------------------------------------
   Session script — what gen compiles BEFORE synthesizing bars (GDD §7)
   -------------------------------------------------------------------------- */

/** One regime segment of the scripted skeleton (per-segment drift/vol/volume). */
export interface Segment {
  /** First bar index (inclusive) governed by this segment. */
  startBar: number;
  /** Last bar index (inclusive). Segments tile the session with no gaps. */
  endBar: number;
  /** Auction regime active during this segment — ground truth for Drill I. */
  regime: Regime;
  /** Per-bar price drift in points (positive = up). */
  drift: number;
  /** Multiplier on the seasonal×GARCH volatility state for this segment. */
  volMult: number;
  /** Multiplier on volume intensity λ (script boost term of the MDH model). */
  volumeBoost: number;
  /** True while this segment one-timeframes (consecutive brackets, one side in control). */
  oneTimeframing: boolean;
}

/** A structure planted by the generator with ground-truth ancestry (GDD §7). */
export interface PlantedStructure {
  /** Discriminator for the planted structure kind. */
  kind: 'nPOC' | 'poorHigh' | 'poorLow' | 'lvnCorridor' | 'priorVAEdge';
  /** Price level of the structure. For lvnCorridor: the corridor's center. */
  price: number;
  /** For lvnCorridor only: full price extent [low, high] of the low-volume neck. */
  priceRange?: [number, number];
  /** Which fabricated prior session (0 = oldest) planted it; -1 = intraday (today). */
  sourceSessionIndex: number;
  /** Resolution state at session end — consumed by boss grading (GDD §7 resolutionState). */
  resolved: boolean;
  /**
   * For resolved structures with a scripted first touch (e.g. an nPOC the
   * script tags mid-session): the bar index of that touch. The price is
   * guaranteed untouched by every earlier bar (planted-structure honesty).
   */
  touchBar?: number;
}

/** A decoy near-structure (0–2 per session, anti-obviousness rule, GDD §7). */
export interface DecoyStructure {
  /** What real structure the decoy imitates. */
  mimics: PlantedStructure['kind'];
  /** Price level of the decoy. */
  price: number;
}

/** One extreme of the session with its planted excess label (Drill B). */
export interface ExtremeLabel {
  /** Which end of the range. */
  side: ExtremeSide;
  /** Price of the extreme. */
  price: number;
  /**
   * Planted excess magnitude in ticks: 0–1 = poor (flat 2–3-touch) extreme,
   * 4–8 = genuine excess tail. 2–3 = ambiguous band, graded probabilistically
   * and marked "judgment call" (GDD Drill B).
   */
  excessTicks: number;
}

/**
 * The compiled script: everything gen decided BEFORE noise-filling.
 * (seed, paramsVersion) makes any session byte-identically re-derivable.
 */
export interface SessionScript {
  /** 64-bit master seed as decimal string (avoid JS number precision loss). */
  seed: string;
  /** Generator parameter-pack version; persisted with every scored decision. */
  paramsVersion: string;
  /** Scripted day type (may be blended toward another by ambiguity). */
  dayType: DayType;
  /** Scripted open type. */
  openType: OpenType;
  /** Open location vs prior value/range. */
  openLocation: OpenLocation;
  /** Regime timeline; segments tile [0, nBars). */
  segments: Segment[];
  /** Session length in 1-min bars (RTH ≈ 390). */
  nBars: number;
  /** Price step of one profile row, in points. */
  rowStep: number;
  /** Initial-balance width as a ratio of "normal" IB width (trend: 0.3–0.5). */
  ibWidthRatio: number;
  /** Exact bar index where one-timeframing breaks; null if it never breaks. */
  oneTimeframingBreakBar: number | null;
}

/* ----------------------------------------------------------------------------
   Session labels — GDD §7 ItemLabels, shipped with every item, consumed by the
   local grader. True by construction AND verified by measurement (classifiers).
   -------------------------------------------------------------------------- */

/** Per-30-min-bracket acceptance flag (Drill H ground truth). */
export interface AcceptanceFlag {
  /** 30-minute bracket index (0 = first bracket / period A). */
  bracket: number;
  /** True if this bracket built value outside prior value (acceptance evidence). */
  accepted: boolean;
}

/** Complete label set for one generated session (GDD §7 ItemLabels). */
export interface SessionLabels {
  /** Scripted day type — ground truth for Drills D and I. */
  dayType: DayType;
  /**
   * Ambiguity mixture weight in [0, 0.5]: 0 = pure dayType, >0.3 = blended
   * item graded probabilistically against the blend (GDD Drill I).
   */
  blendWeight: number;
  /** The day type blended in when blendWeight > 0; null for pure items. */
  blendedToward: DayType | null;
  /** Scripted open type — ground truth for Drill F. */
  openType: OpenType;
  /** Open location — quizzed separately on alternating Drill F items. */
  openLocation: OpenLocation;
  /** Per-segment regime timeline with bar ranges (Drill I snapshot lookup). */
  script: Segment[];
  /** IB width ratio vs normal — trend tell (0.3–0.5 on scripted trend days). */
  ibWidthRatio: number;
  /** Exact bar index where one-timeframing breaks (Drill G); null = never. */
  oneTimeframingBreakBar: number | null;
  /** Both session extremes with planted excessTicks (Drill B). */
  extremes: ExtremeLabel[];
  /** Planted structures with ground-truth ancestry: nPOCs, poor extremes, LVN necks, prior VA edges. */
  planted: PlantedStructure[];
  /** 0–2 decoy near-structures (anti-obviousness). */
  decoys: DecoyStructure[];
  /** Per-bracket acceptance flags (Drill H). */
  acceptanceFlags: AcceptanceFlag[];
  /**
   * True conditional probability per calibration question id, under the item
   * pool's declared regime mixture (Drill E ground truth — graded vs the
   * distribution, never the realized path).
   */
  conditionalProbs: Record<string, number>;
  /** Overall ambiguity weight of the item (difficulty knob echo). */
  ambiguityWeight: number;
  /**
   * The pool's declared base rate per question id — the base-rate bot's
   * answers AND the interstitial reveals (GDD §5). Never hard-coded folklore.
   */
  poolBaseRates: Record<string, number>;
}

/* ----------------------------------------------------------------------------
   Computed structures — outputs of @core, used identically for generation-
   verification, rendering, and grading (single-library rule, GDD §9)
   -------------------------------------------------------------------------- */

/** Volume-at-price histogram + derived landmarks (fig-01 algorithms). */
export interface Profile {
  /** Price of row 0 (the lowest row's center). */
  minPrice: number;
  /** Price step between consecutive rows, in points. */
  rowStep: number;
  /** Volume per row; rows[i] is at price minPrice + i * rowStep. */
  rows: number[];
  /** Row index of the Point of Control (argmax volume; ties → closer to mid). */
  poc: number;
  /** Row index of the Value Area High (top of the 70% two-row expansion). */
  vah: number;
  /** Row index of the Value Area Low (bottom of the 70% expansion). */
  val: number;
  /** Total volume across all rows. */
  totalVolume: number;
  /** High-volume-node bulges (row ranges), dominant first. */
  hvnRanges: RowRange[];
  /** Low-volume-node valleys / necks (row ranges), deepest first. */
  lvnRanges: RowRange[];
}

/** TPO (time-price-opportunity) profile — 30-min brackets lettered A, B, C… */
export interface TpoProfile {
  /** Price of row 0. */
  minPrice: number;
  /** Price step between rows. */
  rowStep: number;
  /** Per row: the bracket letters that touched it, in bracket order (e.g. "ABD"). */
  rows: string[];
  /** Row index of the TPO POC (longest letter count; ties → closer to mid). */
  poc: number;
  /** TPO value area high row (70% of TPO counts). */
  vah: number;
  /** TPO value area low row. */
  val: number;
  /** Initial balance high price (extreme of brackets A+B). */
  ibHigh: number;
  /** Initial balance low price. */
  ibLow: number;
  /** Rows touched by exactly one bracket — single prints. */
  singlePrintRows: number[];
}

/** Session VWAP line with standard-deviation bands (fig-12 math). */
export interface VwapSeries {
  /** vwap[i] = cumulative Σ(p·v)/Σ(v) through bar i. */
  vwap: number[];
  /** vwap + 1 volume-weighted standard deviation, per bar. */
  upper1: number[];
  /** vwap − 1 sd. */
  lower1: number[];
  /** vwap + 2 sd. */
  upper2: number[];
  /** vwap − 2 sd. */
  lower2: number[];
}

/* ----------------------------------------------------------------------------
   Drills (GDD §4 catalog, A–J)
   -------------------------------------------------------------------------- */

/** The ten v1 drills. IDs are stable keys for ratings, queues, and registries. */
export type DrillId =
  | 'poc-va-snap'        // A — tap POC / place VAH / place VAL
  | 'excess-or-poor'     // B — EXCESS vs POOR on a highlighted extreme
  | 'hvn-lvn-marker'     // C — tap the LVN neck / dominant HVN
  | 'shape-alphabet'     // D — classify D / P / b / B / thin-trend
  | 'calibration-range'  // E — probability slider, Brier-scored
  | 'open-type-ladder'   // F — DRIVE / TEST-DRIVE / REJECTION-REVERSE / AUCTION
  | 'one-timeframing-buzzer' // G — release when control breaks
  | 'acceptance-clock'   // H — ACCEPTED vs REJECTED, earlier scores more
  | 'regime-gate'        // I — BALANCE/IMBALANCE then playbook stage
  | 'setup-picker';      // J — one of 9 setups or NO TRADE

/**
 * What the player sees: a slice of a generated session plus derived artifacts.
 * All fields computed by core/gen from the same seed — nothing hand-placed.
 */
export interface Stimulus {
  /** The bars revealed to the player (may be a mid-session prefix). */
  bars: Bar[];
  /** Volume profile of the revealed slice (built by core/profile). */
  profile: Profile;
  /** TPO profile of the slice, when the drill needs brackets (F/G/H/I). */
  tpo?: TpoProfile;
  /** VWAP series, when the drill displays it. */
  vwap?: VwapSeries;
  /** Bar index at which the stimulus is frozen (bars.length - 1 of the full session for completed profiles). */
  revealBar: number;
  /** Row index highlighted for the question (e.g. the extreme in Drill B). */
  highlightRow?: number;
  /** Replay speed multiplier for tape drills (F/G/H), e.g. 8. */
  replaySpeed?: number;
}

/** A single gradable item served to the player. */
export interface DrillItem {
  /** Unique item id: `${drillId}:${seed}:${variant}`. */
  id: string;
  /** Which drill this item belongs to. */
  drillId: DrillId;
  /** Master seed of the generating session (decimal string). */
  seed: string;
  /** Generator parameter-pack version (re-derivability contract). */
  paramsVersion: string;
  /** What the player sees. */
  stimulus: Stimulus;
  /** The question line, e.g. "Tap the POC" / "Excess or poor?". */
  question: string;
  /**
   * Answer choices for chip drills (2–4 entries); empty for tap/slider drills
   * where the answer space is the chart or a probability.
   */
  choices: string[];
  /**
   * Ground truth for grading, drill-dependent:
   *  · chip drills: the correct choice string
   *  · snap drills (A/C): the correct row index (number)
   *  · slider (E): the true conditional probability (number in [0,1])
   *  · buzzer (G): the break bar index (number)
   */
  groundTruth: string | number;
  /** Key into the canonical feedback-template registry (GDD §10-B audit). */
  explanationTemplateId: string;
  /** Full label set of the generating session — the local grader's input. */
  labels: SessionLabels;
  /** Knob-derived item rating (GDD §5), Glicko scale (~800–2400). */
  itemRating: number;
  /** Par time in ms (informational in Rated, deadline in Rush). */
  parMs: number;
}

/* ----------------------------------------------------------------------------
   Answers & verdicts (GDD §2 core loop, §5 scoring)
   -------------------------------------------------------------------------- */

/** Confidence ride-along on binary drills → implicit 90/75/55% (GDD §5). */
export type Confidence = 'sure' | 'lean' | 'guess';

/** The player's committed answer for one item. */
export interface Answer {
  /** Item id this answers. */
  itemId: string;
  /**
   * The player's choice: chip string, tapped row index, probability in [0,1],
   * or release bar index — mirrors DrillItem.groundTruth's space.
   */
  choice: string | number;
  /** Confidence tap, when the drill offers one; null otherwise. */
  confidence: Confidence | null;
  /** Milliseconds from item shown to pointerdown commit. */
  latencyMs: number;
}

/** The graded result, computed locally from labels in <100ms. */
export interface Verdict {
  /** Was the answer correct (for probabilistic grading: scored above chance)? */
  correct: boolean;
  /** Points earned on the item's 0–100 scale (e.g. exact=100, ±1 row=60). */
  score: number;
  /** The rendered one-line auction-logic explanation (never absent — no naked verdicts). */
  explanation: string;
  /** Template id the explanation was rendered from (audit trail, GDD §10-B). */
  explanationTemplateId: string;
  /** Guide section anchors backing the explanation's claims. */
  refs: string[];
  /** True when this was a 3×-rating-weight cardinal error (Drill I trend-called-balance). */
  cardinal: boolean;
  /** Brier component (p − o)² when a probability/confidence was logged; null otherwise. */
  brier: number | null;
}

/* ----------------------------------------------------------------------------
   Rating & scheduling (GDD §3 mastery, §5 rating; schedule lib)
   -------------------------------------------------------------------------- */

/** Glicko-2 state for one skill node (per-node rating, GDD §5). */
export interface RatingState {
  /** Skill node id (drill id or tree node key). */
  nodeId: string;
  /** Glicko-2 rating on the display scale (initial 1500). */
  rating: number;
  /** Rating deviation RD (initial: schedule/glicko.INITIAL_RD). */
  rd: number;
  /** Volatility σ (initial 0.06). */
  volatility: number;
  /** Scored answers absorbed so far (for co-rating migration threshold). */
  nAnswers: number;
}

/** Node mastery states on the tree (GDD §3). */
export type NodeState = 'locked' | 'learning' | 'checkpoint-armed' | 'mastered' | 'rusty';

/** One entry in the spaced-repetition queue (misses on 1d/3d/7d intervals). */
export interface QueueItem {
  /** Unique queue entry id. */
  id: string;
  /** Skill node the miss belongs to. */
  nodeId: string;
  /** Drill of the missed item. */
  drillId: DrillId;
  /** Seed of the ORIGINAL missed item; Woodpecker serves a sibling (same seed, nudged knob substream). */
  seed: string;
  /** paramsVersion of the original item. */
  paramsVersion: string;
  /** Epoch ms when the item becomes due. */
  dueAt: number;
  /** Which expanding interval this is on: 0 = 1d, 1 = 3d, 2 = 7d. */
  intervalIndex: number;
  /** Consecutive lapses on this pattern (8 → leech rule, GDD §3). */
  lapses: number;
}

/**
 * One executed boss/sim trade, attached to a boss-mode decision. Consumed by
 * the Stats expectancy ledger (setup × regime, cells gray until n ≥ 30) and
 * the R-multiple histogram (GDD §5 Expectancy Dashboard). Optional — drill
 * decisions never carry it; the boss engine writes it on fills.
 */
export interface TradeFill {
  /** Playbook setup name declared on the ticket (guide Part IV §4.1–4.9). */
  setup: string;
  /** Regime tag declared on the ticket at entry. */
  regime: Regime;
  /** Realized R-multiple of the trade (P&L / initial risk). */
  rMultiple: number;
}

/** One persisted scored decision — the ledger row (GDD §9, Dexie table). */
export interface DecisionRecord {
  /** Unique decision id. */
  decisionId: string;
  /** Skill node. */
  nodeId: string;
  /** Drill. */
  drillId: DrillId;
  /** Session seed (with paramsVersion, re-derives the verdict forever). */
  seed: string;
  /** Generator parameter-pack version. */
  paramsVersion: string;
  /** The committed answer. */
  answer: Answer;
  /** The graded verdict. */
  verdict: Verdict;
  /** Epoch ms of the decision. */
  at: number;
  /** Mode the decision was made in (only 'rated' and 'checkpoint' move rating). */
  mode: 'rated' | 'rush' | 'streak' | 'woodpecker' | 'calibration' | 'checkpoint' | 'warmup' | 'boss';
  /** The executed trade, boss-mode decisions only (Stats expectancy ledger). */
  trade?: TradeFill;
  /**
   * The served item's knob-derived rating (DrillItem.itemRating) at decision
   * time. Optional (older rows lack it). Rating replay (schedule/tree) uses
   * it as the exact Glicko opponent so replayed history matches the live
   * update path one-for-one — never a second algorithm for the same number.
   */
  itemRating?: number;
}
