/* ============================================================================
   @drills/templates — the canonical feedback-template registry (GDD §10-B).

   Every rendered explanation comes from this registry, keyed by
   (drill, verdict-class, misconception). Templates quote the domain guide
   (/docs/volume-profiles-and-auction-market-theory.md); each entry carries
   the guide section anchors backing its claims (Verdict.refs). Where the GDD
   §4 drill catalog gives a template verbatim, it is used verbatim (slots in
   {braces} are filled with engine-computed values only).

   Style rules enforced by tests (GDD §10-B):
   · task-referenced, never person-referenced (no "you always/you're bad");
   · ≤160 chars to the first period;
   · the literal string "80%" never appears (folklore-audit contexts only,
     none of which live in this registry);
   · no naked verdicts — every template is a full auction-logic sentence.

   Pure TS — zero DOM/Svelte imports.
   ========================================================================== */

export interface FeedbackTemplate {
  /** Registry key, `${drill}.${verdictClass}[.${misconception}]`. */
  id: string;
  /** Guide section anchors backing every claim in the template. */
  refs: string[];
  /** The template string with {slot} placeholders. */
  text: string;
}

/** Fill {slot} placeholders. Throws on a missing slot (audit-friendly). */
export function renderTemplate(t: FeedbackTemplate, slots: Record<string, string | number>): string {
  return t.text.replace(/\{(\w+)\}/g, (_, k: string) => {
    if (!(k in slots)) throw new Error(`template ${t.id}: missing slot {${k}}`);
    return String(slots[k]);
  });
}

const T = (id: string, refs: string[], text: string): [string, FeedbackTemplate] => [
  id,
  { id, refs, text },
];

/**
 * The registry. Claims sourced from the guide:
 * · POC = "fairest price", most-volume row (Part II §2.2, glossary).
 * · VA = 70% expansion (Part II §2.2).
 * · Excess ends the auction; poor extremes = unfinished business, repair
 *   magnets (Part I "Excess"/"Poor high/low", cheat-sheet 7).
 * · P = short-covering above, b = liquidation below (Part III §3.1).
 * · thin-trend: long, skinny, one-timeframe control, do not fade (§3.1).
 * · B/DD: two distributions joined by an LVN neck (§3.1–3.2).
 * · Open types ladder (§3.3, fig-06).
 * · Regime: balance→responsive, imbalance→initiative only; never fade
 *   one-timeframing (§4.0, cheat-sheet 3/6; Part VII §7.1 misconceptions
 *   1 — fading trend days — and 5 — wrong day-type template).
 */
export const TEMPLATES: ReadonlyMap<string, FeedbackTemplate> = new Map<string, FeedbackTemplate>([
  /* -------- Drill A — POC + VA Snap ------------------------------------- */
  T('pocva.hit.exact.poc', ['guide:II.2.2-poc'],
    '✓ Exact. {price} printed the most volume — the mode of the session, its fairest price.'),
  T('pocva.hit.exact.va', ['guide:II.2.2-va'],
    '✓ Exact. The expansion ran {k} rounds; both edges are where the 70% mass ends.'),
  T('pocva.hit.near', ['guide:II.2.2-va'],
    '✓ Within a row. {target} is {price} — one row {dir} the tap; ±1 row scores at this zoom.'),
  // GDD Drill A miss template, direction-generalized (verbatim clause kept):
  T('pocva.miss.va.swallowed', ['guide:II.2.2-va', 'guide:II.2.3-hvn'],
    '✗ {target} is {n} rows {dir} — the 70% expansion swallowed the {side} HVN bulge before that row. Value is fatter {rel} the POC than it looks.'),
  T('pocva.miss.va.ranOut', ['guide:II.2.2-va'],
    '✗ {target} is {n} rows {dir} — the 70% mass ran out before that row. Value is thinner {rel} the POC than it looks.'),
  // Wrong side of the POC entirely (the tap sits below the POC when VAH was
  // asked, or above it for VAL) — a different geometry from a depth misread:
  T('pocva.miss.va.wrongSide', ['guide:II.2.2-va'],
    '✗ {target} is {n} rows {dir} — that tap sits {tapRel} the POC, but the {target} always sits {rel} it. The 70% value area brackets the POC from both sides.'),
  T('pocva.miss.va.fatter', ['guide:II.2.2-va'],
    '✗ {target} is {n} rows {dir} — the 70% expansion absorbed more {side} rows than that. Value is fatter {rel} the POC than it looks.'),
  T('pocva.miss.poc', ['guide:II.2.2-poc'],
    '✗ The POC is {n} rows {dir} at {price} — the single row with the most volume, not the visual middle of the bell.'),

  /* -------- Drill B — Excess or Poor ------------------------------------ */
  // GDD Drill B miss template (verbatim, side/noun slotted):
  T('excess.miss.poorCalledExcess', ['guide:I-poor', 'guide:cheat-7'],
    '✗ Poor {side} — flat touches, no tail. Excess ends the auction; a poor extreme is unfinished business and a repair magnet, not {barrier}.'),
  T('excess.miss.excessCalledPoor', ['guide:I-excess', 'guide:cheat-7'],
    '✗ Excess {side} — a {n}-tick tail of swift rejection. Excess marks the end of one auction and the start of another; this extreme is finished business.'),
  T('excess.hit.poor', ['guide:I-poor'],
    '✓ Poor {side} — even volume right up to the edge. Unfinished business: elevated odds the market revisits and repairs it.'),
  T('excess.hit.excess', ['guide:I-excess'],
    '✓ Excess — the {n}-tick tail is a swift, decisive rejection. A finished auction; extremes with proper excess act as durable references.'),
  T('excess.judgment', ['guide:I-excess', 'guide:I-poor'],
    'Judgment call — a {n}-tick tail sits between poor and excess, so this one is graded as a coin flip, not a binary. The read to log is the context.'),

  /* -------- Drill D — Shape Alphabet ------------------------------------ */
  // GDD Drill D P/b confusion template (verbatim):
  T('shape.miss.PbConfusion', ['guide:III.3.1-P', 'guide:III.3.1-b'],
    "✗ That's a {truth}, not a {pick} — the bulge sits {bulge} with a thin {tail} tail. P is short-covering above; b is liquidation below. Mirror-image, opposite story."),
  T('shape.miss.trendCalledB', ['guide:III.3.1-trend', 'guide:III.3.2-dd'],
    '✗ Trend, not B — stacked small distributions, no single clean LVN neck. A true B is two fat boxes joined by one thin traverse.'),
  T('shape.miss.BCalledTrend', ['guide:III.3.2-dd', 'guide:III.3.1-trend'],
    "✗ That's a B — two distributions joined by one thin LVN neck. A thin-trend profile stays skinny and never builds two fat boxes."),
  T('shape.miss.generic', ['guide:III.3.1-shapes'],
    "✗ That's a {truth}, not a {pick} — {truthClause}."),
  T('shape.hit', ['guide:III.3.1-shapes'], '✓ {truth} — {truthClause}.'),

  /* -------- Drill F — Open-Type Conviction Ladder ------------------------ */
  // GDD Drill F template (verbatim first clause; OTF clause only when true):
  T('open.truth.open-drive', ['guide:III.3.3-drive'],
    "{glyph} Open-drive: price never re-traded the {price} open. Conviction opens don't get faded — the open is the line in the sand."),
  T('open.truth.open-test-drive', ['guide:III.3.3-testdrive'],
    '{glyph} Open-test-drive: an early probe found no business and reversed hard through the {price} open. The failed-test extreme is premium trade location.'),
  T('open.truth.open-rejection-reverse', ['guide:III.3.3-rr'],
    '{glyph} Open-rejection-reverse: the first drive met responsive opposition and returned through the {price} open. Moderate conviction — expect two-sided trade.'),
  T('open.truth.open-auction', ['guide:III.3.3-auction'],
    '{glyph} Open-auction: price rotated across the {price} open with no directional resolution. Low conviction — let value develop before leaning either way.'),

  /* -------- Drill I — Regime Gate ---------------------------------------- */
  // GDD Drill I cardinal-miss template (verbatim, time/direction slotted):
  T('regime.miss.cardinal', ['guide:IV.4.0-regime', 'guide:VII.7.1-1'],
    '✗✗ One-timeframing since {time}, value stair-stepping {dir}, pullbacks under 40% — this is imbalance. Every responsive fade looks perfect on a trend day, and every one loses.'),
  T('regime.miss.balance', ['guide:IV.4.0-regime', 'guide:IV.4.1-fade'],
    '✗ Balance — overlapping value, rotation both ways, no one-timeframing. The playbook is responsive: fade the edges, not chase the middle.'),
  T('regime.miss.imbalance', ['guide:IV.4.0-regime', 'guide:I-otf'],
    '✗ Imbalance — value is migrating with one side in control. While a market is one-timeframing, do not fade it; initiative trades only.'),
  T('regime.miss.nontrend', ['guide:III.3.2-nontrend'],
    '✗ No trade — a squat session at {pct}% of typical range, nobody leading. No conviction to follow, no edge worth fading; standing aside is the position.'),
  T('regime.hit.balance', ['guide:IV.4.0-regime', 'guide:IV.4.1-fade'],
    '✓ Balance — overlapping value and two-sided rotation. Responsive playbook: fade the edges back toward the POC, while it lasts.'),
  T('regime.hit.imbalance', ['guide:IV.4.0-regime', 'guide:I-otf'],
    '✓ Imbalance — value migrating, one side in control. Initiative only: trade with the move; fading it is fighting the OTF player.'),
  T('regime.hit.nontrend', ['guide:III.3.2-nontrend'],
    '✓ Stand aside — {pct}% of typical range and no initiative anywhere. The best playbook for nontrend chop is no playbook.'),
]);

/** Lookup that throws on unknown ids (grader bug = loud failure). */
export function template(id: string): FeedbackTemplate {
  const t = TEMPLATES.get(id);
  if (!t) throw new Error(`unknown feedback template: ${id}`);
  return t;
}

/** Per-shape clauses for shape.hit / shape.miss.generic (guide §3.1). */
export const SHAPE_CLAUSE: Record<string, string> = {
  D: 'a symmetric bell of two-sided rotation around an agreed fair price',
  P: 'a fat bulge up top under a thin lower stem — short-covering into balance',
  b: 'a fat bulge at the lows under a thin upper stem — long liquidation',
  B: 'two distributions joined by a thin LVN neck',
  'thin-trend': 'long and skinny, minimal horizontal development — one side in control all session',
};
