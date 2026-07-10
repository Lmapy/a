/* ============================================================================
   @boss/templates — canonical feedback strings for Boss 1, the Trend-Day
   Fade Gauntlet (GDD §6 Boss 1, §10-B audit rules).

   Every rendered boss explanation comes from this registry. Claims are
   sourced from the domain guide (/docs/volume-profiles-and-auction-market-
   theory.md), Part VII.1 misconception 1 (fading trend days) and Part IV
   §4.0/§4.11; each template carries its guide refs. Where the GDD gives a
   line verbatim (the debrief line), it is used verbatim.

   Style rules (same tests as @drills/templates): task-referenced, ≤160 chars
   to the first period, the literal "80%" never appears, no naked verdicts.

   Pure TS — zero DOM/Svelte imports.
   ========================================================================== */

export interface BossTemplate {
  /** Registry key, `boss1.${event}.${verdictClass}`. */
  id: string;
  /** Guide/GDD section anchors backing every claim. */
  refs: string[];
  /** Template string with {slot} placeholders (engine-computed values only). */
  text: string;
}

/** Fill {slot} placeholders. Throws on a missing slot (audit-friendly). */
export function renderBossTemplate(
  t: BossTemplate,
  slots: Record<string, string | number>,
): string {
  return t.text.replace(/\{(\w+)\}/g, (_, k: string) => {
    if (!(k in slots)) throw new Error(`boss template ${t.id}: missing slot {${k}}`);
    return String(slots[k]);
  });
}

const T = (id: string, refs: string[], text: string): [string, BossTemplate] => [
  id,
  { id, refs, text },
];

export const BOSS_TEMPLATES: ReadonlyMap<string, BossTemplate> = new Map<string, BossTemplate>([
  /* ---- decision verdicts (order ticket) --------------------------------- */
  // The cardinal error this boss exists to punish (misconceptions 2 & 6):
  T('boss1.fade.cardinal', ['guide:VII.7.1-1', 'guide:IV.4.0-regime', 'gdd:6-boss1'],
    '✗✗ Responsive fade on a labeled trend day — one-timeframing since {time} never broke. Every fade from value looks perfect on a trend day, and every one loses.'),
  T('boss1.gowith.hit', ['guide:IV.4.0-regime', 'guide:I-otf'],
    '✓ Imbalance read, initiative trade — with the one-timeframing move, never against it. Structural stop under the pullback low at {stop}.'),
  T('boss1.gowith.wrongRead', ['guide:IV.4.0-regime', 'guide:III.3.4-migration'],
    '✓ Legal trade, wrong read — value stair-stepping {dir} with one-timeframing intact is imbalance, not balance. Log the read, not the fill.'),
  T('boss1.stand.hit', ['guide:VII.7.1-1', 'gdd:6-boss1'],
    '✓ Standing aside is a position. On a trend day the winning lines are initiative-only or flat — the responsive fade is the trap.'),
  T('boss1.stand.wrongRead', ['guide:IV.4.0-regime', 'guide:III.3.4-migration'],
    '— Flat is legal, but the read was wrong: value migrating {dir} behind unbroken one-timeframing is imbalance, not balance.'),

  /* ---- KILL/HOLD reflex verdicts ---------------------------------------- */
  T('boss1.kill.hit', ['guide:VII.7.1-1', 'guide:I-otf'],
    '✓ KILL — short against an unbroken one-timeframing move. A market accepting price outside value is not a fade; cut it.'),
  T('boss1.kill.miss', ['guide:I-otf', 'guide:IV.4.11-stops'],
    '✗ HOLD against one-timeframing — the auction kept migrating against the position and nothing structural invalidated the trend. Counter-trend plus no break means KILL.'),
  T('boss1.hold.hit', ['guide:III.3.2-trend', 'guide:IV.4.8-pullback'],
    '✓ HOLD — the pullback stayed inside the trend contract with one-timeframing intact. Trend pullbacks are entries, not exits.'),
  T('boss1.hold.miss', ['guide:III.3.2-trend', 'guide:IV.4.11-stops'],
    '✗ KILL on a with-trend position — no bracket broke and the pullback held. Nothing structural invalidated the trade; the exit was emotion, not structure.'),
  T('boss1.reflex.slept', ['gdd:6-boss-mechanics'],
    '✗ Slept through it — {sec}s and no answer. A position that cannot be re-underwritten in five seconds is a position that is not understood.'),

  /* ---- desk-limit breach -------------------------------------------------- */
  T('boss1.breach.daily', ['gdd:6-prop-limits', 'guide:VIII.8.1-sizing'],
    'Daily loss limit breached at {clock} — the session ends here, open positions included. The limit is the risk plan, not a suggestion.'),
  T('boss1.breach.trailing', ['gdd:6-prop-limits', 'guide:VIII.8.1-sizing'],
    'Trailing drawdown breached at {clock} — {dd}R given back from the peak ends the session, open positions included.'),
]);

/** Lookup that throws on unknown ids (grader bug = loud failure). */
export function bossTemplate(id: string): BossTemplate {
  const t = BOSS_TEMPLATES.get(id);
  if (!t) throw new Error(`unknown boss template: ${id}`);
  return t;
}

/* ----------------------------------------------------------------------------
   Debrief copy (GDD §6 Boss 1 + guide Part VII.1 — quoted, with provenance)
   -------------------------------------------------------------------------- */

/** GDD §6 Boss 1 debrief line — verbatim. */
export const BOSS1_DEBRIEF_LINE = 'Every fade looked perfect. Every fade lost. That is what a trend day does.';

/** Guide Part VII.1, misconception 1 — the trend-day logic, quoted. */
export const BOSS1_GUIDE_QUOTE =
  'On a genuine trend day every fade from the VAH "looks perfect — and every one loses." ' +
  'Antidotes: the one-timeframing check, the value-migration check, the elongating shape, ' +
  'expanding volume with the move. When in doubt: a market accepting price outside value is not a fade.';

/** Where the quote comes from (shown with it — provenance, never naked). */
export const BOSS1_GUIDE_REF = 'Guide, Part VII.1 — The Classic Mistakes, №1';

/** The luck-channel bright line (GDD §5 boss scoring). */
export const OUTCOME_LINE = 'Outcome ≠ decision quality — P&L is shown, never scored.';

/** The misconception this boss punishes (intro trap line, GDD §6). */
export const BOSS1_TRAP_LINE_HEAD = 'Every fade from VAH will look perfect — ';
export const BOSS1_TRAP_LINE_EM = 'and lose';
export const BOSS1_TRAP_LINE_TAIL = '. One-timeframing never breaks.';
