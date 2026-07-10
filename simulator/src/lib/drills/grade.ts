/* ============================================================================
   @drills/grade — the local grader: (LoopItem, Answer) → Verdict in O(1).

   Computed entirely from the labels/context shipped with the item, so the
   ≤100ms verdict budget never touches anything slow (GDD §2/§9). No naked
   verdicts: every path renders a template from @drills/templates with
   engine-computed slots and carries the guide refs.

   Scoring (GDD §4):
   · A  — exact row 100 · ±1 row 60 · else 0 (correct = within ±1).
   · B  — 100/0; 2–3-tick items are "judgment calls": score 50, never counted
          against the player, excluded from the Brier ledger (GDD: "graded
          probabilistically, marked judgment call, never binary").
   · D/F — 100/0.
   · I  — 100/0; a scripted trend day answered FADE (responsive on a trend
          day = "called balance") is the cardinal error: 3× rating weight
          (applied by the caller via glicko.ratingWeight) + the forced
          template of the specific tells.
   Confidence taps on chip drills fold into the Brier ledger as implicit
   90/75/55% (GDD §5, schedule/persist.CONFIDENCE_PROB).

   Pure TS — zero DOM/Svelte imports.
   ========================================================================== */

import type { Answer, Verdict } from '../types';
import { rowToPrice } from '../core/profile';
import { brierFromConfidence } from '../schedule/persist';
import { SHAPE_CLAUSE, renderTemplate, template } from './templates';
import type { LoopItem } from './items';

function verdictOf(
  correct: boolean,
  score: number,
  templateId: string,
  slots: Record<string, string | number>,
  answer: Answer,
  opts: { cardinal?: boolean; brierEligible?: boolean } = {},
): Verdict {
  const t = template(templateId);
  return {
    correct,
    score,
    explanation: renderTemplate(t, slots),
    explanationTemplateId: templateId,
    refs: t.refs,
    cardinal: opts.cardinal ?? false,
    brier:
      (opts.brierEligible ?? true) && answer.confidence
        ? brierFromConfidence(answer.confidence, correct)
        : null,
  };
}

/* ---------------------------------------------------------------- Drill A */

function gradePocVa(li: LoopItem, answer: Answer): Verdict {
  const { item, ctx } = li;
  const p = item.stimulus.profile;
  const truth = item.groundTruth as number;
  const pickRow = typeof answer.choice === 'number' ? answer.choice : Number.NaN;
  const dist = Math.abs(pickRow - truth);
  const target = ctx.target ?? 'POC';
  const price = rowToPrice(p, truth).toFixed(2);

  if (dist === 0) {
    return target === 'POC'
      ? verdictOf(true, 100, 'pocva.hit.exact.poc', { price }, answer)
      : verdictOf(true, 100, 'pocva.hit.exact.va', { k: ctx.expansionRounds ?? 0 }, answer);
  }
  const dir = truth > pickRow ? 'higher' : 'lower';
  if (dist === 1) {
    return verdictOf(true, 60, 'pocva.hit.near', { target, price, dir: truth > pickRow ? 'above' : 'below' }, answer);
  }
  if (target === 'POC') {
    return verdictOf(false, 0, 'pocva.miss.poc', { n: dist, dir, price }, answer);
  }
  // VAH/VAL miss: did the expansion run PAST the tapped row (truth farther
  // from the POC than the tap), or stop short of it?
  const truthFarther = Math.abs(truth - p.poc) > Math.abs(pickRow - p.poc);
  const side = target === 'VAH' ? 'upper' : 'lower';
  const rel = target === 'VAH' ? 'above' : 'below';
  // Wrong side of the POC (tap below it when VAH was asked / above it for
  // VAL): neither "ran out" nor "swallowed" describes that geometry — the
  // requested edge lives on the other side of the POC by construction.
  // Template routing only; the distance/score math above is untouched.
  const wrongSide = target === 'VAH' ? pickRow < p.poc : pickRow > p.poc;
  if (wrongSide) {
    return verdictOf(
      false,
      0,
      'pocva.miss.va.wrongSide',
      { target, n: dist, dir, rel, tapRel: target === 'VAH' ? 'below' : 'above' },
      answer,
    );
  }
  if (truthFarther) {
    // mention the HVN bulge only when the profile actually has one out there
    const bulge = p.hvnRanges.some((r) => (target === 'VAH' ? r.hi > pickRow : r.lo < pickRow));
    return verdictOf(
      false,
      0,
      bulge ? 'pocva.miss.va.swallowed' : 'pocva.miss.va.fatter',
      { target, n: dist, dir, side, rel },
      answer,
    );
  }
  return verdictOf(false, 0, 'pocva.miss.va.ranOut', { target, n: dist, dir, rel }, answer);
}

/* ---------------------------------------------------------------- Drill B */

function gradeExcess(li: LoopItem, answer: Answer): Verdict {
  const { ctx } = li;
  const side = ctx.extremeSide ?? 'high';
  const ticks = ctx.extremeTicks ?? 0;
  if (ctx.extremeClass === 'judgment') {
    // 2–3-tick band: judgment call, never a binary (GDD Drill B). Excluded
    // from accuracy penalties and from the Brier ledger.
    return verdictOf(true, 50, 'excess.judgment', { n: ticks }, answer, { brierEligible: false });
  }
  const truth = li.item.groundTruth as string;
  const pick = String(answer.choice);
  const correct = pick === truth;
  if (correct) {
    return truth === 'EXCESS'
      ? verdictOf(true, 100, 'excess.hit.excess', { n: ticks }, answer)
      : verdictOf(true, 100, 'excess.hit.poor', { side }, answer);
  }
  return truth === 'POOR'
    ? verdictOf(false, 0, 'excess.miss.poorCalledExcess', { side, barrier: side === 'high' ? 'resistance' : 'support' }, answer)
    : verdictOf(false, 0, 'excess.miss.excessCalledPoor', { side, n: ticks }, answer);
}

/* ---------------------------------------------------------------- Drill D */

function gradeShape(li: LoopItem, answer: Answer): Verdict {
  const truth = li.item.groundTruth as string;
  const pick = String(answer.choice);
  const clause = SHAPE_CLAUSE[truth === 'TREND' ? 'thin-trend' : truth];
  if (pick === truth) {
    return verdictOf(true, 100, 'shape.hit', { truth, truthClause: clause }, answer);
  }
  const pair = new Set([truth, pick]);
  if (pair.has('P') && pair.has('b')) {
    return verdictOf(
      false,
      0,
      'shape.miss.PbConfusion',
      {
        truth,
        pick,
        bulge: truth === 'b' ? 'low' : 'high',
        tail: truth === 'b' ? 'upper' : 'lower',
      },
      answer,
    );
  }
  if (truth === 'TREND' && pick === 'B') {
    return verdictOf(false, 0, 'shape.miss.trendCalledB', {}, answer);
  }
  if (truth === 'B' && pick === 'TREND') {
    return verdictOf(false, 0, 'shape.miss.BCalledTrend', {}, answer);
  }
  return verdictOf(false, 0, 'shape.miss.generic', { truth, pick, truthClause: clause }, answer);
}

/* ---------------------------------------------------------------- Drill F */

function gradeOpenType(li: LoopItem, answer: Answer): Verdict {
  const truth = li.item.groundTruth as string;
  const pick = String(answer.choice);
  const correct = pick === truth;
  const price = (li.ctx.openPrice ?? li.item.stimulus.bars[0].o).toFixed(2);
  return verdictOf(
    correct,
    correct ? 100 : 0,
    li.item.explanationTemplateId, // open.truth.<openType>
    { glyph: correct ? '✓' : '✗', price },
    answer,
  );
}

/* ---------------------------------------------------------------- Drill I */

function gradeRegime(li: LoopItem, answer: Answer): Verdict {
  const { ctx } = li;
  const truth = li.item.groundTruth as string;
  const pick = String(answer.choice);
  if (pick === truth) {
    const id =
      truth === 'FADE' ? 'regime.hit.balance' : truth === 'FOLLOW' ? 'regime.hit.imbalance' : 'regime.hit.nontrend';
    return verdictOf(true, 100, id, { pct: ctx.rangePct ?? 0 }, answer);
  }
  // Cardinal error (GDD Drill I): a scripted trend day answered responsive —
  // "trend called balance". 3× rating weight applied by the caller.
  if (ctx.dayType === 'trend' && pick === 'FADE') {
    return verdictOf(
      false,
      0,
      'regime.miss.cardinal',
      { time: ctx.otfSince ?? barClockFallback(), dir: ctx.stairDir ?? 'up' },
      answer,
      { cardinal: true },
    );
  }
  const id =
    truth === 'FADE' ? 'regime.miss.balance' : truth === 'FOLLOW' ? 'regime.miss.imbalance' : 'regime.miss.nontrend';
  return verdictOf(false, 0, id, { pct: ctx.rangePct ?? 0 }, answer);
}

function barClockFallback(): string {
  return '10:30'; // IB end — only reachable if ctx.otfSince was never set
}

/* ---------------------------------------------------------------- dispatch */

/** Grade an answer against its item. Pure, instant, deterministic. */
export function gradeLoopItem(li: LoopItem, answer: Answer): Verdict {
  switch (li.item.drillId) {
    case 'poc-va-snap':
      return gradePocVa(li, answer);
    case 'excess-or-poor':
      return gradeExcess(li, answer);
    case 'shape-alphabet':
      return gradeShape(li, answer);
    case 'open-type-ladder':
      return gradeOpenType(li, answer);
    case 'regime-gate':
      return gradeRegime(li, answer);
    default:
      throw new Error(`gradeLoopItem: unwired drill ${li.item.drillId}`);
  }
}
