/* ============================================================================
   @drills/summary — block-of-10 summary math (GDD §2 interstitial + §5).

   Everything here is computed from the block's served items and verdicts:
   accuracy, median latency vs par, Bot Points (base-rate bot per GDD §5,
   the bot answering the pool's declared rate), and the base-rate
   interstitial sentence — always the MEASURED mixture of the block actually
   served, never a folklore number.

   Pure TS — zero DOM/Svelte imports.
   ========================================================================== */

import type { Answer, Verdict } from '../types';
import { botPoints, confidenceProb } from '../schedule/persist';
import type { BotComparableEntry } from '../schedule/persist';
import type { LoopDrillId, LoopItem } from './items';

export interface RepRecord {
  li: LoopItem;
  answer: Answer;
  verdict: Verdict;
}

export interface BlockSummary {
  reps: number;
  correct: number;
  accuracy: number;
  medianLatencyMs: number;
  parMs: number;
  /** Bot Points over the block's confidence-tapped reps (null when none). */
  botPoints: number | null;
  /** The base-rate interstitial line (GDD §2, every ~10 reps). */
  baseRateLine: string;
  /**
   * The round calibration sentence (GDD §8 screen 3: "When you said sure,
   * you were right 84% — well calibrated this round."). Null when the block
   * carried no confidence-tapped, Brier-eligible reps.
   */
  calibrationLine: string | null;
  misses: number;
  cardinals: number;
}

export function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = xs.slice().sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/**
 * The base-rate reveal, measured from the block that was actually served
 * (revealed reference classes beat scoring feedback for calibration —
 * GDD §2 quoting Lichtenstein & Fischhoff).
 */
export function baseRateLine(drillId: LoopDrillId, reps: RepRecord[]): string {
  const n = reps.length;
  if (n === 0) return '';
  const pct = (k: number): number => Math.round((k / n) * 100);
  switch (drillId) {
    case 'excess-or-poor': {
      const poor = reps.filter((r) => r.li.item.groundTruth === 'POOR').length;
      return `In this set, highlighted extremes were poor ${pct(poor)}% of the time.`;
    }
    case 'shape-alphabet': {
      const counts = new Map<string, number>();
      for (const r of reps) {
        const k = String(r.li.item.groundTruth);
        counts.set(k, (counts.get(k) ?? 0) + 1);
      }
      const mix = ['D', 'P', 'b', 'B', 'TREND']
        .filter((s) => counts.has(s))
        .map((s) => `${s} ${counts.get(s)}`)
        .join(' · ');
      return `This set's mix — ${mix} of ${n}.`;
    }
    case 'poc-va-snap': {
      const widths = reps.map((r) => {
        const p = r.li.item.stimulus.profile;
        return p.vah - p.val + 1;
      });
      return `In this set, the 70% value area averaged ${Math.round(widths.reduce((a, b) => a + b, 0) / n)} of ${Math.round(
        reps.reduce((a, r) => a + r.li.item.stimulus.profile.rows.length, 0) / n,
      )} rows.`;
    }
    case 'open-type-ladder': {
      const conviction = reps.filter(
        (r) => r.li.item.groundTruth === 'DRIVE' || r.li.item.groundTruth === 'TEST-DRIVE',
      ).length;
      return `In this set, ${conviction} of ${n} opens carried conviction (drive or test-drive).`;
    }
    case 'regime-gate': {
      const imb = reps.filter((r) => r.li.item.groundTruth === 'FOLLOW').length;
      return `In this set, ${pct(imb)}% of snapshots were genuine imbalance.`;
    }
  }
}

/**
 * Bot Points over the block (GDD §5 headline): the bot answers the pool's
 * declared base rate for "this call is right" — uniform over the chip set
 * (the pool's uniform/declared mixture) — while the player's confidence tap
 * is the implicit 90/75/55%. Only confidence-tapped, brier-eligible reps
 * enter. Null when the block carried no calibration signal.
 */
export function blockBotPoints(reps: RepRecord[]): number | null {
  const entries: BotComparableEntry[] = [];
  for (const r of reps) {
    if (!r.answer.confidence || r.verdict.brier === null) continue;
    const nChoices = Math.max(2, r.li.item.choices.length);
    entries.push({
      p: confidenceProb(r.answer.confidence),
      hit: r.verdict.correct,
      at: 0,
      baseRate: 1 / nChoices,
    });
  }
  if (entries.length === 0) return null;
  return botPoints(entries);
}

/**
 * The round calibration sentence (GDD §8 screen 3). Measured from the
 * block's confidence-tapped, Brier-eligible reps: the sentence reports the
 * MOST-CONFIDENT tap the player actually used this round ("sure" before
 * "lean" before "guess"), its observed hit rate, and the honesty clause vs
 * the tap's implicit probability (90/75/55%). Null when nothing qualifies
 * (snap drills, or a block of judgment calls).
 */
export function calibrationLine(reps: RepRecord[]): string | null {
  const tapped = reps.filter((r) => r.answer.confidence && r.verdict.brier !== null);
  if (tapped.length === 0) return null;
  const order: NonNullable<Answer['confidence']>[] = ['sure', 'lean', 'guess'];
  const conf = order.find((c) => tapped.some((r) => r.answer.confidence === c))!;
  const of = tapped.filter((r) => r.answer.confidence === conf);
  const hitPct = Math.round((of.filter((r) => r.verdict.correct).length / of.length) * 100);
  const saidPct = Math.round(confidenceProb(conf) * 100);
  const gap = hitPct - saidPct;
  const clause =
    Math.abs(gap) <= 5
      ? 'well calibrated this round'
      : gap < 0
        ? 'overconfident this round'
        : 'underconfident this round';
  return `When you said ${conf}, you were right ${hitPct}% — ${clause}.`;
}

/** Assemble the whole block summary. */
export function summarize(drillId: LoopDrillId, reps: RepRecord[]): BlockSummary {
  const correct = reps.filter((r) => r.verdict.correct).length;
  return {
    reps: reps.length,
    correct,
    accuracy: reps.length > 0 ? correct / reps.length : 0,
    medianLatencyMs: median(reps.map((r) => r.answer.latencyMs)),
    parMs: reps[0]?.li.item.parMs ?? 0,
    botPoints: blockBotPoints(reps),
    baseRateLine: baseRateLine(drillId, reps),
    calibrationLine: calibrationLine(reps),
    misses: reps.filter((r) => !r.verdict.correct).length,
    cardinals: reps.filter((r) => r.verdict.cardinal).length,
  };
}
