/* ============================================================================
   @core/tpo — TPO (time-price-opportunity) profile: 30-min brackets lettered
   A, B, C…, TPO POC/VA, initial balance, single prints.

   Reference: /home/user/a/scripts/figures/fig-12-vp-tpo-vwap.py (TPO build).
   Pure TS — zero DOM/Svelte imports. Used by Drills F/G/H/I stimuli and the
   one-timeframing detector.

   STATUS: COMPLETE — bracket lettering, IB, TPO counts, single prints, and
   the one-timeframing break detector; POC/VA reuse the volume-profile
   expansion on TPO counts (standard practice). fig-12 golden parity pinned
   by __fixtures__/fig12.ts. Owner: core team.
   ========================================================================== */

import type { Bar, TpoProfile } from '../types';
import { expandValueArea, findPoc } from './profile';

/** Bars per 30-minute TPO bracket at 1-min resolution. */
export const BARS_PER_BRACKET = 30;

/** Bracket letters in order (A = first 30 minutes). */
export const BRACKET_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

/** 30-min bracket index for a bar index. */
export function bracketOf(barIndex: number): number {
  return Math.floor(barIndex / BARS_PER_BRACKET);
}

/**
 * Build a TPO profile from 1-min bars.
 * Each bracket marks every row its price range [l, h] touched.
 *
 * @param bars     session bars from the open (bracket letters assume bars[i].t === i)
 * @param rowStep  price per row, > 0
 */
export function buildTpoProfile(bars: Bar[], rowStep: number): TpoProfile {
  if (bars.length === 0 || rowStep <= 0) {
    throw new Error('buildTpoProfile: need at least one bar and rowStep > 0');
  }
  let lo = Infinity;
  let hi = -Infinity;
  for (const b of bars) {
    if (b.l < lo) lo = b.l;
    if (b.h > hi) hi = b.h;
  }
  const minPrice = Math.floor(lo / rowStep) * rowStep;
  const nRows = Math.max(1, Math.floor((hi - minPrice) / rowStep) + 1);

  // Per-bracket touched-row sets
  const nBrackets = bracketOf(bars[bars.length - 1].t) + 1;
  const touched: boolean[][] = Array.from({ length: nBrackets }, () =>
    new Array<boolean>(nRows).fill(false),
  );
  for (const b of bars) {
    const br = bracketOf(b.t);
    const rLo = Math.max(0, Math.floor((b.l - minPrice) / rowStep));
    const rHi = Math.min(nRows - 1, Math.floor((b.h - minPrice) / rowStep));
    for (let r = rLo; r <= rHi; r++) touched[br][r] = true;
  }

  const rows: string[] = new Array(nRows).fill('');
  for (let br = 0; br < nBrackets; br++) {
    const letter = BRACKET_LETTERS[br] ?? '?';
    for (let r = 0; r < nRows; r++) {
      if (touched[br][r]) rows[r] += letter;
    }
  }

  const counts = rows.map((s) => s.length);
  const poc = findPoc(counts);
  const { vah, val } = expandValueArea(counts, poc, 0.7);

  // Initial balance = extremes of brackets A + B (first hour)
  let ibHigh = -Infinity;
  let ibLow = Infinity;
  for (const b of bars) {
    if (bracketOf(b.t) > 1) continue;
    if (b.h > ibHigh) ibHigh = b.h;
    if (b.l < ibLow) ibLow = b.l;
  }

  const singlePrintRows: number[] = [];
  for (let r = 0; r < nRows; r++) if (counts[r] === 1) singlePrintRows.push(r);

  return { minPrice, rowStep, rows, poc, vah, val, ibHigh, ibLow, singlePrintRows };
}

/**
 * Detect the bar index where one-timeframing breaks (Drill G ground-truth
 * verification). One-timeframing UP = every 30-min bracket trades entirely at
 * or above the prior bracket's low (higher lows); the break is the FIRST BAR
 * that trades below the previous completed bracket's low. DOWN is the mirror
 * (lower highs; break = first bar above the previous bracket's high).
 *
 * The first bracket has no prior reference, so a break can only occur from
 * bracket 1 on. Equal lows/highs do NOT break control (Dalton: control holds
 * until the opposing side actually violates the prior period's extreme).
 *
 * @param bars       session bars from the open (bracket letters assume bars[i].t === i)
 * @param direction  which side is in control ('up' = buyers)
 * @returns bar index of the break, or null if control never breaks
 */
export function findOneTimeframingBreak(
  bars: Bar[],
  direction: 'up' | 'down',
): number | null {
  let prevExtreme: number | null = null; // prior completed bracket's low (up) / high (down)
  let curExtreme: number | null = null;
  let curBracket = -1;
  for (const b of bars) {
    const br = bracketOf(b.t);
    if (br !== curBracket) {
      // close out any brackets between curBracket and br (gaps roll forward)
      if (curBracket >= 0) prevExtreme = curExtreme;
      curBracket = br;
      curExtreme = null;
    }
    if (prevExtreme !== null) {
      if (direction === 'up' && b.l < prevExtreme) return b.t;
      if (direction === 'down' && b.h > prevExtreme) return b.t;
    }
    const x = direction === 'up' ? b.l : b.h;
    if (curExtreme === null) curExtreme = x;
    else curExtreme = direction === 'up' ? Math.min(curExtreme, x) : Math.max(curExtreme, x);
  }
  return null;
}
