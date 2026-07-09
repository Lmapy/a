import { describe, expect, it } from 'vitest';
import type { Bar } from '../types';
import { fig12 } from './__fixtures__/fig12';
import {
  BARS_PER_BRACKET,
  bracketOf,
  buildTpoProfile,
  findOneTimeframingBreak,
} from './tpo';

function bar(t: number, l: number, h: number): Bar {
  return { t, o: l, h, l, c: h, v: 100 };
}

describe('bracketOf', () => {
  it('maps 1-min bars to 30-min brackets', () => {
    expect(bracketOf(0)).toBe(0);
    expect(bracketOf(29)).toBe(0);
    expect(bracketOf(30)).toBe(1);
    expect(bracketOf(BARS_PER_BRACKET * 3 + 5)).toBe(3);
  });
});

describe('buildTpoProfile', () => {
  // Bracket A (bars 0..29) trades 100..101; bracket B (bars 30..59) trades 100.5..102
  const bars: Bar[] = [];
  for (let t = 0; t < 30; t++) bars.push(bar(t, 100, 101));
  for (let t = 30; t < 60; t++) bars.push(bar(t, 100.5, 102));

  it('letters rows by bracket and finds single prints', () => {
    const tpo = buildTpoProfile(bars, 0.5);
    // row at price 100 touched only by A; row at 102 only by B; overlap rows AB
    const rowAt = (price: number) => Math.round((price - tpo.minPrice) / tpo.rowStep);
    expect(tpo.rows[rowAt(100)]).toBe('A');
    expect(tpo.rows[rowAt(101)]).toBe('AB');
    expect(tpo.rows[rowAt(102)]).toBe('B');
    expect(tpo.singlePrintRows).toContain(rowAt(100));
    expect(tpo.singlePrintRows).toContain(rowAt(102));
    expect(tpo.singlePrintRows).not.toContain(rowAt(101));
  });

  it('computes the initial balance from brackets A+B', () => {
    const tpo = buildTpoProfile(bars, 0.5);
    expect(tpo.ibHigh).toBe(102);
    expect(tpo.ibLow).toBe(100);
  });

  it('VA contains the TPO POC', () => {
    const tpo = buildTpoProfile(bars, 0.5);
    expect(tpo.val).toBeLessThanOrEqual(tpo.poc);
    expect(tpo.vah).toBeGreaterThanOrEqual(tpo.poc);
  });

  it('throws on empty input', () => {
    expect(() => buildTpoProfile([], 0.5)).toThrow();
  });
});

describe('golden parity — fig-12 (TPO profile from a minute path)', () => {
  // The Python reference marks, per 30-min bracket, EVERY row between the
  // bracket's min and max (a continuous path touches them all). Point bars
  // can hop over a row, so connect each bar to its predecessor WITHIN the
  // bracket — the union of bar ranges then equals the bracket's [min, max].
  const bars: Bar[] = fig12.price.map((p, i) => {
    const prev = i > 0 && bracketOf(i) === bracketOf(i - 1) ? fig12.price[i - 1] : p;
    return {
      t: i,
      o: prev,
      h: Math.max(p, prev),
      l: Math.min(p, prev),
      c: p,
      v: fig12.vol[i],
    };
  });

  it('TPO POC/VA rows, IB, and max stack match the Python reference', () => {
    const tpo = buildTpoProfile(bars, fig12.rowStep);
    expect(tpo.minPrice).toBe(fig12.minPrice);
    expect(tpo.rows).toHaveLength(fig12.nRows);
    expect(tpo.poc).toBe(fig12.tpoPoc);
    expect(tpo.val).toBe(fig12.tpoVal);
    expect(tpo.vah).toBe(fig12.tpoVah);
    expect(Math.max(...tpo.rows.map((r) => r.length))).toBe(fig12.maxTpoStack);
    expect(tpo.ibHigh).toBe(fig12.ibHigh);
    expect(tpo.ibLow).toBe(fig12.ibLow);
  });
});

describe('findOneTimeframingBreak', () => {
  it('returns the first bar trading below the prior bracket low (up case)', () => {
    const bars: Bar[] = [];
    for (let t = 0; t < 30; t++) bars.push(bar(t, 100, 101)); // A: low 100
    for (let t = 30; t < 60; t++) bars.push(bar(t, 100.5, 102)); // B: low 100.5 ≥ 100 ✓
    for (let t = 60; t < 75; t++) bars.push(bar(t, 101, 103)); // C holds…
    bars.push(bar(75, 100.2, 101)); // …then trades below B's low (100.5)
    for (let t = 76; t < 90; t++) bars.push(bar(t, 101, 102));
    expect(findOneTimeframingBreak(bars, 'up')).toBe(75);
  });

  it('returns the first bar trading above the prior bracket high (down case)', () => {
    const bars: Bar[] = [];
    for (let t = 0; t < 30; t++) bars.push(bar(t, 104, 105)); // A: high 105
    for (let t = 30; t < 60; t++) bars.push(bar(t, 102, 104.5)); // B: high 104.5
    bars.push(bar(60, 103, 104.8)); // C: 104.8 > B's high 104.5 → break
    expect(findOneTimeframingBreak(bars, 'down')).toBe(60);
  });

  it('equal extremes do NOT break control', () => {
    const bars: Bar[] = [];
    for (let t = 0; t < 30; t++) bars.push(bar(t, 100, 101));
    for (let t = 30; t < 60; t++) bars.push(bar(t, 100, 102)); // matches A's low exactly
    expect(findOneTimeframingBreak(bars, 'up')).toBeNull();
  });

  it('returns null while control never breaks', () => {
    const bars: Bar[] = [];
    for (let br = 0; br < 5; br++) {
      for (let t = br * 30; t < br * 30 + 30; t++) {
        bars.push(bar(t, 100 + br, 101.5 + br)); // stair-stepping higher lows
      }
    }
    expect(findOneTimeframingBreak(bars, 'up')).toBeNull();
  });

  it('cannot break inside the first bracket (no prior reference)', () => {
    const bars: Bar[] = [];
    for (let t = 0; t < 30; t++) bars.push(bar(t, 100 - t * 0.1, 101));
    expect(findOneTimeframingBreak(bars, 'up')).toBeNull();
  });
});
