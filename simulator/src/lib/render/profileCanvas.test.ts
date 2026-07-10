/* Layout math of the canvas renderer — the pure part (mockup geometry).
   The pixel path itself is exercised in e2e (real chromium canvas, where the
   <8ms redraw budget is asserted against window.__auction.lastRenderMs). */
import { describe, expect, it } from 'vitest';
import {
  AXIS_W,
  BAR_X,
  PAD_B,
  PAD_T,
  lineLayout,
  lineX,
  lineY,
  priceAxisRows,
  profileLayout,
  rowAtY,
  rowCenterY,
  rowY,
  screenRow,
} from './profileCanvas';
import type { Bar } from '../types';

describe('profileLayout (mockup geometry)', () => {
  it('matches the mockup formulas at the mockup size', () => {
    // mockup: 40 rows in a ~358×420 chart body
    const L = profileLayout(40, 358, 420);
    expect(L.rowH).toBeCloseTo((420 - PAD_T - PAD_B) / 40, 12);
    expect(L.barH).toBeCloseTo(Math.max(5, L.rowH - 3.5), 12);
    expect(L.maxLen).toBe(358 - AXIS_W - BAR_X - 12);
    expect(L.bandR).toBe(358 - AXIS_W + 8);
  });

  it('screenRow is an involution (top row = last profile row)', () => {
    for (const n of [1, 2, 37, 120]) {
      for (let r = 0; r < n; r++) {
        expect(screenRow(screenRow(r, n), n)).toBe(r);
      }
      expect(screenRow(n - 1, n)).toBe(0); // highest price renders on top
    }
  });

  it('rowY is monotone down the screen and spans the padded area', () => {
    const L = profileLayout(60, 390, 400);
    expect(rowY(L, 0)).toBe(PAD_T);
    expect(rowY(L, 60)).toBeCloseTo(400 - PAD_B, 10);
    for (let i = 1; i <= 60; i++) expect(rowY(L, i)).toBeGreaterThan(rowY(L, i - 1));
  });

  it('rowAtY inverts rowCenterY exactly for every row (snap-to-row, GDD §10-A)', () => {
    for (const [n, w, h] of [
      [40, 358, 420],
      [113, 390, 300],
      [7, 200, 500],
    ] as const) {
      const L = profileLayout(n, w, h);
      for (let r = 0; r < n; r++) {
        expect(rowAtY(L, rowCenterY(L, r))).toBe(r);
      }
    }
  });

  it('rowAtY snaps within the tolerance and clamps at the edges', () => {
    const L = profileLayout(20, 358, 420);
    const c = rowCenterY(L, 10);
    expect(rowAtY(L, c + L.rowH * 0.49)).toBe(10); // same slot until the boundary
    expect(rowAtY(L, c + L.rowH * 0.51)).toBe(9); // next slot down = lower price row
    expect(rowAtY(L, 5)).toBe(19); // top padding within 22px snaps to the top row
    expect(rowAtY(L, 415)).toBe(0); // bottom padding snaps to the lowest row
    expect(rowAtY(L, 0)).toBeNull(); // beyond the 22px snap tolerance
  });

  it('priceAxisRows labels every 4th row and skips the POC callout zone', () => {
    const rows = priceAxisRows(40, 27);
    expect(rows).toContain(0);
    expect(rows).toContain(4);
    // 25/26/27/28/29 are within |i − 27| < 3 → 28 (a multiple of 4) skipped
    expect(rows).not.toContain(28);
    for (const i of rows) expect(i % 4).toBe(0);
    // without a POC, nothing is skipped
    expect(priceAxisRows(40, null)).toContain(28);
  });
});

describe('lineLayout (opening-line chart)', () => {
  const bars: Bar[] = Array.from({ length: 90 }, (_, t) => ({
    t,
    o: 5000 + t * 0.1,
    h: 5001 + t * 0.1,
    l: 4999 + t * 0.1,
    c: 5000.5 + t * 0.1,
    v: 100,
  }));

  it('pads the price span and maps monotonically', () => {
    const L = lineLayout(bars, 358, 300);
    expect(L.pMin).toBeLessThan(4999);
    expect(L.pMax).toBeGreaterThan(5001 + 8.9);
    expect(lineX(L, 0, 90)).toBeLessThan(lineX(L, 89, 90));
    expect(lineY(L, L.pMin)).toBeGreaterThan(lineY(L, L.pMax)); // higher price = higher on screen
    // plot fits inside the canvas
    expect(lineX(L, 89, 90)).toBeLessThanOrEqual(358 - AXIS_W);
    expect(lineY(L, L.pMax)).toBeGreaterThanOrEqual(0);
    expect(lineY(L, L.pMin)).toBeLessThanOrEqual(300);
  });
});
