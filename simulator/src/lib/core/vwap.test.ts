import { describe, expect, it } from 'vitest';
import type { Bar } from '../types';
import { fig12 } from './__fixtures__/fig12';
import { computeVwap, typicalPrice } from './vwap';

function bar(t: number, p: number, v: number): Bar {
  return { t, o: p, h: p, l: p, c: p, v };
}

describe('typicalPrice', () => {
  it('is (h + l + c) / 3', () => {
    expect(typicalPrice({ t: 0, o: 1, h: 3, l: 1, c: 2, v: 5 })).toBe(2);
  });
});

describe('computeVwap', () => {
  it('matches the hand-computed volume-weighted mean', () => {
    // bars at 100 (v=1) and 200 (v=3): vwap = (100 + 600) / 4 = 175
    const series = computeVwap([bar(0, 100, 1), bar(1, 200, 3)]);
    expect(series.vwap[0]).toBe(100);
    expect(series.vwap[1]).toBe(175);
  });

  it('bands are symmetric around vwap and ordered', () => {
    const bars = [bar(0, 100, 2), bar(1, 102, 1), bar(2, 98, 3), bar(3, 101, 2)];
    const s = computeVwap(bars);
    for (let i = 0; i < bars.length; i++) {
      expect(s.upper1[i] - s.vwap[i]).toBeCloseTo(s.vwap[i] - s.lower1[i], 10);
      expect(s.upper2[i] - s.vwap[i]).toBeCloseTo(2 * (s.upper1[i] - s.vwap[i]), 10);
      expect(s.upper2[i]).toBeGreaterThanOrEqual(s.upper1[i]);
      expect(s.lower2[i]).toBeLessThanOrEqual(s.lower1[i]);
    }
  });

  it('zero-dispersion input collapses the bands onto the vwap', () => {
    const s = computeVwap([bar(0, 100, 1), bar(1, 100, 1)]);
    expect(s.upper2[1]).toBe(100);
    expect(s.lower2[1]).toBe(100);
  });

  it('returns empty series for empty input', () => {
    const s = computeVwap([]);
    expect(s.vwap).toHaveLength(0);
  });
});

describe('golden parity — fig-12 (running VWAP + σ bands over a session)', () => {
  it('end-of-session VWAP and σ match the Python reference', () => {
    const bars: Bar[] = fig12.price.map((p, i) => ({
      t: i,
      o: p,
      h: p,
      l: p,
      c: p,
      v: fig12.vol[i],
    }));
    const s = computeVwap(bars);
    const last = bars.length - 1;
    expect(s.vwap[last]).toBeCloseTo(fig12.vwapEnd, 9);
    expect(s.upper1[last] - s.vwap[last]).toBeCloseTo(fig12.sdEnd, 9);
    expect(s.lower2[last]).toBeCloseTo(fig12.vwapEnd - 2 * fig12.sdEnd, 9);
  });
});
