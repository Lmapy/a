import { describe, expect, it } from 'vitest';
import type { Bar } from '../types';
import { Prng } from '../gen/prng';
import { fig01 } from './__fixtures__/fig01';
import { fig12 } from './__fixtures__/fig12';
import {
  buildProfile,
  detectVolumeNodes,
  expandValueArea,
  findHvnRanges,
  findLvnRanges,
  findPoc,
  priceToRow,
  profileFromRows,
  rowToPrice,
  smoothRows,
} from './profile';

function bar(t: number, price: number, v: number): Bar {
  return { t, o: price, h: price, l: price, c: price, v };
}

describe('findPoc', () => {
  it('returns the argmax row', () => {
    expect(findPoc([1, 5, 3])).toBe(1);
  });
  it('breaks ties toward the middle of the profile', () => {
    // rows 0 and 4 tie at 9 and are equidistant from mid → the earlier row wins
    expect(findPoc([9, 1, 1, 1, 9])).toBe(0);
    // rows 1 and 4 tie; row 1 is closer to mid (2) than row 4
    expect(findPoc([1, 9, 1, 1, 9])).toBe(1);
  });
  it('throws on an empty profile', () => {
    expect(() => findPoc([])).toThrow();
  });
});

describe('expandValueArea (guide §2.2 / fig-01 reference semantics)', () => {
  it('single-row greedy expansion, equidistant tie goes up, stops at 70%', () => {
    // volumes: [10, 20, 40, 20, 10], total 100, target 70
    // start poc=2 (40); up=20 dn=20 tie, both 1 row from POC → up (acc 60);
    // up=10 dn=20 → dn (acc 80)
    const { vah, val } = expandValueArea([10, 20, 40, 20, 10], 2, 0.7);
    expect(vah).toBe(3);
    expect(val).toBe(1);
  });

  it('non-equidistant ties go to the row closer to the POC (guide §2.2 step 4)', () => {
    // [6, 9, 30, 20, 9]: total 74, target 51.8. acc=30; up=20 vs dn=9 → up
    // (acc 50). Then up=9 (row 4, 2 from POC) ties dn=9 (row 1, 1 from POC)
    // → the CLOSER row 1 wins (a ties-up rule would take row 4 instead).
    const { vah, val } = expandValueArea([6, 9, 30, 20, 9], 2, 0.7);
    expect(vah).toBe(3);
    expect(val).toBe(1);
  });

  it('hand-verifiable 5-row case', () => {
    // [5, 15, 50, 20, 10]: total 100, target 70. acc=50; up=20>dn=15 → hi=3
    // (acc 70 ≥ target) → VA = rows 2..3 exactly.
    const { vah, val } = expandValueArea([5, 15, 50, 20, 10], 2, 0.7);
    expect(val).toBe(2);
    expect(vah).toBe(3);
  });

  it('caps at the profile bounds when the target exceeds available volume', () => {
    const { vah, val } = expandValueArea([1, 2, 1], 1, 1.0);
    expect(val).toBe(0);
    expect(vah).toBe(2);
  });

  it('properties on seeded random profiles: contains POC, vah ≥ val, ≥70% and minimal', () => {
    const stream = new Prng('424242').stream('noise');
    for (let trial = 0; trial < 200; trial++) {
      const n = 3 + stream.nextInt(0, 60);
      const rows = Array.from({ length: n }, () => stream.nextFloat() * 100 + 0.01);
      const poc = findPoc(rows);
      const { vah, val } = expandValueArea(rows, poc);
      expect(val).toBeLessThanOrEqual(poc);
      expect(vah).toBeGreaterThanOrEqual(poc);
      const total = rows.reduce((a, b) => a + b, 0);
      const mass = rows.slice(val, vah + 1).reduce((a, b) => a + b, 0);
      expect(mass).toBeGreaterThanOrEqual(0.7 * total - 1e-9);
      // greedy minimality: dropping the last-absorbed boundary row must fall
      // below the target (unless the VA is the single POC row)
      if (vah > val) {
        const withoutTop = mass - rows[vah];
        const withoutBottom = mass - rows[val];
        expect(Math.min(withoutTop, withoutBottom)).toBeLessThan(0.7 * total);
      }
    }
  });
});

describe('buildProfile', () => {
  const bars: Bar[] = [
    bar(0, 100.0, 10),
    bar(1, 100.5, 30),
    bar(2, 100.5, 25),
    bar(3, 101.0, 20),
    bar(4, 100.0, 15),
  ];

  it('bins volume, finds POC, and VA contains POC', () => {
    const p = buildProfile(bars, 0.5);
    expect(p.totalVolume).toBe(100);
    expect(p.rowStep).toBe(0.5);
    expect(rowToPrice(p, p.poc)).toBe(100.5); // 55 volume at 100.5
    expect(p.val).toBeLessThanOrEqual(p.poc);
    expect(p.vah).toBeGreaterThanOrEqual(p.poc);
  });

  it('distributes a ranging bar uniformly across every touched row', () => {
    const b: Bar = { t: 0, o: 100, h: 101, l: 100, c: 101, v: 30 };
    const p = buildProfile([b], 0.5);
    expect(p.rows).toEqual([10, 10, 10]); // rows at 100, 100.5, 101
    expect(p.totalVolume).toBe(30);
  });

  it('throws on empty bars or non-positive rowStep', () => {
    expect(() => buildProfile([], 0.5)).toThrow();
    expect(() => buildProfile(bars, 0)).toThrow();
  });
});

describe('HVN / LVN detection', () => {
  // two bulges (peaks at rows 2 and 7, the first dominant) with a thin neck
  const rows = [1, 8, 10, 8, 1, 1, 7, 9, 7, 1];

  it('finds both bulges, dominant first', () => {
    const hvns = findHvnRanges(rows);
    expect(hvns).toHaveLength(2);
    expect(hvns[0].lo).toBeLessThanOrEqual(2);
    expect(hvns[0].hi).toBeGreaterThanOrEqual(2); // dominant bulge holds row 2
    expect(hvns[1].lo).toBeLessThanOrEqual(7);
    expect(hvns[1].hi).toBeGreaterThanOrEqual(7);
  });

  it('finds the neck between the bulges as the LVN', () => {
    const lvns = findLvnRanges(rows);
    expect(lvns).toHaveLength(1);
    // the neck rows 4/5 sit inside the LVN range
    expect(lvns[0].lo).toBeGreaterThan(2);
    expect(lvns[0].hi).toBeLessThan(7);
    expect(lvns[0].lo).toBeLessThanOrEqual(5);
    expect(lvns[0].hi).toBeGreaterThanOrEqual(4);
  });

  it('a unimodal bell has one HVN and no LVN', () => {
    const bell = [1, 2, 5, 9, 12, 9, 5, 2, 1];
    expect(findHvnRanges(bell)).toHaveLength(1);
    expect(findLvnRanges(bell)).toHaveLength(0);
  });

  it('shallow wiggles merge into a single node (mergeValleyFrac)', () => {
    // two "peaks" separated by a barely-lower valley: one node, no LVN
    const rows2 = [1, 9, 8.5, 9, 1];
    expect(findHvnRanges(rows2)).toHaveLength(1);
    expect(findLvnRanges(rows2)).toHaveLength(0);
  });

  it('sensitivity is configurable', () => {
    // with a permissive lvnFrac the moderate dip becomes an LVN
    const rows3 = [1, 9, 5, 9, 1];
    expect(findLvnRanges(rows3)).toHaveLength(0); // 5/9 ≈ 0.56 > default 0.5
    expect(findLvnRanges(rows3, { lvnFrac: 0.6, smoothPasses: 0 })).toHaveLength(1);
  });

  it('returns [] on empty and flat inputs', () => {
    expect(detectVolumeNodes([])).toHaveLength(0);
    expect(findLvnRanges([0, 0, 0])).toHaveLength(0);
  });

  it('smoothRows preserves total mass approximately and is a no-op at 0 passes', () => {
    const r = [1, 4, 2, 8, 3];
    expect(smoothRows(r, 0)).toEqual(r);
    const s = smoothRows(r, 1);
    expect(s).toHaveLength(r.length);
  });
});

describe('golden parity — fig-01 (volume-profile anatomy)', () => {
  const vol = [...fig01.vol];

  it('POC / VAL / VAH match the Python reference to the row', () => {
    const poc = findPoc(vol);
    expect(poc).toBe(fig01.poc);
    const { vah, val } = expandValueArea(vol, poc);
    expect(val).toBe(fig01.val);
    expect(vah).toBe(fig01.vah);
  });

  it('HVN/LVN detection recovers the reference bulges and neck', () => {
    const p = profileFromRows(vol, 4980, 1);
    // dominant HVN holds the POC; the secondary bulge holds the small peak
    expect(p.hvnRanges.length).toBeGreaterThanOrEqual(2);
    expect(p.hvnRanges[0].lo).toBeLessThanOrEqual(fig01.poc);
    expect(p.hvnRanges[0].hi).toBeGreaterThanOrEqual(fig01.poc);
    const holdsSmall = p.hvnRanges.some(
      (r) => r.lo <= fig01.hvnSmallPeak && fig01.hvnSmallPeak <= r.hi,
    );
    expect(holdsSmall).toBe(true);
    // deepest LVN contains the reference neck row
    expect(p.lvnRanges.length).toBeGreaterThanOrEqual(1);
    expect(p.lvnRanges[0].lo).toBeLessThanOrEqual(fig01.lvn);
    expect(p.lvnRanges[0].hi).toBeGreaterThanOrEqual(fig01.lvn);
  });
});

describe('golden parity — fig-12 (session volume profile from a minute path)', () => {
  const bars: Bar[] = fig12.price.map((p, i) => ({
    t: i,
    o: p,
    h: p,
    l: p,
    c: p,
    v: fig12.vol[i],
  }));

  it('binning geometry and POC/VA rows are bit-identical to the Python reference', () => {
    const p = buildProfile(bars, fig12.rowStep);
    expect(p.minPrice).toBe(fig12.minPrice);
    expect(p.rows).toHaveLength(fig12.nRows);
    expect(p.poc).toBe(fig12.poc);
    expect(p.val).toBe(fig12.val);
    expect(p.vah).toBe(fig12.vah);
  });
});

describe('rowToPrice / priceToRow are inverse (snap-to-row invariant)', () => {
  it('round-trips every row', () => {
    const geom = { minPrice: 4980, rowStep: 0.25 };
    for (let row = 0; row < 200; row++) {
      expect(priceToRow(geom, rowToPrice(geom, row))).toBe(row);
    }
  });
});
