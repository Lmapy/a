import { describe, expect, it } from 'vitest';
import type { Bar, DayType, OpenType, ProfileShape } from '../types';
import { fig02 } from './__fixtures__/fig02';
import { fig05 } from './__fixtures__/fig05';
import { fig06 } from './__fixtures__/fig06';
import { fig09 } from './__fixtures__/fig09';
import { fig10 } from './__fixtures__/fig10';
import { buildProfile, profileFromRows } from './profile';
import { buildTpoProfile } from './tpo';
import {
  classifyDayType,
  classifyExtreme,
  classifyOpenType,
  classifyRegime,
  classifyShape,
  classifyValueRelationship,
  ibWidthFraction,
  isAcceptedOutsideValue,
  measureExcessRows,
  measureExcessTicks,
} from './classify';

function pointBar(t: number, price: number, v = 100): Bar {
  return { t, o: price, h: price, l: price, c: price, v };
}

function rangeBar(t: number, l: number, h: number, c = h, v = 100): Bar {
  return { t, o: l, h, l, c, v };
}

/* ----------------------------------------------------------------------------
   Shape alphabet
   -------------------------------------------------------------------------- */

describe('classifyShape', () => {
  it('hand cases: symmetric bell = D, upper bulge = P, lower bulge = b', () => {
    const bell = profileFromRows([1, 3, 8, 12, 8, 3, 1], 100, 1);
    expect(classifyShape(bell)).toBe('D');
    // dominant bulge in the upper third, thin tail down
    const p = profileFromRows([1, 1, 1, 1, 1, 1, 2, 9, 14, 9, 2], 100, 1);
    expect(classifyShape(p)).toBe('P');
    const b = profileFromRows([2, 9, 14, 9, 2, 1, 1, 1, 1, 1, 1], 100, 1);
    expect(classifyShape(b)).toBe('b');
  });

  it('hand case: two comparable bulges with a thin neck = B', () => {
    const rows = [1, 8, 12, 8, 1, 1, 1, 7, 11, 7, 1];
    expect(classifyShape(profileFromRows(rows, 100, 1))).toBe('B');
  });

  it('hand case: elongated even distribution with POC at an extreme = thin-trend', () => {
    const rows = Array.from({ length: 40 }, (_, i) => 8 + 0.05 * (40 - i));
    rows[2] = 12; // modest bump pins the POC low
    expect(classifyShape(profileFromRows(rows, 100, 1))).toBe('thin-trend');
  });

  it('hand case: a ladder of 3+ comparable bulges = thin-trend, never B (fig-02: B is TWO bulges)', () => {
    // staircase footprint: three comparable consolidation beads with thin traverses
    const ladder = [1, 8, 12, 8, 1, 1, 1, 7, 11, 7, 1, 1, 1, 8, 12, 8, 1];
    expect(classifyShape(profileFromRows(ladder, 100, 1))).toBe('thin-trend');
    // raising the ladder threshold restores the pre-rule reading (option honored)
    expect(classifyShape(profileFromRows(ladder, 100, 1), { ladderMinPeaks: 4 })).toBe('B');
    // exactly two bulges stay B regardless
    const two = [1, 8, 12, 8, 1, 1, 1, 7, 11, 7, 1];
    expect(classifyShape(profileFromRows(two, 100, 1))).toBe('B');
    // a third bulge BELOW the comparable threshold does not trip the ladder
    const twoAndDwarf = [1, 8, 12, 8, 1, 1, 1, 7, 11, 7, 1, 1, 1, 2, 3, 2, 1];
    expect(classifyShape(profileFromRows(twoAndDwarf, 100, 1))).toBe('B');
  });

  it('golden parity — fig-02: recovers all five reference shapes', () => {
    for (const ref of fig02.profiles) {
      const profile = profileFromRows([...ref.rows], 0, 1);
      expect(classifyShape(profile), ref.name).toBe(ref.shape as ProfileShape);
    }
  });

  it('golden parity — fig-02: the B profile carries an LVN neck at the reference row', () => {
    const bRef = fig02.profiles[3];
    const profile = profileFromRows([...bRef.rows], 0, 1);
    expect(profile.lvnRanges.length).toBeGreaterThanOrEqual(1);
    expect(profile.lvnRanges[0].lo).toBeLessThanOrEqual(fig02.bNeckIndex);
    expect(profile.lvnRanges[0].hi).toBeGreaterThanOrEqual(fig02.bNeckIndex);
  });
});

/* ----------------------------------------------------------------------------
   Day types
   -------------------------------------------------------------------------- */

function panelBars(path: readonly number[], vols: readonly number[]): Bar[] {
  return path.map((p, i) => pointBar(i, p, vols[i]));
}

describe('classifyDayType', () => {
  it('hand case: range extension both sides = neutral', () => {
    const bars: Bar[] = [];
    for (let t = 0; t < 60; t++) bars.push(rangeBar(t, 99, 101, 100));
    for (let t = 60; t < 90; t++) bars.push(rangeBar(t, 100, 102.5, 101)); // above IB
    for (let t = 90; t < 120; t++) bars.push(rangeBar(t, 97.5, 100, 99)); // below IB
    const tpo = buildTpoProfile(bars, 0.5);
    expect(classifyDayType(bars, tpo)).toBe('neutral');
  });

  it('hand case: narrow IB, one-sided extension, close at the extreme = trend', () => {
    const bars: Bar[] = [];
    for (let t = 0; t < 60; t++) bars.push(rangeBar(t, 100, 101, 100.5));
    for (let t = 60; t < 390; t++) {
      const drift = ((t - 60) / 330) * 8;
      bars.push(rangeBar(t, 100.5 + drift, 101.5 + drift, 101.4 + drift));
    }
    const tpo = buildTpoProfile(bars, 0.25);
    expect(classifyDayType(bars, tpo)).toBe('trend');
  });

  it('hand case: no extension = normal; tiny range vs typicalRange = nontrend', () => {
    const bars: Bar[] = [];
    for (let t = 0; t < 120; t++) bars.push(rangeBar(t, 99, 101, 100));
    const tpo = buildTpoProfile(bars, 0.5);
    expect(classifyDayType(bars, tpo)).toBe('normal');
    expect(classifyDayType(bars, tpo, { typicalRange: 10 })).toBe('nontrend');
  });

  it('golden parity — fig-05: recovers all six reference day types', () => {
    const ranges = fig05.panels.map((p) => {
      const path = p.path;
      return Math.max(...path) - Math.min(...path);
    });
    const typicalRange = ranges.slice().sort((a, b) => a - b)[3]; // upper median
    for (const panel of fig05.panels) {
      const bars = panelBars(panel.path, panel.vols);
      const tpo = buildTpoProfile(bars, fig05.binw);
      const got = classifyDayType(bars, tpo, { typicalRange });
      expect(got, panel.name).toBe(panel.dayType as DayType);
    }
  });
});

describe('ibWidthFraction', () => {
  it('is IB range over session range', () => {
    const bars: Bar[] = [];
    for (let t = 0; t < 60; t++) bars.push(rangeBar(t, 100, 102, 101));
    for (let t = 60; t < 90; t++) bars.push(rangeBar(t, 102, 108, 107));
    expect(ibWidthFraction(bars)).toBeCloseTo(2 / 8, 12);
  });

  it('golden sanity — fig-05: trend day has the narrowest IB, normal the widest', () => {
    const fracs = fig05.panels.map((p) => ({
      name: p.name,
      frac: ibWidthFraction(panelBars(p.path, p.vols), fig05.ibMin),
    }));
    const byFrac = fracs.slice().sort((a, b) => a.frac - b.frac);
    expect(byFrac[0].name).toBe('Trend day');
    expect(byFrac[byFrac.length - 1].name).toBe('Normal day');
  });
});

/* ----------------------------------------------------------------------------
   Open types
   -------------------------------------------------------------------------- */

describe('classifyOpenType', () => {
  it('hand case: never re-trades the open = open-drive', () => {
    const bars = [pointBar(0, 100)];
    for (let t = 1; t < 60; t++) bars.push(pointBar(t, 100.5 + t * 0.05));
    expect(classifyOpenType(bars, 100)).toBe('open-drive');
  });

  it('hand case: early failed probe then a held reversal = open-test-drive', () => {
    const bars = [pointBar(0, 100)];
    for (let t = 1; t <= 10; t++) bars.push(pointBar(t, 100 - t * 0.2)); // probe to 98
    for (let t = 11; t < 90; t++) bars.push(pointBar(t, 98 + (t - 10) * 0.08)); // drive up
    expect(classifyOpenType(bars, 100)).toBe('open-test-drive');
  });

  it('hand case: late failed drive reversing through the open = open-rejection-reverse', () => {
    const bars = [pointBar(0, 100)];
    for (let t = 1; t <= 30; t++) bars.push(pointBar(t, 100 + t * 0.08)); // drive to 102.4
    for (let t = 31; t < 90; t++) bars.push(pointBar(t, 102.4 - (t - 30) * 0.08)); // fail down
    expect(classifyOpenType(bars, 100)).toBe('open-rejection-reverse');
  });

  it('hand case: rotation across the open = open-auction', () => {
    const bars: Bar[] = [];
    for (let t = 0; t < 90; t++) bars.push(pointBar(t, 100 + Math.sin(t / 5)));
    expect(classifyOpenType(bars, 100)).toBe('open-auction');
  });

  it('golden parity — fig-06: recovers all four reference open types', () => {
    const cases: [readonly number[], OpenType][] = [
      [fig06.paths.openDrive, 'open-drive'],
      [fig06.paths.openTestDrive, 'open-test-drive'],
      [fig06.paths.openRejectionReverse, 'open-rejection-reverse'],
      [fig06.paths.openAuction, 'open-auction'],
    ];
    for (const [path, want] of cases) {
      const bars = path.map((p, i) => pointBar(i, p));
      expect(classifyOpenType(bars, fig06.open), want).toBe(want);
    }
  });
});

/* ----------------------------------------------------------------------------
   Excess vs poor extremes
   -------------------------------------------------------------------------- */

describe('measureExcessTicks (TPO single-print tail)', () => {
  // Bracket A spikes to 110 and comes back; B..E rotate 100..104.
  const bars: Bar[] = [rangeBar(0, 100, 110, 104)];
  for (let t = 1; t < 30; t++) bars.push(rangeBar(t, 100, 104, 102));
  for (let t = 30; t < 150; t++) bars.push(rangeBar(t, 100, 104, 102));

  it('counts the single-print run at the extreme', () => {
    const tpo = buildTpoProfile(bars, 1);
    // rows 105..110 are A-only single prints → 6 ticks of excess at the high
    expect(measureExcessTicks(tpo, 'high')).toBe(6);
    // the low (100) is traded by every bracket → poor, 0 ticks
    expect(measureExcessTicks(tpo, 'low')).toBe(0);
  });
});

describe('measureExcessRows / classifyExtreme (fig-09 volume semantics)', () => {
  it('hand case: thin taper = excess, flat substantial top = poor', () => {
    const excess = [2, 10, 20, 14, 6, 2, 1.5, 1, 0.5]; // taper over the top 4 rows (6 > 0.25×20)
    expect(measureExcessRows(excess, 'high')).toBe(4);
    expect(classifyExtreme(excess, 'high')).toBe('excess');
    const poor = [2, 10, 20, 16, 14, 13, 13, 13, 13]; // squared-off top
    expect(measureExcessRows(poor, 'high')).toBe(0);
    expect(classifyExtreme(poor, 'high')).toBe('poor');
  });

  it('the 2–3 row band is a judgment call (GDD Drill B)', () => {
    const ambiguous = [2, 10, 20, 14, 8, 3, 2];
    expect(measureExcessRows(ambiguous, 'high')).toBe(2);
    expect(classifyExtreme(ambiguous, 'high')).toBe('judgment');
  });

  it('works on the low side too', () => {
    const excessLow = [0.5, 1, 1.5, 2, 6, 14, 20, 10, 2];
    expect(measureExcessRows(excessLow, 'low')).toBe(4);
    expect(classifyExtreme(excessLow, 'low')).toBe('excess');
  });

  it('golden parity — fig-09: excess high vs poor high', () => {
    const excessTail = measureExcessRows([...fig09.excessHigh], 'high');
    expect(excessTail).toBeGreaterThanOrEqual(4); // genuine excess band
    expect(classifyExtreme([...fig09.excessHigh], 'high')).toBe('excess');
    expect(classifyExtreme([...fig09.poorHigh], 'high')).toBe('poor');
    // the taper the reference script enforces over its top rows is recovered
    expect(excessTail).toBe(fig09.tailRows);
  });
});

/* ----------------------------------------------------------------------------
   Acceptance outside value
   -------------------------------------------------------------------------- */

describe('isAcceptedOutsideValue', () => {
  const priorVah = 101;
  const priorVal = 99;

  function acceptanceBars(): Bar[] {
    const bars: Bar[] = [];
    for (let t = 0; t < 30; t++) bars.push(rangeBar(t, 99.5, 100.5, 100)); // inside value
    for (let t = 30; t < 60; t++) bars.push(rangeBar(t, 101, 102, 101.6)); // value outside
    for (let t = 60; t < 90; t++) bars.push(rangeBar(t, 101.5, 102.5, 102.2)); // and again
    return bars;
  }

  it('two consecutive brackets outside + dPOC migration = accepted', () => {
    expect(isAcceptedOutsideValue(acceptanceBars(), priorVah, priorVal, 2)).toBe(true);
  });

  it('only one bracket outside is NOT yet acceptance', () => {
    expect(isAcceptedOutsideValue(acceptanceBars(), priorVah, priorVal, 1)).toBe(false);
  });

  it('a poke that closes back inside value is rejection', () => {
    const bars: Bar[] = [];
    for (let t = 0; t < 30; t++) bars.push(rangeBar(t, 99.5, 100.5, 100));
    for (let t = 30; t < 60; t++) bars.push(rangeBar(t, 100, 101.8, 100.6)); // pokes, closes in
    for (let t = 60; t < 90; t++) bars.push(rangeBar(t, 99.5, 100.8, 100.2));
    expect(isAcceptedOutsideValue(bars, priorVah, priorVal, 2)).toBe(false);
  });

  it('acceptance below value works symmetrically', () => {
    const bars: Bar[] = [];
    for (let t = 0; t < 30; t++) bars.push(rangeBar(t, 99.5, 100.5, 100));
    for (let t = 30; t < 60; t++) bars.push(rangeBar(t, 97.8, 99, 98.2, 100));
    for (let t = 60; t < 90; t++) bars.push(rangeBar(t, 97.2, 98.4, 97.6, 100));
    expect(isAcceptedOutsideValue(bars, priorVah, priorVal, 2)).toBe(true);
  });

  it('bracket 0 can never be accepted (needs two periods)', () => {
    const bars = acceptanceBars();
    expect(isAcceptedOutsideValue(bars, priorVah, priorVal, 0)).toBe(false);
  });
});

/* ----------------------------------------------------------------------------
   Regime
   -------------------------------------------------------------------------- */

describe('classifyRegime', () => {
  it('hand case: sustained one-timeframing drift = imbalance', () => {
    const bars: Bar[] = [];
    for (let t = 0; t < 180; t++) {
      const drift = t * 0.03;
      bars.push(rangeBar(t, 100 + drift, 100.6 + drift, 100.5 + drift));
    }
    expect(classifyRegime(bars, 179)).toBe('imbalance');
  });

  it('hand case: rotation around a fixed level = balance', () => {
    const bars: Bar[] = [];
    for (let t = 0; t < 180; t++) {
      const wob = Math.sin(t / 7) * 1.2;
      bars.push(rangeBar(t, 99.4 + wob, 100.6 + wob, 100 + wob));
    }
    expect(classifyRegime(bars, 179)).toBe('balance');
  });

  it('golden sanity — fig-05: trend day reads imbalance, normal day reads balance', () => {
    const trend = fig05.panels.find((p) => p.name === 'Trend day')!;
    const normal = fig05.panels.find((p) => p.name === 'Normal day')!;
    const trendBars = panelBars(trend.path, trend.vols);
    const normalBars = panelBars(normal.path, normal.vols);
    expect(classifyRegime(trendBars, trendBars.length - 1)).toBe('imbalance');
    expect(classifyRegime(normalBars, normalBars.length - 1)).toBe('balance');
  });
});

/* ----------------------------------------------------------------------------
   Value relationships
   -------------------------------------------------------------------------- */

describe('classifyValueRelationship', () => {
  it('hand cases for every relationship', () => {
    const prior = { val: 99, vah: 101 };
    expect(classifyValueRelationship({ val: 102, vah: 104 }, prior)).toBe('higher');
    expect(classifyValueRelationship({ val: 96, vah: 98 }, prior)).toBe('lower');
    expect(classifyValueRelationship({ val: 99.2, vah: 100.8 }, prior)).toBe('inside');
    expect(classifyValueRelationship({ val: 98, vah: 102 }, prior)).toBe('outside');
    expect(classifyValueRelationship({ val: 99.1, vah: 101.1 }, prior)).toBe('overlapping');
    // migrated with little overlap (0.5 of a 2-wide band) = higher, not overlapping
    expect(classifyValueRelationship({ val: 100.5, vah: 102.5 }, prior)).toBe('higher');
  });

  it('golden parity — fig-10: the six-day staircase reads as the reference relationships', () => {
    for (let i = 1; i < fig10.days.length; i++) {
      const got = classifyValueRelationship(fig10.days[i], fig10.days[i - 1]);
      expect(got, `D${i + 1} vs D${i}`).toBe(fig10.relationships[i - 1]);
    }
  });
});

/* ----------------------------------------------------------------------------
   Contract smoke (all classifiers return taxonomy members on generic input)
   -------------------------------------------------------------------------- */

describe('classifier taxonomy membership (contract smoke)', () => {
  const bars: Bar[] = Array.from({ length: 60 }, (_, t) => ({
    t,
    o: 100,
    h: 101,
    l: 99,
    c: 100.5,
    v: 100,
  }));

  it('every classifier stays inside its taxonomy on degenerate input', () => {
    const p = buildProfile(bars, 0.5);
    expect(['D', 'P', 'b', 'B', 'thin-trend']).toContain(classifyShape(p));
    const tpo = buildTpoProfile(bars, 0.5);
    expect([
      'normal',
      'normal-variation',
      'trend',
      'double-distribution-trend',
      'nontrend',
      'neutral',
    ]).toContain(classifyDayType(bars, tpo));
    expect([
      'open-drive',
      'open-test-drive',
      'open-rejection-reverse',
      'open-auction',
    ]).toContain(classifyOpenType(bars, 100));
    expect(measureExcessTicks(tpo, 'high')).toBeGreaterThanOrEqual(0);
    expect(typeof isAcceptedOutsideValue(bars, 101, 99, 1)).toBe('boolean');
    expect(['balance', 'imbalance']).toContain(classifyRegime(bars, 30));
  });
});
