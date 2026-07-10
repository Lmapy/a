/* ============================================================================
   Label-recovery + planted-structure honesty + stylized invariants for the
   session generator (GDD §7 verification loop / §10 test plan).

   These tests run the game's OWN classifiers (core/classify) over generated
   sessions and demand the scripted labels back. Thresholds are the task
   contract (≥90% at easy difficulty); actual measured rates are logged.
   ========================================================================== */

import { describe, expect, it } from 'vitest';
import type { DayType, OpenLocation, OpenType } from '../types';
import { Prng } from './prng';
import { EASY_KNOBS, ROW_STEP, compileScript } from './scripts';
import { GEN_DAYTYPE_OPTS, OPEN_VERIFY_BARS, generateSession, typicalPriorRange } from './generator';
import type { GeneratedSession } from './generator';
import { buildProfile } from '../core/profile';
import { buildTpoProfile } from '../core/tpo';
import { classifyDayType, classifyOpenType, measureExcessTicks } from '../core/classify';
import { OPEN_CHIP, buildLoopItem } from '../drills/items';

/** Canonical compatible (day type × open type × location) pairs (GDD §7). */
const CANONICAL: Array<[DayType, OpenType, OpenLocation]> = [
  ['normal', 'open-auction', 'in-value'],
  ['normal-variation', 'open-test-drive', 'out-of-value-in-range'],
  ['trend', 'open-drive', 'out-of-range'],
  ['double-distribution-trend', 'open-drive', 'out-of-range'],
  ['nontrend', 'open-auction', 'in-value'],
  ['neutral', 'open-rejection-reverse', 'in-value'],
];

function gen(dayType: DayType, openType: OpenType, loc: OpenLocation, seed: string): GeneratedSession {
  const script = compileScript(new Prng(seed), dayType, openType, loc, EASY_KNOBS);
  return generateSession(script);
}

function sessionRange(s: GeneratedSession): number {
  let hi = -Infinity;
  let lo = Infinity;
  for (const b of s.bars) {
    if (b.h > hi) hi = b.h;
    if (b.l < lo) lo = b.l;
  }
  return hi - lo;
}

function priorRange(s: GeneratedSession): number {
  return typicalPriorRange(s.priors);
}

const N = 40;

describe('label recovery (N=40 per day type, easy difficulty)', () => {
  for (const [dayType, openType, loc] of CANONICAL) {
    it(`${dayType} × ${openType}: classifiers recover the scripted labels ≥90%`, () => {
      let dayHits = 0;
      let openHits = 0;
      for (let i = 0; i < N; i++) {
        const s = gen(dayType, openType, loc, `${7000 + i * 13}`);
        const tpo = buildTpoProfile(s.bars, ROW_STEP);
        const measuredDay = classifyDayType(s.bars, tpo, { ...GEN_DAYTYPE_OPTS, typicalRange: priorRange(s) });
        const measuredOpen = classifyOpenType(s.bars.slice(0, 60), s.bars[0].o);
        if (measuredDay === s.labels.dayType) dayHits++;
        if (measuredOpen === s.labels.openType) openHits++;
      }
      // eslint-disable-next-line no-console
      console.log(
        `[recovery] ${dayType.padEnd(26)} day ${((100 * dayHits) / N).toFixed(1)}%  open ${((100 * openHits) / N).toFixed(1)}%`,
      );
      expect(dayHits / N).toBeGreaterThanOrEqual(0.9);
      expect(openHits / N).toBeGreaterThanOrEqual(0.9);
    });
  }
});

describe('served open-type truth ≡ core measurement (GDD §7 final verification)', () => {
  // Escapes of the pre-fix verification loop (re-noise budget exhausted with
  // no final dayOk/openOk check): neutral days whose scripted
  // open-rejection-reverse measured back as TEST-DRIVE or AUCTION. The first
  // three are the audit's repro seeds; the rest fell out of a 400-item sweep.
  const REGRESSION_SEEDS = ['92342', '615177', '1783867', '304', '1647', '2004', '2089', '3228', '4639'];

  const measuredChip = (li: ReturnType<typeof buildLoopItem>): string =>
    OPEN_CHIP[classifyOpenType(li.gen.bars.slice(0, OPEN_VERIFY_BARS), li.gen.bars[0].o)];

  it('the regression seeds serve measurement-consistent truths, deterministically', () => {
    for (const seed of REGRESSION_SEEDS) {
      const li = buildLoopItem('open-type-ladder', seed);
      expect(measuredChip(li), `seed ${seed}`).toBe(li.item.groundTruth);
      // the regenerate/fallback path must stay deterministic: same seed ⇒
      // byte-identical served session and truth
      const again = buildLoopItem('open-type-ladder', seed);
      expect(again.item.groundTruth).toBe(li.item.groundTruth);
      expect(again.gen.bars).toEqual(li.gen.bars);
      expect(again.gen.labels).toEqual(li.gen.labels);
    }
  });

  it('400 sequential open-type-ladder items: served truth ALWAYS equals the measured classification', () => {
    let openMiss = 0;
    let dayMiss = 0;
    for (let i = 0; i < 400; i++) {
      const li = buildLoopItem('open-type-ladder', `${i + 1}`);
      if (measuredChip(li) !== li.item.groundTruth) openMiss++;
      const tpo = buildTpoProfile(li.gen.bars, ROW_STEP);
      const measuredDay = classifyDayType(li.gen.bars, tpo, {
        ...GEN_DAYTYPE_OPTS,
        typicalRange: typicalPriorRange(li.gen.priors),
      });
      if (measuredDay !== li.gen.labels.dayType) dayMiss++;
    }
    // eslint-disable-next-line no-console
    console.log(
      `[recovery] served open-type items: open ${(100 * (1 - openMiss / 400)).toFixed(1)}%  day ${(100 * (1 - dayMiss / 400)).toFixed(1)}% (N=400)`,
    );
    expect(openMiss).toBe(0); // ALWAYS — not a rate: the label is the measurement
    expect(dayMiss).toBe(0);
  }, 120000);
});

describe('determinism (GDD §7)', () => {
  it('same seed ⇒ byte-identical bars, labels, and priors', () => {
    const a = gen('trend', 'open-drive', 'out-of-range', '424242');
    const b = gen('trend', 'open-drive', 'out-of-range', '424242');
    expect(a.bars).toEqual(b.bars);
    expect(a.labels).toEqual(b.labels);
    expect(a.priors).toEqual(b.priors);
  });

  it('different seeds ⇒ different sessions', () => {
    const a = gen('normal', 'open-auction', 'in-value', '1001');
    const b = gen('normal', 'open-auction', 'in-value', '1002');
    expect(a.bars).not.toEqual(b.bars);
  });
});

describe('planted-structure honesty', () => {
  it('nPOC price is never traded before the scripted touch (and touched after when resolved)', () => {
    let checked = 0;
    let resolvedChecked = 0;
    for (const [dayType, openType, loc] of CANONICAL) {
      for (let i = 0; i < 10; i++) {
        const s = gen(dayType, openType, loc, `${9100 + i * 7}`);
        const npoc = s.labels.planted.find((p) => p.kind === 'nPOC');
        if (!npoc) continue;
        checked++;
        const until = npoc.touchBar ?? s.bars.length;
        for (let t = 0; t < until; t++) {
          const b = s.bars[t];
          expect(b.l <= npoc.price && npoc.price <= b.h).toBe(false);
        }
        if (npoc.touchBar !== undefined) {
          resolvedChecked++;
          expect(npoc.resolved).toBe(true);
          const touched = s.bars.some((b) => b.t >= npoc.touchBar! && b.l <= npoc.price && npoc.price <= b.h);
          expect(touched).toBe(true);
        } else {
          expect(npoc.resolved).toBe(false);
        }
      }
    }
    // eslint-disable-next-line no-console
    console.log(`[honesty] nPOC checked in ${checked}/60 sessions (${resolvedChecked} resolved)`);
    expect(checked).toBeGreaterThanOrEqual(50); // the plant must virtually always land
    expect(resolvedChecked).toBeGreaterThanOrEqual(5);
  });

  it('donor prior POC is naked: no later prior session traded through it', () => {
    for (let i = 0; i < 8; i++) {
      const s = gen('trend', 'open-drive', 'out-of-range', `${520 + i * 11}`);
      const npoc = s.labels.planted.find((p) => p.kind === 'nPOC');
      if (!npoc) continue;
      for (const prior of [s.priors[1], s.priors[2]]) {
        for (const b of prior.bars) {
          expect(b.l <= npoc.price && npoc.price <= b.h).toBe(false);
        }
      }
    }
  });

  it('planted poor extremes are measurably flat: excess ≤1 tick and ≥2 shoulder prints', () => {
    let poorSeen = 0;
    for (const [dayType, openType, loc] of CANONICAL) {
      for (let i = 0; i < 10; i++) {
        const s = gen(dayType, openType, loc, `${9100 + i * 7}`);
        const tpo = buildTpoProfile(s.bars, ROW_STEP);
        for (const p of s.labels.planted) {
          if (p.kind !== 'poorHigh' && p.kind !== 'poorLow') continue;
          poorSeen++;
          const side = p.kind === 'poorHigh' ? 'high' : 'low';
          const label = s.labels.extremes.find((e) => e.side === side)!;
          expect(measureExcessTicks(tpo, side)).toBeLessThanOrEqual(1);
          // flat shoulder: ≥2 bars print the shoulder level
          const shoulder =
            side === 'high' ? label.price - label.excessTicks * ROW_STEP : label.price + label.excessTicks * ROW_STEP;
          const prints = s.bars.filter((b) =>
            side === 'high' ? b.h >= shoulder - 1e-9 : b.l <= shoulder + 1e-9,
          ).length;
          expect(prints).toBeGreaterThanOrEqual(2);
        }
      }
    }
    // eslint-disable-next-line no-console
    console.log(`[honesty] poor extremes verified: ${poorSeen}`);
    expect(poorSeen).toBeGreaterThanOrEqual(10);
  });

  it('planted excess extremes measure back in the excess band (never read as poor)', () => {
    let excessSeen = 0;
    for (const [dayType, openType, loc] of CANONICAL) {
      for (let i = 0; i < 10; i++) {
        const s = gen(dayType, openType, loc, `${9100 + i * 7}`);
        const tpo = buildTpoProfile(s.bars, ROW_STEP);
        for (const e of s.labels.extremes) {
          if (e.excessTicks < 4) continue;
          excessSeen++;
          expect(measureExcessTicks(tpo, e.side)).toBeGreaterThanOrEqual(3);
        }
      }
    }
    // eslint-disable-next-line no-console
    console.log(`[honesty] excess extremes verified: ${excessSeen}`);
    expect(excessSeen).toBeGreaterThanOrEqual(10);
  });

  it('DD sessions carry an LVN corridor with measurably thin rows', () => {
    for (let i = 0; i < 8; i++) {
      const s = gen('double-distribution-trend', 'open-drive', 'out-of-range', `${3100 + i * 17}`);
      const lvn = s.labels.planted.find((p) => p.kind === 'lvnCorridor');
      expect(lvn).toBeDefined();
      const [lo, hi] = lvn!.priceRange!;
      const profile = buildProfile(s.bars, ROW_STEP);
      let corridorMax = 0;
      let peak = 0;
      for (let r = 0; r < profile.rows.length; r++) {
        const price = profile.minPrice + r * profile.rowStep;
        if (price >= lo && price <= hi) corridorMax = Math.max(corridorMax, profile.rows[r]);
        peak = Math.max(peak, profile.rows[r]);
      }
      expect(corridorMax).toBeLessThan(0.35 * peak);
    }
  });

  it('open-drive sessions never re-trade the open after the opening bar', () => {
    for (let i = 0; i < 10; i++) {
      const s = gen('trend', 'open-drive', 'out-of-range', `${801 + i * 3}`);
      const open = s.bars[0].o;
      const above = s.bars.slice(1).every((b) => b.l > open);
      const below = s.bars.slice(1).every((b) => b.h < open);
      expect(above || below).toBe(true);
    }
  });
});

describe('stylized invariants (GDD §7 ensemble)', () => {
  it('intraday volume is U-shaped: first+last hour average > middle-hours average', () => {
    for (const [dayType, openType, loc] of CANONICAL) {
      for (let i = 0; i < 5; i++) {
        const s = gen(dayType, openType, loc, `${601 + i * 29}`);
        const avg = (bars: typeof s.bars) => bars.reduce((a, b) => a + b.v, 0) / bars.length;
        const edges = avg([...s.bars.slice(0, 60), ...s.bars.slice(330)]);
        const middle = avg(s.bars.slice(90, 300));
        expect(edges).toBeGreaterThan(middle);
      }
    }
  });

  it('trend-day range exceeds nontrend-day range on every paired seed', () => {
    for (let i = 0; i < 12; i++) {
      const seed = `${1700 + i * 31}`;
      const trend = gen('trend', 'open-drive', 'out-of-range', seed);
      const nontrend = gen('nontrend', 'open-auction', 'in-value', seed);
      expect(sessionRange(trend)).toBeGreaterThan(sessionRange(nontrend));
    }
  });

  it('trend impulses carry more volume than their pullbacks (volume rises with the move)', () => {
    for (let i = 0; i < 6; i++) {
      const s = gen('trend', 'open-drive', 'out-of-range', `${2900 + i * 41}`);
      const segs = s.labels.script;
      const avgSeg = (from: number, to: number) => {
        const bars = s.bars.slice(from, to + 1);
        return bars.reduce((a, b) => a + b.v, 0) / bars.length;
      };
      const impulse = segs.filter((g) => g.regime === 'imbalance' && g.volumeBoost > 1);
      const pullback = segs.filter((g) => g.regime === 'imbalance' && g.volumeBoost < 1);
      expect(impulse.length).toBeGreaterThan(0);
      expect(pullback.length).toBeGreaterThan(0);
      const vi = impulse.reduce((a, g) => a + avgSeg(g.startBar, g.endBar), 0) / impulse.length;
      const vp = pullback.reduce((a, g) => a + avgSeg(g.startBar, g.endBar), 0) / pullback.length;
      expect(vi).toBeGreaterThan(vp);
    }
  });

  it('bars are well-formed and segments tile the session for every day type', () => {
    for (const [dayType, openType, loc] of CANONICAL) {
      const s = gen(dayType, openType, loc, '55555');
      expect(s.bars).toHaveLength(390);
      s.bars.forEach((b, i) => {
        expect(b.t).toBe(i);
        expect(b.h).toBeGreaterThanOrEqual(Math.max(b.o, b.c));
        expect(b.l).toBeLessThanOrEqual(Math.min(b.o, b.c));
        expect(b.v).toBeGreaterThan(0);
      });
      const segs = s.labels.script;
      expect(segs[0].startBar).toBe(0);
      for (let i = 1; i < segs.length; i++) expect(segs[i].startBar).toBe(segs[i - 1].endBar + 1);
      expect(segs[segs.length - 1].endBar).toBe(389);
      expect(s.labels.acceptanceFlags).toHaveLength(13);
      expect(s.priors).toHaveLength(3);
    }
  });
});

describe('all day-type × open-type combinations', () => {
  const DAYS: DayType[] = ['normal', 'normal-variation', 'trend', 'double-distribution-trend', 'nontrend', 'neutral'];
  const OPENS: OpenType[] = ['open-drive', 'open-test-drive', 'open-rejection-reverse', 'open-auction'];

  it('every combo compiles and generates a well-formed session (coercions included)', () => {
    for (const d of DAYS) {
      for (const o of OPENS) {
        const script = compileScript(new Prng('271828'), d, o, 'in-value', EASY_KNOBS);
        expect(script.dayType).toBe(d);
        const s = generateSession(script);
        expect(s.labels.openType).toBe(script.openType); // possibly coerced — labels echo the truth
        expect(s.bars).toHaveLength(390);
        for (const b of s.bars) {
          expect(b.h).toBeGreaterThanOrEqual(Math.max(b.o, b.c));
          expect(b.l).toBeLessThanOrEqual(Math.min(b.o, b.c));
          expect(b.v).toBeGreaterThan(0);
          expect(Number.isFinite(b.o + b.h + b.l + b.c)).toBe(true);
        }
        // drive/test-drive floors hold whatever the day type
        if (script.openType === 'open-drive') {
          const open = s.bars[0].o;
          const above = s.bars.slice(1).every((b) => b.l > open);
          const below = s.bars.slice(1).every((b) => b.h < open);
          expect(above || below).toBe(true);
        }
      }
    }
  });
});

describe('ambiguity blend (difficulty knob)', () => {
  it('blended items label the mixture and still generate cleanly', () => {
    const knobs = { ...EASY_KNOBS, ambiguity: 0.4 };
    const script = compileScript(new Prng('31415'), 'trend', 'open-drive', 'out-of-range', knobs);
    const s = generateSession(script);
    expect(s.labels.blendWeight).toBeCloseTo(0.4);
    expect(s.labels.blendedToward).toBe('double-distribution-trend');
    expect(s.labels.ambiguityWeight).toBeCloseTo(0.4);
    expect(s.bars).toHaveLength(390);
  });
});
