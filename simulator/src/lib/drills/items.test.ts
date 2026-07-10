/* Item builders: determinism + ground-truth honesty per drill (GDD §4/§7). */
import { describe, expect, it } from 'vitest';
import { siblingSeed } from '../gen/prng';
import { EASY_KNOBS } from '../gen/scripts';
import { buildProfile } from '../core/profile';
import { classifyShape } from '../core/classify';
import {
  LOOP_DRILLS,
  OPEN_CHIP,
  OPEN_REVEAL_BARS,
  PAR_MS,
  REGIME_CHIPS,
  barClock,
  buildLoopItem,
  excessClassOf,
  renderFlags,
} from './items';
import type { LoopDrillId } from './items';

const seeds = (base: string, n: number): string[] =>
  Array.from({ length: n }, (_, i) => siblingSeed(base, i));

describe('buildLoopItem — shared contract', () => {
  it('is deterministic: same (drill, seed, knobs) ⇒ identical item + ctx', () => {
    for (const drill of LOOP_DRILLS) {
      const a = buildLoopItem(drill, '987654321', EASY_KNOBS);
      const b = buildLoopItem(drill, '987654321', EASY_KNOBS);
      expect(JSON.stringify(a.item)).toBe(JSON.stringify(b.item));
      expect(JSON.stringify(a.ctx)).toBe(JSON.stringify(b.ctx));
      expect(JSON.stringify(a.gen.bars)).toBe(JSON.stringify(b.gen.bars));
    }
  });

  it('carries engine-derived metadata: id, par, knob-derived rating', () => {
    for (const drill of LOOP_DRILLS) {
      const { item } = buildLoopItem(drill, '13371337', EASY_KNOBS);
      expect(item.id).toBe(`${drill}:13371337:${item.id.split(':')[2]}`);
      expect(item.drillId).toBe(drill);
      expect(item.parMs).toBe(PAR_MS[drill]);
      expect(item.itemRating).toBeGreaterThanOrEqual(800);
      expect(item.itemRating).toBeLessThanOrEqual(2400);
      expect(item.question.length).toBeGreaterThan(0);
      expect(item.paramsVersion.length).toBeGreaterThan(0);
    }
  });
});

describe('poc-va-snap (Drill A)', () => {
  it('ground truth is the rendered profile’s own POC/VAH/VAL row', () => {
    for (const s of seeds('11110000', 6)) {
      const { item, ctx } = buildLoopItem('poc-va-snap', s, EASY_KNOBS);
      const p = item.stimulus.profile;
      const expected = ctx.target === 'POC' ? p.poc : ctx.target === 'VAH' ? p.vah : p.val;
      expect(item.groundTruth).toBe(expected);
      expect(item.choices).toEqual([]);
      expect(item.question).toBe(ctx.target === 'POC' ? 'Tap the POC' : `Place ${ctx.target}`);
      expect(ctx.expansionRounds).toBe(p.vah - p.val);
    }
  });

  it('never reveals its own answer: POC items hide the POC, all hide the VA', () => {
    for (const s of seeds('11110001', 8)) {
      const { item, ctx } = buildLoopItem('poc-va-snap', s, EASY_KNOBS);
      const flags = renderFlags(item);
      expect(flags.showVa).toBe(false);
      expect(flags.showPoc).toBe(ctx.target !== 'POC');
    }
  });
});

describe('excess-or-poor (Drill B)', () => {
  it('binary items match the planted excessTicks bands; judgment band is marked', () => {
    for (const s of seeds('22220000', 10)) {
      const { item, ctx } = buildLoopItem('excess-or-poor', s, EASY_KNOBS);
      const e = item.labels.extremes.find((x) => x.side === ctx.extremeSide);
      expect(e).toBeDefined();
      expect(ctx.extremeTicks).toBe(e?.excessTicks);
      expect(ctx.extremeClass).toBe(excessClassOf(e?.excessTicks ?? 0));
      if (ctx.extremeClass !== 'judgment') {
        expect(item.groundTruth).toBe(ctx.extremeClass === 'excess' ? 'EXCESS' : 'POOR');
      }
      expect(item.choices).toEqual(['EXCESS', 'POOR']);
      // highlighted row exists, sits at the extreme it claims, and is the
      // profile's own outermost NONEMPTY row (bin containment — no volume
      // bar may print beyond the highlighted extreme)
      const row = item.stimulus.highlightRow!;
      const rows = item.stimulus.profile.rows;
      expect(row).toBeGreaterThanOrEqual(0);
      expect(row).toBeLessThan(rows.length);
      expect(rows[row]).toBeGreaterThan(0);
      if (ctx.extremeSide === 'high') {
        for (let r = row + 1; r < rows.length; r++) expect(rows[r]).toBe(0);
        expect(row).toBeGreaterThan((rows.length - 1) / 2);
      } else {
        for (let r = 0; r < row; r++) expect(rows[r]).toBe(0);
        expect(row).toBeLessThan((rows.length - 1) / 2);
      }
      // the pool declares its own mixture (never folklore)
      expect(item.labels.poolBaseRates['excess-or-poor']).toBe(0.5);
    }
  });
});

describe('shape-alphabet (Drill D)', () => {
  it('labels are honest: EVERY letter matches the measured classifyShape (GDD Drill D)', () => {
    const seen = new Set<string>();
    for (const s of seeds('33330000', 14)) {
      const { item, gen } = buildLoopItem('shape-alphabet', s, EASY_KNOBS);
      const truth = String(item.groundTruth);
      seen.add(truth);
      expect(item.choices).toEqual(['D', 'P', 'b', 'B', 'TREND']);
      // the letter the game grades against IS the letter its own classifier
      // measures on the rendered profile — a player answering the measured
      // shape can never be marked wrong
      const measured = classifyShape(buildProfile(gen.bars, gen.script.rowStep));
      expect(measured === 'thin-trend' ? 'TREND' : measured).toBe(truth);
      // scripted-day-type ancestry where one exists
      if (truth === 'D') expect(item.labels.dayType).toBe('normal');
      if (truth === 'TREND') expect(item.labels.dayType).toBe('trend');
      if (truth === 'B') expect(item.labels.dayType).toBe('double-distribution-trend');
    }
    expect(seen.size).toBeGreaterThanOrEqual(3); // the pool actually mixes
  });
});

describe('open-type-ladder (Drill F)', () => {
  it('reveals exactly the first 90 minutes and labels from the verified openType', () => {
    for (const s of seeds('44440000', 8)) {
      const { item, ctx } = buildLoopItem('open-type-ladder', s, EASY_KNOBS);
      expect(item.stimulus.bars.length).toBe(OPEN_REVEAL_BARS);
      expect(item.stimulus.revealBar).toBe(OPEN_REVEAL_BARS - 1);
      expect(item.groundTruth).toBe(OPEN_CHIP[item.labels.openType]);
      expect(item.choices).toContain(item.groundTruth);
      expect(item.choices.length).toBe(4);
      expect(ctx.openPrice).toBe(item.stimulus.bars[0].o);
    }
  });
});

describe('regime-gate (Drill I)', () => {
  it('truth = script segment at the snapshot (nontrend ⇒ NO TRADE)', () => {
    const seen = new Set<string>();
    for (const s of seeds('55550000', 12)) {
      const { item, ctx } = buildLoopItem('regime-gate', s, EASY_KNOBS);
      const truth = String(item.groundTruth);
      seen.add(truth);
      expect([...REGIME_CHIPS]).toContain(truth);
      const snap = item.stimulus.revealBar;
      expect(snap).toBeGreaterThanOrEqual(205);
      expect(snap).toBeLessThanOrEqual(330);
      if (item.labels.dayType === 'nontrend') {
        expect(truth).toBe('NO TRADE');
        expect(ctx.rangePct).toBeLessThan(100);
      } else {
        const seg = item.labels.script.find((g) => snap >= g.startBar && snap <= g.endBar);
        expect(truth).toBe(seg?.regime === 'imbalance' ? 'FOLLOW' : 'FADE');
      }
      // cardinal-template honesty: trend snapshots stay inside the unbroken run
      if (item.labels.dayType === 'trend' && item.labels.oneTimeframingBreakBar !== null) {
        expect(snap).toBeLessThan(item.labels.oneTimeframingBreakBar);
      }
      // the stimulus is a true mid-session prefix
      expect(item.stimulus.bars.length).toBe(snap + 1);
    }
    expect(seen.size).toBeGreaterThanOrEqual(2);
  });
});

describe('barClock', () => {
  it('maps bar indices to the RTH clock', () => {
    expect(barClock(0)).toBe('09:30');
    expect(barClock(60)).toBe('10:30');
    expect(barClock(389)).toBe('15:59');
  });
});
