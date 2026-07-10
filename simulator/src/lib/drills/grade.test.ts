/* Grader + template registry: scoring bands, misconception routing, cardinal
   errors, Brier folding, and the GDD §10-B template style audit. */
import { describe, expect, it } from 'vitest';
import type { Answer } from '../types';
import { EASY_KNOBS } from '../gen/scripts';
import { siblingSeed } from '../gen/prng';
import { buildLoopItem } from './items';
import type { LoopItem } from './items';
import { gradeLoopItem } from './grade';
import { TEMPLATES } from './templates';

const ans = (choice: string | number, confidence: Answer['confidence'] = null): Answer => ({
  itemId: 'x',
  choice,
  confidence,
  latencyMs: 1000,
});

/** Minimal hand-rolled chip item (grader only reads these fields). */
function chipItem(partial: {
  drillId: LoopItem['item']['drillId'];
  groundTruth: string;
  choices: string[];
  ctx?: LoopItem['ctx'];
  templateId?: string;
}): LoopItem {
  const bars = [{ t: 0, o: 5000, h: 5001, l: 4999, c: 5000.5, v: 100 }];
  return {
    item: {
      id: 't:1:0',
      drillId: partial.drillId,
      seed: '1',
      paramsVersion: 'test',
      stimulus: {
        bars,
        profile: {
          minPrice: 4999,
          rowStep: 0.5,
          rows: [1, 2, 3, 2, 1],
          poc: 2,
          vah: 3,
          val: 1,
          totalVolume: 9,
          hvnRanges: [],
          lvnRanges: [],
        },
        revealBar: 0,
      },
      question: 'q',
      choices: partial.choices,
      groundTruth: partial.groundTruth,
      explanationTemplateId: partial.templateId ?? 'shape.hit',
      labels: {
        dayType: 'normal',
        blendWeight: 0,
        blendedToward: null,
        openType: 'open-auction',
        openLocation: 'in-value',
        script: [],
        ibWidthRatio: 1,
        oneTimeframingBreakBar: null,
        extremes: [],
        planted: [],
        decoys: [],
        acceptanceFlags: [],
        conditionalProbs: {},
        ambiguityWeight: 0,
        poolBaseRates: {},
      },
      itemRating: 1000,
      parMs: 3000,
    },
    ctx: partial.ctx ?? {},
    gen: null as unknown as LoopItem['gen'], // grader never touches gen
  };
}

describe('Drill A grading (real items)', () => {
  const li = buildLoopItem('poc-va-snap', siblingSeed('909090', 1), EASY_KNOBS);
  const truth = li.item.groundTruth as number;

  it('exact = 100, ±1 = 60 (correct), ≥2 rows = 0 with direction template', () => {
    const exact = gradeLoopItem(li, ans(truth));
    expect(exact.correct).toBe(true);
    expect(exact.score).toBe(100);
    expect(exact.explanation.startsWith('✓')).toBe(true);

    const near = gradeLoopItem(li, ans(truth - 1));
    expect(near.correct).toBe(true);
    expect(near.score).toBe(60);

    const missRow = truth >= 3 ? truth - 3 : truth + 3;
    const miss = gradeLoopItem(li, ans(missRow));
    expect(miss.correct).toBe(false);
    expect(miss.score).toBe(0);
    expect(miss.explanation.startsWith('✗')).toBe(true);
    expect(miss.explanation).toContain('3 rows');
    expect(miss.refs.length).toBeGreaterThan(0);
    // snap drills carry no confidence tap → no Brier component
    expect(miss.brier).toBeNull();
  });

  it('a VA tap on the WRONG side of the POC routes to the bracket template', () => {
    // deterministically find one VAH and one VAL item in the sibling family
    for (const want of ['VAH', 'VAL'] as const) {
      let vi: LoopItem | null = null;
      for (let a = 0; a < 40 && vi === null; a++) {
        const cand = buildLoopItem('poc-va-snap', siblingSeed('909090', 100 + a), EASY_KNOBS);
        if (cand.ctx.target === want) vi = cand;
      }
      expect(vi, `no ${want} item found`).not.toBeNull();
      const poc = vi!.item.stimulus.profile.poc;
      // tap 2 rows PAST the POC on the opposite side of the requested edge
      const wrongPick = want === 'VAH' ? poc - 2 : poc + 2;
      const v = gradeLoopItem(vi!, ans(wrongPick));
      expect(v.correct).toBe(false);
      expect(v.score).toBe(0);
      expect(v.explanationTemplateId).toBe('pocva.miss.va.wrongSide');
      expect(v.explanation).toContain(want === 'VAH' ? 'below the POC' : 'above the POC');
      expect(v.explanation).toContain('brackets the POC');
      // same-side misses keep their existing templates
      const truth = vi!.item.groundTruth as number;
      const samePick = want === 'VAH' ? truth + 3 : truth - 3;
      const w = gradeLoopItem(vi!, ans(samePick));
      expect(w.explanationTemplateId).not.toBe('pocva.miss.va.wrongSide');
      expect(['pocva.miss.va.ranOut', 'pocva.miss.va.swallowed', 'pocva.miss.va.fatter']).toContain(
        w.explanationTemplateId,
      );
    }
  });
});

describe('Drill B grading', () => {
  it('poor called excess routes to the repair-magnet template with side noun', () => {
    const li = chipItem({
      drillId: 'excess-or-poor',
      groundTruth: 'POOR',
      choices: ['EXCESS', 'POOR'],
      ctx: { extremeSide: 'high', extremeTicks: 0, extremeClass: 'poor' },
    });
    const v = gradeLoopItem(li, ans('EXCESS', 'sure'));
    expect(v.correct).toBe(false);
    expect(v.explanationTemplateId).toBe('excess.miss.poorCalledExcess');
    expect(v.explanation).toContain('repair magnet');
    expect(v.explanation).toContain('not resistance');
    expect(v.brier).toBeCloseTo(0.81, 10); // sure (0.9) and wrong → (0.9−0)²
  });

  it('low-side poor uses "support"; excess hit reports the measured tail', () => {
    const lo = chipItem({
      drillId: 'excess-or-poor',
      groundTruth: 'POOR',
      choices: ['EXCESS', 'POOR'],
      ctx: { extremeSide: 'low', extremeTicks: 1, extremeClass: 'poor' },
    });
    expect(gradeLoopItem(lo, ans('EXCESS')).explanation).toContain('not support');

    const ex = chipItem({
      drillId: 'excess-or-poor',
      groundTruth: 'EXCESS',
      choices: ['EXCESS', 'POOR'],
      ctx: { extremeSide: 'high', extremeTicks: 6, extremeClass: 'excess' },
    });
    const v = gradeLoopItem(ex, ans('EXCESS', 'lean'));
    expect(v.correct).toBe(true);
    expect(v.explanation).toContain('6-tick tail');
    expect(v.brier).toBeCloseTo(0.0625, 10); // lean (0.75) and right → (0.75−1)²
  });

  it('the 2–3-tick band is a judgment call: score 50, excluded from Brier', () => {
    const li = chipItem({
      drillId: 'excess-or-poor',
      groundTruth: 'EXCESS',
      choices: ['EXCESS', 'POOR'],
      ctx: { extremeSide: 'high', extremeTicks: 2, extremeClass: 'judgment' },
    });
    const v = gradeLoopItem(li, ans('POOR', 'sure'));
    expect(v.correct).toBe(true);
    expect(v.score).toBe(50);
    expect(v.explanationTemplateId).toBe('excess.judgment');
    expect(v.brier).toBeNull();
  });
});

describe('Drill D grading', () => {
  const shapeItem = (truth: string): LoopItem =>
    chipItem({
      drillId: 'shape-alphabet',
      groundTruth: truth,
      choices: ['D', 'P', 'b', 'B', 'TREND'],
      ctx: { shapeTruth: truth },
    });

  it('routes P/b confusion to the GDD mirror-image template', () => {
    const v = gradeLoopItem(shapeItem('b'), ans('P'));
    expect(v.explanationTemplateId).toBe('shape.miss.PbConfusion');
    expect(v.explanation).toBe(
      "✗ That's a b, not a P — the bulge sits low with a thin upper tail. P is short-covering above; b is liquidation below. Mirror-image, opposite story.",
    );
    const w = gradeLoopItem(shapeItem('P'), ans('b'));
    expect(w.explanation).toContain('the bulge sits high with a thin lower tail');
  });

  it('routes trend↔B confusion to the neck templates; generic otherwise', () => {
    expect(gradeLoopItem(shapeItem('TREND'), ans('B')).explanationTemplateId).toBe('shape.miss.trendCalledB');
    expect(gradeLoopItem(shapeItem('B'), ans('TREND')).explanationTemplateId).toBe('shape.miss.BCalledTrend');
    expect(gradeLoopItem(shapeItem('D'), ans('TREND')).explanationTemplateId).toBe('shape.miss.generic');
    const hit = gradeLoopItem(shapeItem('D'), ans('D', 'guess'));
    expect(hit.correct).toBe(true);
    expect(hit.brier).toBeCloseTo(0.2025, 10); // guess (0.55) and right
  });
});

describe('Drill F grading', () => {
  it('explains with the truth’s ladder rung and the actual open price', () => {
    const li = buildLoopItem('open-type-ladder', siblingSeed('818181', 3), EASY_KNOBS);
    const truth = String(li.item.groundTruth);
    const wrong = li.item.choices.find((c) => c !== truth)!;
    const miss = gradeLoopItem(li, ans(wrong, 'lean'));
    expect(miss.correct).toBe(false);
    expect(miss.explanation.startsWith('✗')).toBe(true);
    expect(miss.explanation).toContain(li.item.stimulus.bars[0].o.toFixed(2));
    const hit = gradeLoopItem(li, ans(truth, 'lean'));
    expect(hit.correct).toBe(true);
    expect(hit.explanation.startsWith('✓')).toBe(true);
    expect(hit.explanationTemplateId).toBe(`open.truth.${li.item.labels.openType}`);
  });
});

describe('Drill I grading', () => {
  const regimeItem = (truth: string, dayType: 'trend' | 'normal' | 'nontrend'): LoopItem => {
    const li = chipItem({
      drillId: 'regime-gate',
      groundTruth: truth,
      choices: ['FADE', 'FOLLOW', 'NO TRADE'],
      ctx: {
        regimeTruth: truth as 'FADE' | 'FOLLOW' | 'NO TRADE',
        dayType,
        otfSince: '10:30',
        stairDir: 'up',
        rangePct: 43,
      },
    });
    li.item.labels.dayType = dayType;
    return li;
  };

  it('FADE on a scripted trend day is the cardinal error (3× weight upstream)', () => {
    const v = gradeLoopItem(regimeItem('FOLLOW', 'trend'), ans('FADE', 'sure'));
    expect(v.correct).toBe(false);
    expect(v.cardinal).toBe(true);
    expect(v.explanationTemplateId).toBe('regime.miss.cardinal');
    expect(v.explanation).toContain('One-timeframing since 10:30');
    expect(v.explanation).toContain('Every responsive fade looks perfect on a trend day');
  });

  it('non-cardinal misses route by truth; hits confirm the playbook', () => {
    const notrade = gradeLoopItem(regimeItem('FOLLOW', 'trend'), ans('NO TRADE'));
    expect(notrade.cardinal).toBe(false);
    expect(notrade.explanationTemplateId).toBe('regime.miss.imbalance');

    const v = gradeLoopItem(regimeItem('NO TRADE', 'nontrend'), ans('FOLLOW'));
    expect(v.explanationTemplateId).toBe('regime.miss.nontrend');
    expect(v.explanation).toContain('43%');

    const hit = gradeLoopItem(regimeItem('FADE', 'normal'), ans('FADE'));
    expect(hit.correct).toBe(true);
    expect(hit.cardinal).toBe(false);
    expect(hit.explanationTemplateId).toBe('regime.hit.balance');
  });
});

describe('template registry style audit (GDD §10-B)', () => {
  it('every template: ≤160 chars to first period, no folklore "80%", task-referenced', () => {
    for (const t of TEMPLATES.values()) {
      const rendered = t.text.replace(/\{\w+\}/g, 'X');
      const firstSentence = rendered.split('.')[0];
      expect(firstSentence.length, t.id).toBeLessThanOrEqual(160);
      expect(rendered, t.id).not.toContain('80%');
      expect(rendered.toLowerCase(), t.id).not.toMatch(/you always|you're bad|you never/);
      expect(t.refs.length, t.id).toBeGreaterThan(0);
      expect(t.text.length, t.id).toBeGreaterThan(20); // no naked verdicts
    }
  });

  it('verdicts are never naked: every graded path carries an explanation', () => {
    const li = buildLoopItem('shape-alphabet', '246810', EASY_KNOBS);
    for (const c of li.item.choices) {
      const v = gradeLoopItem(li, ans(c));
      expect(v.explanation.length).toBeGreaterThan(20);
      expect(v.refs.length).toBeGreaterThan(0);
    }
  });
});
