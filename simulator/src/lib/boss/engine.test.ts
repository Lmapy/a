/* ============================================================================
   @boss/engine tests — session predicates, ticket math, the equity reducer
   with prop-desk limits, legality grading, Read Score, the luck
   decomposition, measured tells, and the fade-bot demo run (all GDD §6).
   ========================================================================== */

import { describe, expect, it } from 'vitest';
import type { Bar } from '../types';
import {
  BOSS_TEMPLATES,
  BOSS1_DEBRIEF_LINE,
  bossTemplate,
  renderBossTemplate,
} from './templates';
import {
  CARDINAL_WEIGHT,
  DAILY_LOSS_LIMIT_R,
  DEMO_SEED,
  GOWITH_TARGET_R,
  IB_TELL_MAX,
  OTF_HOLD_THROUGH_BAR,
  PASS_READ_SCORE,
  PULLBACK_TELL_MAX,
  TRAILING_DD_R,
  TREND_RUN_START_BAR,
  barClock,
  buildBossSession,
  closeAtEnd,
  decisionBarsOf,
  decompose,
  enterPosition,
  fadeBot,
  gradeDecision,
  gradeReflex,
  impulseSegments,
  killPosition,
  measureTells,
  newEquity,
  otfBreakSince,
  otfSinceClock,
  planFade,
  planGoWith,
  readScore,
  reflexTruth,
  runCaption,
  scriptRegimeAt,
  stepBar,
  unrealizedR,
} from './engine';
import type { BossSession, GradedEvent, OpenPosition } from './engine';

/* ------------------------------------------------------------- fixtures */

let _demo: BossSession | null = null;
function demo(): BossSession {
  if (!_demo) _demo = buildBossSession(DEMO_SEED);
  return _demo;
}

function mkBar(t: number, o: number, h: number, l: number, c: number, v = 100): Bar {
  return { t, o, h, l, c, v };
}

/* ------------------------------------------------------------- session */

describe('buildBossSession', () => {
  it('is deterministic: same seed ⇒ byte-identical session', () => {
    const a = buildBossSession('7');
    const b = buildBossSession('7');
    expect(JSON.stringify(a.gen.bars)).toBe(JSON.stringify(b.gen.bars));
    expect(JSON.stringify(a.gen.labels)).toBe(JSON.stringify(b.gen.labels));
    expect(a.variant).toBe(b.variant);
    expect(a.decisionBars).toEqual(b.decisionBars);
  });

  it('meets the gauntlet predicates: labeled trend day, up, scripted OTF never breaks, poor high', () => {
    const s = demo();
    expect(s.gen.labels.dayType).toBe('trend');
    expect(s.trendDir).toBe(1);
    expect(s.gen.labels.oneTimeframingBreakBar).toBeNull();
    const hi = s.gen.labels.extremes.find((e) => e.side === 'high');
    expect(hi).toBeDefined();
    expect(hi!.excessTicks).toBeLessThanOrEqual(1); // poor high overhead
    // close in the extreme of the range (trend contract): day closes up
    const bars = s.gen.bars;
    expect(bars[bars.length - 1].c).toBeGreaterThan(bars[0].o);
  });

  it('measured one-timeframing holds through every decision and interrupt', () => {
    const s = demo();
    const brk = otfBreakSince(s.gen.bars, TREND_RUN_START_BAR, 'up');
    expect(brk === null || brk >= OTF_HOLD_THROUGH_BAR).toBe(true);
  });

  it('derives three scripted decision points at impulse ends', () => {
    const s = demo();
    expect(s.decisionBars).toHaveLength(3);
    const ends = impulseSegments(s.gen.labels.script).map((x) => x.endBar);
    for (const d of s.decisionBars) {
      expect(ends).toContain(d);
      expect(d).toBeGreaterThanOrEqual(60);
      expect(d).toBeLessThanOrEqual(330);
    }
    expect(decisionBarsOf(s.gen.labels, s.gen.script.nBars)).toEqual(s.decisionBars);
  });

  it('measured IB tell stays inside the taught band (≤ IB_TELL_MAX) across seeds', () => {
    for (const seed of [DEMO_SEED, '7', '42', '1001', '20260710']) {
      const s = buildBossSession(seed);
      const t = measureTells(
        s.gen.bars,
        s.gen.labels,
        s.gen.bars.length - 1,
        s.gen.script.rowStep,
        1,
        s.normalIbPts,
      );
      expect(t.ibRatio, `seed ${seed}`).toBeLessThanOrEqual(IB_TELL_MAX);
      expect(t.ibRatio, `seed ${seed}`).toBeGreaterThan(0.15);
    }
  });

  it('measures prior-day landmarks and a normal IB width from the priors', () => {
    const s = demo();
    expect(s.pVal).toBeLessThan(s.pVah);
    expect(s.pPoc).toBeGreaterThanOrEqual(s.pVal);
    expect(s.pPoc).toBeLessThanOrEqual(s.pVah);
    expect(s.normalIbPts).toBeGreaterThan(0);
    // out-of-range open-drive up: the whole session trades above prior value
    expect(s.gen.bars[0].o).toBeGreaterThan(s.pVah);
  });
});

describe('clock + script lookups', () => {
  it('barClock maps bar 0 → 09:30 and bar 390 → 16:00', () => {
    expect(barClock(0)).toBe('09:30');
    expect(barClock(60)).toBe('10:30');
    expect(barClock(390)).toBe('16:00');
  });

  it('scriptRegimeAt reads the segment timeline', () => {
    const labels = demo().gen.labels;
    expect(scriptRegimeAt(labels, 10)).toBe('balance');
    for (const d of demo().decisionBars) expect(scriptRegimeAt(labels, d)).toBe('imbalance');
  });

  it('otfSinceClock anchors at the start of the contiguous imbalance run', () => {
    const labels = demo().gen.labels;
    // trend script: imbalance legs tile from bar 60 on
    expect(otfSinceClock(labels, demo().decisionBars[1])).toBe(barClock(60));
  });
});

/* ------------------------------------------------------------- tickets */

describe('ticket plans', () => {
  it('planFade: short at close, tidy stop above the high, ≥2R rotation target', () => {
    const s = demo();
    for (const d of s.decisionBars) {
      const p = planFade(s.gen.bars, d, s.gen.script.rowStep);
      expect(p.entry).toBe(s.gen.bars[d].c);
      expect(p.stop).toBeGreaterThan(p.entry);
      expect(p.target).toBeLessThan(p.entry);
      expect(p.riskPts).toBeCloseTo(p.stop - p.entry, 12);
      expect(p.rewardR).toBeGreaterThanOrEqual(2 - 1e-12);
    }
  });

  it('planGoWith: long at close, stop under the swing low, 2R projection', () => {
    const s = demo();
    for (const d of s.decisionBars) {
      const p = planGoWith(s.gen.bars, d, s.gen.script.rowStep);
      expect(p.stop).toBeLessThan(p.entry);
      expect(p.target).toBeGreaterThan(p.entry);
      expect(p.rewardR).toBe(GOWITH_TARGET_R);
      expect(p.target - p.entry).toBeCloseTo(GOWITH_TARGET_R * p.riskPts, 9);
    }
  });
});

/* -------------------------------------------------------- equity reducer */

const SHORT_PLAN = { entry: 100, stop: 102, target: 95, riskPts: 2, rewardR: 2.5 };
const LONG_PLAN = { entry: 100, stop: 98, target: 104, riskPts: 2, rewardR: 2 };

describe('equity reducer', () => {
  it('short: stop-out at −1R when the high crosses the stop', () => {
    const st = newEquity();
    enterPosition(st, 'fade', SHORT_PLAN, 5, 0);
    stepBar(st, mkBar(6, 100, 103, 99, 101));
    expect(st.lastExit).toMatchObject({ outcome: 'stop', rPnl: -1, decisionIndex: 0 });
    expect(st.realizedR).toBe(-1);
    expect(st.open).toBeNull();
  });

  it('short: target exit at +rewardR when the low crosses the target', () => {
    const st = newEquity();
    enterPosition(st, 'fade', SHORT_PLAN, 5, 0);
    stepBar(st, mkBar(6, 100, 101, 94, 96));
    expect(st.lastExit).toMatchObject({ outcome: 'target', rPnl: 2.5 });
  });

  it('stop has priority when both print inside one bar (conservative fill)', () => {
    const st = newEquity();
    enterPosition(st, 'fade', SHORT_PLAN, 5, 0);
    stepBar(st, mkBar(6, 100, 103, 94, 96));
    expect(st.lastExit?.outcome).toBe('stop');
  });

  it('long positions mirror the exit logic', () => {
    const st = newEquity();
    enterPosition(st, 'go-with', LONG_PLAN, 5, 0);
    stepBar(st, mkBar(6, 100, 105, 99, 104));
    expect(st.lastExit).toMatchObject({ outcome: 'target', rPnl: 2 });
  });

  it('marks open equity per bar; peak trails REALIZED equity only', () => {
    const st = newEquity();
    enterPosition(st, 'go-with', { entry: 100, stop: 90, target: 200, riskPts: 10, rewardR: 10 }, 0, 0);
    stepBar(st, mkBar(1, 100, 121, 99, 120));
    expect(st.equityR).toBeCloseTo(2, 9); // +20 pts on 10-pt risk
    expect(st.peakR).toBe(0); // an unbanked open profit is not a peak
    stepBar(st, mkBar(2, 120, 120, 94, 95));
    expect(st.equityR).toBeCloseTo(-0.5, 9);
    expect(st.breach).toBeNull(); // dd from peak 0 is 0.5 < TRAILING_DD_R
  });

  it('daily loss limit force-closes the open position (breach outcome)', () => {
    const st = newEquity();
    st.realizedR = -2.2;
    enterPosition(st, 'fade', { entry: 100, stop: 105, target: 80, riskPts: 5, rewardR: 4 }, 5, 2);
    stepBar(st, mkBar(6, 100, 102, 99, 102));
    expect(st.equityR).toBeCloseTo(-2.6, 9);
    expect(st.equityR).toBeLessThanOrEqual(DAILY_LOSS_LIMIT_R);
    expect(st.breach).toBe('daily');
    expect(st.breachBar).toBe(6);
    expect(st.lastExit).toMatchObject({ outcome: 'breach', decisionIndex: 2 });
    expect(st.open).toBeNull();
    expect(st.realizedR).toBeCloseTo(-2.6, 9);
  });

  it('trailing drawdown from the realized peak breaches the run', () => {
    const st = newEquity();
    st.peakR = 1.8;
    st.realizedR = 1.8 - TRAILING_DD_R;
    stepBar(st, mkBar(9, 100, 100, 100, 100));
    expect(st.breach).toBe('trailing');
  });

  it('killPosition realizes at the close and can raise the realized peak', () => {
    const st = newEquity();
    enterPosition(st, 'fade', SHORT_PLAN, 5, 0);
    killPosition(st, mkBar(7, 100, 100.5, 98.5, 99));
    expect(st.open).toBeNull();
    expect(st.realizedR).toBeCloseTo(0.5, 9);
    expect(st.peakR).toBeCloseTo(0.5, 9);
    expect(st.lastExit).toMatchObject({ outcome: 'close', bar: 7 });
    // no-op when flat
    expect(killPosition(st, mkBar(8, 99, 99, 99, 99)).realizedR).toBeCloseTo(0.5, 9);
  });

  it('closeAtEnd realizes an open position at the final close', () => {
    const st = newEquity();
    enterPosition(st, 'fade', SHORT_PLAN, 5, 1);
    closeAtEnd(st, mkBar(389, 100, 101, 99, 99));
    expect(st.lastExit).toMatchObject({ outcome: 'close', decisionIndex: 1 });
    expect(st.realizedR).toBeCloseTo(0.5, 9); // short, 1 pt in favor on 2-pt risk
    expect(st.open).toBeNull();
  });

  it('unrealizedR is signed by direction', () => {
    const pos: OpenPosition = { action: 'fade', dir: -1, entryBar: 0, plan: SHORT_PLAN, decisionIndex: 0 };
    expect(unrealizedR(pos, 98)).toBeCloseTo(1, 9);
    expect(unrealizedR(pos, 102)).toBeCloseTo(-1, 9);
  });
});

/* ------------------------------------------------------------- grading */

describe('gradeDecision', () => {
  const labels = () => demo().gen.labels;
  const bar = () => demo().decisionBars[1];

  it('fade on a labeled trend day is the cardinal error: 0, ×3, forced template', () => {
    const g = gradeDecision('fade', 'balance', labels(), bar(), planFade(demo().gen.bars, bar(), 0.5), 1);
    expect(g.legal).toBe(false);
    expect(g.cardinal).toBe(true);
    expect(g.weight).toBe(CARDINAL_WEIGHT);
    expect(g.score).toBe(0);
    expect(g.verdict.cardinal).toBe(true);
    expect(g.verdict.explanation.startsWith('✗✗')).toBe(true);
    expect(g.verdict.explanation).toContain('one-timeframing since 10:30');
    expect(g.verdict.explanationTemplateId).toBe('boss1.fade.cardinal');
  });

  it('fade is cardinal regardless of the declared read', () => {
    const g = gradeDecision('fade', 'imbalance', labels(), bar(), null, 1);
    expect(g.cardinal).toBe(true);
    expect(g.score).toBe(0);
  });

  it('go-with + imbalance read = 100; go-with + balance read = 55', () => {
    const plan = planGoWith(demo().gen.bars, bar(), 0.5);
    const hit = gradeDecision('go-with', 'imbalance', labels(), bar(), plan, 1);
    expect(hit).toMatchObject({ legal: true, cardinal: false, readCorrect: true, score: 100, weight: 1 });
    expect(hit.verdict.explanation).toContain(plan.stop.toFixed(2));
    const wrong = gradeDecision('go-with', 'balance', labels(), bar(), plan, 1);
    expect(wrong).toMatchObject({ legal: true, readCorrect: false, score: 55 });
  });

  it('stand-aside is legal: 75 with the right read, 40 with the wrong one', () => {
    expect(gradeDecision('stand-aside', 'imbalance', labels(), bar(), null, 1).score).toBe(75);
    expect(gradeDecision('stand-aside', 'balance', labels(), bar(), null, 1).score).toBe(40);
  });
});

describe('gradeReflex', () => {
  const shortPos: OpenPosition = { action: 'fade', dir: -1, entryBar: 0, plan: SHORT_PLAN, decisionIndex: 0 };
  const longPos: OpenPosition = { action: 'go-with', dir: 1, entryBar: 0, plan: LONG_PLAN, decisionIndex: 0 };

  it('truth: counter-trend position → KILL, with-trend → HOLD', () => {
    expect(reflexTruth(shortPos, 1)).toBe('KILL');
    expect(reflexTruth(longPos, 1)).toBe('HOLD');
  });

  it('scores on correctness AND latency: 100 sharp / 60 slow / 0 wrong / slept', () => {
    expect(gradeReflex('KILL', 'KILL', 1200).score).toBe(100);
    expect(gradeReflex('KILL', 'KILL', 3500).score).toBe(60);
    expect(gradeReflex('KILL', 'HOLD', 800)).toMatchObject({ correct: false, score: 0 });
    const slept = gradeReflex('KILL', null, 5000);
    expect(slept).toMatchObject({ slept: true, score: 0 });
    expect(slept.verdict.explanationTemplateId).toBe('boss1.reflex.slept');
  });
});

/* --------------------------------------------- read score + decomposition */

const EV = (over: Partial<GradedEvent>): GradedEvent => ({
  kind: 'decision',
  score: 100,
  weight: 1,
  legal: true,
  readCorrect: true,
  cardinal: false,
  rPnl: null,
  ...over,
});

describe('readScore', () => {
  it('is the weighted mean; a cardinal drags 3×', () => {
    const events = [
      EV({ score: 0, weight: CARDINAL_WEIGHT, legal: false, cardinal: true }),
      ...Array.from({ length: 5 }, () => EV({})),
    ];
    expect(readScore(events)).toBe(63); // 500 / 8
    expect(readScore([])).toBe(0);
  });
});

describe('decompose (luck channel, GDD §5)', () => {
  it('legal + correctly-read wins are earned; legal losses go to luck', () => {
    const events = [EV({ rPnl: 0.8 }), EV({ rPnl: -1 })];
    const d = decompose(events, -0.2, 80);
    expect(d.earnedR).toBeCloseTo(0.8, 9);
    expect(d.errorR).toBe(0);
    expect(d.luckR).toBeCloseTo(-1, 9);
    expect(d.sentence).toContain('+0.8R');
    expect(d.sentence).not.toContain('wrong reads');
    expect(d.tail).toBe('Right process, unlucky outcome.');
  });

  it('illegal reads that made money earn zero credit', () => {
    const events = [EV({ rPnl: 2, legal: false, readCorrect: false, cardinal: true, score: 0, weight: 3 })];
    const d = decompose(events, 2, 20);
    expect(d.earnedR).toBe(0);
    expect(d.errorR).toBe(0);
    expect(d.luckR).toBeCloseTo(2, 9);
    expect(d.tail).toBe('Lucky outcome, wrong process — no credit.');
  });

  it('illegal losses are the error channel, never variance (fade-everything run)', () => {
    const events = Array.from({ length: 3 }, () =>
      EV({ rPnl: -0.9, legal: false, readCorrect: false, cardinal: true, score: 0, weight: 3 }),
    );
    const d = decompose(events, -2.7, 0);
    expect(d.earnedR).toBe(0);
    expect(d.errorR).toBeCloseTo(-2.7, 9);
    expect(d.luckR).toBeCloseTo(0, 9); // predictable losses are NOT luck
    expect(d.sentence).toContain('wrong reads cost −2.7R');
    expect(d.sentence).toContain("isn't variance");
    expect(d.tail).toBe('Process and outcome agree — drill the tells.');
  });

  it('legal but misread losses also land in the error channel', () => {
    const d = decompose([EV({ rPnl: -1, legal: true, readCorrect: false, score: 55 })], -1, 55);
    expect(d.errorR).toBeCloseTo(-1, 9);
    expect(d.luckR).toBeCloseTo(0, 9);
  });

  it('reflex events never enter the earned channel', () => {
    const d = decompose([EV({ kind: 'reflex', rPnl: 1 })], 1, PASS_READ_SCORE);
    expect(d.earnedR).toBe(0);
  });
});

/* ---------------------------------------------------------------- tells */

describe('measureTells', () => {
  it('measures the trend contract at the last decision bar', () => {
    const s = demo();
    const d = s.decisionBars[2];
    const t = measureTells(s.gen.bars, s.gen.labels, d, s.gen.script.rowStep, 1, s.normalIbPts);
    expect(t.ibRatio).toBeGreaterThan(0.15);
    expect(t.ibRatio).toBeLessThan(0.8); // narrow IB vs the priors
    expect(t.maxPullbackFrac).not.toBeNull();
    expect(t.maxPullbackFrac!).toBeLessThanOrEqual(PULLBACK_TELL_MAX);
    expect(t.impulseVolBasis).toBe('pullbacks');
    expect(t.impulseVolRatio).toBeGreaterThan(1); // volume expands with the move
    expect(t.otfBrackets).toBeGreaterThanOrEqual(5);
  });

  it('has no completed pullback before the first decision (balance vol basis)', () => {
    const s = demo();
    const t = measureTells(s.gen.bars, s.gen.labels, s.decisionBars[0], s.gen.script.rowStep, 1, s.normalIbPts);
    expect(t.maxPullbackFrac).toBeNull();
    expect(t.impulseVolBasis).toBe('balance');
  });
});

/* ---------------------------------------------------------- fade-bot run */

describe('fadeBot (the classic mistake, simulated)', () => {
  it('demo seed: three fades, every one loses, the daily limit ends the run', () => {
    const run = fadeBot(demo());
    expect(run.trades).toHaveLength(3);
    for (const t of run.trades) {
      expect(t.action).toBe('fade');
      expect(t.exit).not.toBeNull();
      expect(t.exit!.rPnl).toBeLessThan(0);
    }
    expect(run.breach).toBe('daily');
    expect(run.pnlR).toBeLessThanOrEqual(DAILY_LOSS_LIMIT_R);
    expect(runCaption(run)).toBe(`3 fades taken · 3 losses · limit breached ${barClock(run.breachBar!)}`);
  });

  it('fades lose across seed families (the boss premise, measured)', () => {
    let fades = 0;
    let losses = 0;
    for (const seed of ['7', '11', '21', '101']) {
      const run = fadeBot(buildBossSession(seed));
      for (const t of run.trades) {
        fades++;
        if ((t.exit?.rPnl ?? 0) < 0) losses++;
      }
    }
    expect(fades).toBeGreaterThanOrEqual(8);
    expect(losses / fades).toBeGreaterThanOrEqual(0.9);
  });

  it('is deterministic', () => {
    const a = fadeBot(demo());
    const b = fadeBot(buildBossSession(DEMO_SEED));
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });
});

/* ------------------------------------------------------------- templates */

describe('boss templates (GDD §10-B style rules)', () => {
  it('every template: ≤160 chars to the first period, no "80%", task-referenced', () => {
    for (const t of BOSS_TEMPLATES.values()) {
      const firstSentence = t.text.split('.')[0];
      expect(firstSentence.length, t.id).toBeLessThanOrEqual(160);
      expect(t.text).not.toContain('80%');
      expect(t.text.toLowerCase()).not.toContain('you always');
      expect(t.refs.length, t.id).toBeGreaterThan(0);
    }
  });

  it('renderBossTemplate fills slots and throws on missing ones', () => {
    const t = bossTemplate('boss1.fade.cardinal');
    expect(renderBossTemplate(t, { time: '10:30' })).toContain('10:30');
    expect(() => renderBossTemplate(t, {})).toThrow(/missing slot/);
  });

  it('the debrief line is the GDD line, verbatim', () => {
    expect(BOSS1_DEBRIEF_LINE).toBe('Every fade looked perfect. Every fade lost. That is what a trend day does.');
  });
});
