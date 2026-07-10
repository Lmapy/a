import { describe, expect, it } from 'vitest';
import { Prng } from './prng';
import { DEFAULT_KNOBS, SESSION_BARS, compileScript } from './scripts';

describe('compileScript (stub contract)', () => {
  it('is deterministic: same seed => identical script', () => {
    const a = compileScript(new Prng('777'), 'trend', 'open-drive', 'out-of-range');
    const b = compileScript(new Prng('777'), 'trend', 'open-drive', 'out-of-range');
    expect(a).toEqual(b);
  });

  it('segments tile [0, nBars) with no gaps or overlaps', () => {
    const s = compileScript(new Prng('31337'), 'normal', 'open-auction', 'in-value', DEFAULT_KNOBS);
    expect(s.nBars).toBe(SESSION_BARS);
    expect(s.segments[0].startBar).toBe(0);
    for (let i = 1; i < s.segments.length; i++) {
      expect(s.segments[i].startBar).toBe(s.segments[i - 1].endBar + 1);
    }
    expect(s.segments[s.segments.length - 1].endBar).toBe(s.nBars - 1);
  });

  it('echoes taxonomy and carries seed + paramsVersion (re-derivability)', () => {
    const s = compileScript(new Prng('5'), 'normal-variation', 'open-test-drive', 'out-of-value-in-range');
    expect(s.dayType).toBe('normal-variation');
    expect(s.openType).toBe('open-test-drive');
    expect(s.openLocation).toBe('out-of-value-in-range');
    expect(s.seed).toBe('5');
    expect(s.paramsVersion).toBeTruthy();
    expect(s.rowStep).toBeGreaterThan(0);
  });

  it('coerces open types that contradict the day-type range premise', () => {
    // a nontrend day cannot host a drive; a neutral day cannot hold a
    // never-re-trade floor while extending both sides of the IB
    const a = compileScript(new Prng('5'), 'nontrend', 'open-drive', 'in-value');
    expect(a.openType).toBe('open-auction');
    const b = compileScript(new Prng('5'), 'neutral', 'open-test-drive', 'in-value');
    expect(b.openType).toBe('open-rejection-reverse');
  });
});
