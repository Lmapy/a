import { describe, expect, it } from 'vitest';
import { Prng } from './prng';
import { generateSession, seasonalVol } from './generator';
import { compileScript } from './scripts';

function makeScript(seed: string) {
  return compileScript(new Prng(seed), 'normal', 'open-auction', 'in-value');
}

describe('generateSession (stub contract)', () => {
  it('determinism: same seed => byte-identical session (GDD §7)', () => {
    const a = generateSession(makeScript('20260709'));
    const b = generateSession(makeScript('20260709'));
    expect(a.bars).toEqual(b.bars);
    expect(a.labels).toEqual(b.labels);
  });

  it('different seeds => different paths', () => {
    const a = generateSession(makeScript('1'));
    const b = generateSession(makeScript('2'));
    expect(a.bars).not.toEqual(b.bars);
  });

  it('emits well-formed bars: h >= max(o,c), l <= min(o,c), v > 0, t sequential', () => {
    const { bars } = generateSession(makeScript('8675309'));
    expect(bars).toHaveLength(390);
    bars.forEach((bar, i) => {
      expect(bar.t).toBe(i);
      expect(bar.h).toBeGreaterThanOrEqual(Math.max(bar.o, bar.c));
      expect(bar.l).toBeLessThanOrEqual(Math.min(bar.o, bar.c));
      expect(bar.v).toBeGreaterThan(0);
    });
  });

  it('labels echo the script ground truth', () => {
    const script = makeScript('44');
    const { labels } = generateSession(script);
    expect(labels.dayType).toBe(script.dayType);
    expect(labels.openType).toBe(script.openType);
    expect(labels.script).toEqual(script.segments);
    expect(labels.ibWidthRatio).toBe(script.ibWidthRatio);
  });
});

describe('seasonalVol', () => {
  it('open is the loudest, midday the quietest, close between (U-shape)', () => {
    const open = seasonalVol(0, 390);
    const midday = seasonalVol(195, 390);
    const close = seasonalVol(389, 390);
    expect(open).toBeGreaterThan(midday);
    expect(close).toBeGreaterThan(midday);
    expect(open).toBeGreaterThan(close);
  });

  it('open/midday ratio is in the GDD 2–4× band', () => {
    const ratio = seasonalVol(0, 390) / seasonalVol(195, 390);
    expect(ratio).toBeGreaterThanOrEqual(2);
    expect(ratio).toBeLessThanOrEqual(4);
  });
});
