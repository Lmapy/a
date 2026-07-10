import { describe, expect, it } from 'vitest';
import { Prng, RandomStream, fnv1a64, siblingSeed, type PrngState } from './prng';

describe('Prng — determinism', () => {
  it('same seed + same stream name => identical sequence', () => {
    const a = new Prng('123456789').stream('noise');
    const b = new Prng('123456789').stream('noise');
    for (let i = 0; i < 1000; i++) expect(a.nextU32()).toBe(b.nextU32());
  });

  it('different stream names => independent sequences', () => {
    const p = new Prng('42');
    const noise = p.stream('noise');
    const volume = p.stream('volume');
    const noiseSeq = Array.from({ length: 8 }, () => noise.nextU32());
    const volumeSeq = Array.from({ length: 8 }, () => volume.nextU32());
    expect(noiseSeq).not.toEqual(volumeSeq);
  });

  it('substream independence (Woodpecker-sibling guarantee): consuming one stream never affects another', () => {
    const p1 = new Prng('987654321');
    const skeleton1 = p1.stream('skeleton');
    // consume lots of 'decoys' draws on p1 only
    const d = p1.stream('decoys');
    for (let i = 0; i < 500; i++) d.nextU32();

    const p2 = new Prng('987654321');
    const skeleton2 = p2.stream('skeleton');
    for (let i = 0; i < 100; i++) expect(skeleton1.nextU32()).toBe(skeleton2.nextU32());
  });

  it('different seeds => different sequences', () => {
    const a = new Prng('1').stream('noise');
    const b = new Prng('2').stream('noise');
    const seqA = Array.from({ length: 8 }, () => a.nextU32());
    const seqB = Array.from({ length: 8 }, () => b.nextU32());
    expect(seqA).not.toEqual(seqB);
  });

  it('accepts 64-bit seeds as decimal strings without precision loss', () => {
    const big = '18446744073709551615'; // 2^64 - 1
    const p = new Prng(big);
    expect(p.seedString).toBe(big);
    expect(() => p.stream('noise').nextU32()).not.toThrow();
  });
});

describe('RandomStream — ranges & state', () => {
  it('nextFloat stays in [0, 1)', () => {
    const s = new Prng('7').stream('noise');
    for (let i = 0; i < 10000; i++) {
      const f = s.nextFloat();
      expect(f).toBeGreaterThanOrEqual(0);
      expect(f).toBeLessThan(1);
    }
  });

  it('nextInt covers the full inclusive range and nothing else', () => {
    const s = new Prng('7').stream('volume');
    const seen = new Set<number>();
    for (let i = 0; i < 5000; i++) {
      const n = s.nextInt(3, 7);
      expect(n).toBeGreaterThanOrEqual(3);
      expect(n).toBeLessThanOrEqual(7);
      seen.add(n);
    }
    expect(seen.size).toBe(5);
  });

  it('getState/setState snapshot-restores exactly (save/scrub/micro-replay)', () => {
    const s = new Prng('99').stream('noise');
    for (let i = 0; i < 37; i++) s.nextU32();
    const snap = s.getState();
    const ahead = Array.from({ length: 20 }, () => s.nextU32());
    s.setState(snap);
    const replay = Array.from({ length: 20 }, () => s.nextU32());
    expect(replay).toEqual(ahead);
  });

  it('nextGaussian has roughly zero mean and unit variance', () => {
    const s = new Prng('2024').stream('noise');
    const n = 20000;
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      const g = s.nextGaussian();
      sum += g;
      sumSq += g * g;
    }
    const mean = sum / n;
    const variance = sumSq / n - mean * mean;
    expect(Math.abs(mean)).toBeLessThan(0.03);
    expect(variance).toBeGreaterThan(0.9);
    expect(variance).toBeLessThan(1.1);
  });

  it('nextStudentT(6) is standardized (unit variance) and fatter-tailed than normal', () => {
    const s = new Prng('2025').stream('noise');
    const n = 30000;
    let sumSq = 0;
    let kurt = 0;
    for (let i = 0; i < n; i++) {
      const t = s.nextStudentT(6);
      sumSq += t * t;
      kurt += t * t * t * t;
    }
    const variance = sumSq / n;
    expect(variance).toBeGreaterThan(0.9);
    expect(variance).toBeLessThan(1.15);
    // excess kurtosis of standardized t(6) = 3; allow slack, but must exceed normal's 0
    expect(kurt / n / (variance * variance) - 3).toBeGreaterThan(0.8);
  });

  it('guards against the all-zero xoshiro state', () => {
    const s = new RandomStream([0, 0, 0, 0] as PrngState);
    // an unguarded all-zero xoshiro state emits zeros forever; ours must not
    const vals = new Set(Array.from({ length: 10 }, () => s.nextU32()));
    expect(vals.size).toBeGreaterThan(1);
  });
});

describe('fnv1a64 / siblingSeed', () => {
  it('fnv1a64 matches the known reference vector', () => {
    // FNV-1a 64 of empty string is the offset basis
    expect(fnv1a64('')).toBe(0xcbf29ce484222325n);
    // distinct names hash distinctly
    expect(fnv1a64('skeleton')).not.toBe(fnv1a64('noise'));
  });

  it('siblingSeed is deterministic, variant-sensitive, and differs from the parent', () => {
    const parent = '5551212';
    expect(siblingSeed(parent, 1)).toBe(siblingSeed(parent, 1));
    expect(siblingSeed(parent, 1)).not.toBe(siblingSeed(parent, 2));
    expect(siblingSeed(parent, 1)).not.toBe(parent);
  });
});
