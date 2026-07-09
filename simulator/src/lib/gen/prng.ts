/* ============================================================================
   Seeded PRNG with named substreams (GDD §7 determinism spec).

   Design:
   · 64-bit master seed (decimal string, BigInt internally).
   · splitmix64 over (masterSeed XOR fnv1a64(streamName)) derives each named
     substream's state — so nudging the knob that consumes one substream
     ("decoys") never reshuffles another ("skeleton"): the Woodpecker-sibling
     guarantee.
   · Each substream is a 32-bit xoshiro128** generator (fast, integer-only —
     no engine-dependent transcendental bits in the hot path).
   · State is 4 uint32 per stream → free save/scrub/micro-replay.

   Pure TypeScript, zero imports. Deterministic across engines.
   ========================================================================== */

const MASK64 = (1n << 64n) - 1n;

/** FNV-1a 64-bit hash of a stream name → BigInt. */
export function fnv1a64(name: string): bigint {
  let h = 0xcbf29ce484222325n;
  for (let i = 0; i < name.length; i++) {
    h ^= BigInt(name.charCodeAt(i) & 0xff);
    h = (h * 0x100000001b3n) & MASK64;
  }
  return h;
}

/** One splitmix64 step. Returns [nextState, output]. */
function splitmix64(state: bigint): [bigint, bigint] {
  state = (state + 0x9e3779b97f4a7c15n) & MASK64;
  let z = state;
  z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
  z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
  z = z ^ (z >> 31n);
  return [state, z];
}

function rotl(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}

/** Serializable substream state: exactly four uint32 words. */
export type PrngState = [number, number, number, number];

/**
 * A named deterministic random stream (xoshiro128**).
 * Construct via Prng.stream(name) — never directly.
 */
export class RandomStream {
  private s0: number;
  private s1: number;
  private s2: number;
  private s3: number;

  constructor(state: PrngState) {
    // xoshiro must never be seeded all-zero; splitmix derivation makes this
    // astronomically unlikely, but guard anyway.
    if ((state[0] | state[1] | state[2] | state[3]) === 0) state = [1, 2, 3, 4];
    [this.s0, this.s1, this.s2, this.s3] = state;
  }

  /** Next uint32 in [0, 2^32). */
  nextU32(): number {
    const result = (Math.imul(rotl(Math.imul(this.s1, 5) >>> 0, 7), 9)) >>> 0;
    const t = (this.s1 << 9) >>> 0;
    this.s2 = (this.s2 ^ this.s0) >>> 0;
    this.s3 = (this.s3 ^ this.s1) >>> 0;
    this.s1 = (this.s1 ^ this.s2) >>> 0;
    this.s0 = (this.s0 ^ this.s3) >>> 0;
    this.s2 = (this.s2 ^ t) >>> 0;
    this.s3 = rotl(this.s3, 11);
    return result;
  }

  /** Uniform float in [0, 1) with 32 bits of resolution. */
  nextFloat(): number {
    return this.nextU32() / 4294967296;
  }

  /** Uniform integer in [min, max] inclusive. */
  nextInt(min: number, max: number): number {
    if (max < min) throw new Error(`nextInt: max (${max}) < min (${min})`);
    const span = max - min + 1;
    return min + Math.floor(this.nextFloat() * span);
  }

  /**
   * Standard normal via Box–Muller on stream floats. Uses Math.sqrt/log/cos —
   * acceptable: IEEE-754 correctly-rounded ops give cross-engine determinism
   * for these; only exotic transcendentals are banned by the GDD.
   */
  nextGaussian(): number {
    let u = this.nextFloat();
    if (u === 0) u = 2 ** -32; // avoid log(0)
    const v = this.nextFloat();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /**
   * Standardized Student-t with ν degrees of freedom (variance normalized to 1
   * for ν > 2) — the innovation distribution of the GARCH kernel (GDD §7).
   */
  nextStudentT(nu: number): number {
    // t = Z / sqrt(ChiSq(nu)/nu); ChiSq via sum of nu squared normals
    // (nu is a small integer knob, 4–8, so this is cheap and exact).
    const z = this.nextGaussian();
    let chi = 0;
    const k = Math.round(nu);
    for (let i = 0; i < k; i++) {
      const g = this.nextGaussian();
      chi += g * g;
    }
    const t = z / Math.sqrt(chi / k);
    return nu > 2 ? t / Math.sqrt(nu / (nu - 2)) : t;
  }

  /** Snapshot the current state (4 uint32) — for save/scrub/micro-replay. */
  getState(): PrngState {
    return [this.s0, this.s1, this.s2, this.s3];
  }

  /** Restore a previously snapshotted state. */
  setState(state: PrngState): void {
    [this.s0, this.s1, this.s2, this.s3] = state;
  }
}

/** The named substreams the generator uses (GDD §7). Extend, don't rename. */
export type StreamName =
  | 'skeleton' // script compilation: segment boundaries, anchors
  | 'noise'    // Brownian-bridge fill + GARCH innovations
  | 'volume'   // per-bar volume draws
  | 'prints'   // print-arrival timing (v2 footprint)
  | 'decoys'   // decoy structure placement
  | (string & {}); // forward-compatible custom streams

/**
 * Master PRNG for one session. Same (seed) => identical substreams; each
 * named substream is independent of every other.
 */
export class Prng {
  private readonly seed: bigint;

  /** @param masterSeed 64-bit seed as decimal string or bigint. */
  constructor(masterSeed: string | bigint) {
    this.seed = BigInt(masterSeed) & MASK64;
  }

  /** The master seed as a decimal string (persist this). */
  get seedString(): string {
    return this.seed.toString(10);
  }

  /**
   * Derive the named substream. Deterministic: same (seed, name) always
   * yields an identical stream; different names yield independent streams.
   */
  stream(name: StreamName): RandomStream {
    let s = (this.seed ^ fnv1a64(name)) & MASK64;
    const words: number[] = [];
    for (let i = 0; i < 2; i++) {
      let out: bigint;
      [s, out] = splitmix64(s);
      words.push(Number(out & 0xffffffffn), Number((out >> 32n) & 0xffffffffn));
    }
    return new RandomStream(words as PrngState);
  }
}

/**
 * Derive a sibling seed for Woodpecker re-serves: same session family,
 * variant-nudged. Deterministic and collision-avoiding via splitmix64.
 */
export function siblingSeed(masterSeed: string | bigint, variant: number): string {
  const s = (BigInt(masterSeed) ^ fnv1a64(`sibling:${variant}`)) & MASK64;
  const [, out] = splitmix64(s);
  return out.toString(10);
}
