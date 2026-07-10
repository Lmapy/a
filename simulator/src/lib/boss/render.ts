/* ============================================================================
   @boss/render — the boss tape: price path over the session with the
   developing volume profile on the right edge. Geometry is a 1:1 port of
   the intro mockup's SVG (screen-boss-intro.html chart script: padL 12,
   axis 60, profile zone 52, hairline grid + right axis labels, dashed pVAH
   reference, ✗ pins over the path, POC row tagged orange, close dot).

   ACCURACY RULE (GDD §9): the profile is buildProfile() over the revealed
   bars (1-pt rows, as the mockup's panel head declares); every level line,
   pin and label is a price computed by the engine. This module positions
   pixels only.

   Colors/fonts come from tokens.css via readTapeTheme (getComputedStyle) —
   zero hex in here beyond the shared fallbacks.
   ========================================================================== */

import type { Bar } from '../types';
import { buildProfile, rowToPrice } from '../core/profile';
import { readChartTheme } from '../render/profileCanvas';
import type { ChartTheme } from '../render/profileCanvas';

export interface TapeTheme extends ChartTheme {
  /** --ink-body: the price path stroke (mockup uses body ink, not blue). */
  inkBody: string;
}

/** Read the tape theme from computed styles (tokens.css custom props). */
export function readTapeTheme(el: Element): TapeTheme {
  const base = readChartTheme(el);
  const v = getComputedStyle(el).getPropertyValue('--ink-body').trim();
  return { ...base, inkBody: v === '' ? '#c3c2b7' : v };
}

/** Mockup constants (screen-boss-intro.html). */
export const TAPE_PAD_L = 12;
export const TAPE_AXIS_W = 60;
export const TAPE_PROF_W = 52;

/** A pinned past call drawn over the path (✗ FADE 1 …). */
export interface TapePin {
  bar: number;
  price: number;
  /** Glyph inside the pin circle (✗ / ✓ / —). */
  glyph: string;
  /** Mono label above the pin (e.g. "FADE 1"). */
  label: string;
  tone: 'sell' | 'buy' | 'ghost';
}

/** A horizontal engine-computed level (entry/stop/target) with a text tag. */
export interface TapeLevel {
  price: number;
  label: string;
  tone: 'sell' | 'buy' | 'muted';
  /** Put the tag under the line (avoids collisions between close levels). */
  labelBelow?: boolean;
}

/**
 * How far beyond the revealed bar range the pVAH reference may extend the
 * y domain before it degrades to a bottom-edge tag (an out-of-range open
 * trades tens of points above prior value — pinning it in would squash the
 * live tape into a corner).
 */
export const PVAH_DOMAIN_SLACK = 0.5;

export interface TapeRenderOptions {
  theme: TapeTheme;
  /** X domain: total session bars (the tape grows rightward into it). */
  nBarsTotal: number;
  /** Profile row step in points (the panel head declares it). */
  rowStep: number;
  /** Prior-day VAH reference (dashed, tagged pVAH); null to omit. */
  pVah?: number | null;
  /** Extra prices the y domain must include (e.g. open position levels). */
  includePrices?: number[];
  pins?: TapePin[];
  levels?: TapeLevel[];
  /** Bottom-left mono caption (intro: the computed run summary). */
  caption?: string | null;
  padT?: number;
  padB?: number;
}

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());

function toneColor(theme: TapeTheme, tone: TapePin['tone'] | TapeLevel['tone']): string {
  if (tone === 'sell') return theme.sellRed;
  if (tone === 'buy') return theme.buyGreen;
  if (tone === 'muted') return theme.inkMuted;
  return theme.inkGhost;
}

/* ----------------------------------------------------------------------------
   Text-label placement (collision avoidance)

   Levels, pins and reference tags all float over the same narrow tape, so
   labels are placed via a small occupied-rect ledger: every drawn label (and
   every fixed pin circle) claims a rectangle; later labels try an ordered
   list of candidate anchors and take the first free one, then clamp inside
   the canvas so nothing ever clips at an edge. Fixes the playtest artifacts
   (TGT clipped at the left edge; STOP overprinting a pin label).
   -------------------------------------------------------------------------- */

interface LabelRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

function overlaps(a: LabelRect, b: LabelRect): boolean {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

class LabelLedger {
  private rects: LabelRect[] = [];

  claim(r: LabelRect): void {
    this.rects.push(r);
  }

  /**
   * First candidate whose rect (clamped into `bounds`) is free of claimed
   * rects AND of `blocked` geometry (e.g. the price path); a second pass
   * relaxes `blocked`, and the final fallback is the last candidate clamped.
   * `cands` are top-left anchors of a w×h label box.
   */
  place(
    cands: { x: number; y: number }[],
    w: number,
    h: number,
    bounds: LabelRect,
    blocked?: (r: LabelRect) => boolean,
  ): LabelRect {
    const clamp = (c: { x: number; y: number }): LabelRect => ({
      x: Math.min(Math.max(c.x, bounds.x), bounds.x + bounds.w - w),
      y: Math.min(Math.max(c.y, bounds.y), bounds.y + bounds.h - h),
      w,
      h,
    });
    let out: LabelRect | null = null;
    for (const relax of blocked ? [false, true] : [true]) {
      for (const c of cands) {
        const r = clamp(c);
        if (this.rects.some((p) => overlaps(p, r))) continue;
        if (!relax && blocked && blocked(r)) continue;
        out = r;
        break;
      }
      if (out) break;
    }
    out ??= clamp(cands[cands.length - 1]);
    this.claim(out);
    return out;
  }
}

/**
 * Render the boss tape. Returns measured ms (budget: same <8ms as the drill
 * profile). Bars must be a session prefix (bars[i].t === i).
 */
export function renderTape(
  canvas: HTMLCanvasElement,
  bars: Bar[],
  cssW: number,
  cssH: number,
  opts: TapeRenderOptions,
): number {
  const t0 = now();
  const { theme } = opts;
  const dpr = (globalThis as { devicePixelRatio?: number }).devicePixelRatio ?? 1;
  const pw = Math.max(1, Math.round(cssW * dpr));
  const ph = Math.max(1, Math.round(cssH * dpr));
  if (canvas.width !== pw) canvas.width = pw;
  if (canvas.height !== ph) canvas.height = ph;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('bossTape: no 2d context');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  if (bars.length < 2) return now() - t0;

  const padT = opts.padT ?? 14;
  const padB = opts.padB ?? (opts.caption ? 30 : 14);
  const plotH = cssH - padT - padB;
  const pathR = cssW - TAPE_AXIS_W - TAPE_PROF_W - 10; // right edge of the path zone
  const gridR = cssW - TAPE_AXIS_W - 4; // grid + profile right edge

  // --- y domain: revealed range + required prices, 4% padded ---------------
  let pMin = Infinity;
  let pMax = -Infinity;
  for (const b of bars) {
    if (b.l < pMin) pMin = b.l;
    if (b.h > pMax) pMax = b.h;
  }
  for (const p of opts.includePrices ?? []) {
    if (p < pMin) pMin = p;
    if (p > pMax) pMax = p;
  }
  // pVAH joins the domain only while near; far away it becomes an edge tag
  const barSpan = Math.max(1e-9, pMax - pMin);
  let pVahInDomain = false;
  if (opts.pVah != null) {
    if (opts.pVah >= pMin - PVAH_DOMAIN_SLACK * barSpan && opts.pVah <= pMax + PVAH_DOMAIN_SLACK * barSpan) {
      pVahInDomain = true;
      if (opts.pVah < pMin) pMin = opts.pVah;
      if (opts.pVah > pMax) pMax = opts.pVah;
    }
  }
  const span = Math.max(1e-9, pMax - pMin);
  pMin -= span * 0.04;
  pMax += span * 0.04;

  const y = (price: number): number => padT + ((pMax - price) / (pMax - pMin)) * plotH;
  const x = (bar: number): number =>
    TAPE_PAD_L + (bar / Math.max(1, opts.nBarsTotal - 1)) * (pathR - TAPE_PAD_L);

  const hline = (x1: number, x2: number, yy: number): void => {
    const yr = Math.round(yy) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x1, yr);
    ctx.lineTo(x2, yr);
    ctx.stroke();
  };

  // --- profile of the revealed bars (1-pt rows per the panel head) ---------
  const profile = buildProfile(bars, opts.rowStep);
  let maxVol = 0;
  for (const v of profile.rows) if (v > maxVol) maxVol = v;
  if (maxVol <= 0) maxVol = 1;
  const pocPrice = rowToPrice(profile, profile.poc);
  const pocY = y(pocPrice);

  // --- gridlines + right axis labels (skip collisions with the POC label) --
  ctx.strokeStyle = theme.hairline;
  ctx.lineWidth = 1;
  ctx.font = `11px ${theme.fontMono}`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'alphabetic';
  for (let k = 0; k <= 4; k++) {
    const price = pMin + ((pMax - pMin) * k) / 4;
    const yy = y(price);
    ctx.globalAlpha = 0.8;
    hline(TAPE_PAD_L, gridR, yy);
    ctx.globalAlpha = 1;
    if (Math.abs(yy - pocY) < 14) continue;
    ctx.fillStyle = theme.inkMuted;
    ctx.fillText(price.toFixed(2), cssW - 10, yy + 3.5);
  }

  // --- label ledger: floating text never collides or clips (see helper) ----
  const ledger = new LabelLedger();
  const labelBounds: LabelRect = {
    x: 2,
    y: padT + 2,
    w: pathR - 2 - 2,
    h: cssH - padB - 2 - (padT + 2),
  };
  // pin circles are fixed geometry — claim them up front so level labels
  // placed first still route around them
  const pinPos = (opts.pins ?? []).map((pin) => ({
    pin,
    cx: x(pin.bar),
    // keep the pin + its label clear of the panel head at the top
    cy: Math.max(padT + 12, y(pin.price) - 13),
  }));
  for (const p of pinPos) ledger.claim({ x: p.cx - 9.5, y: p.cy - 9.5, w: 19, h: 19 });
  // the price path is fixed geometry too: a candidate label spot that the
  // path runs through is "blocked" (softer than a claim — used when free
  // alternatives exist, relaxed otherwise)
  const pathBlocked = (r: LabelRect): boolean => {
    for (let t = 0; t < bars.length; t++) {
      const px = x(t);
      if (px < r.x - 2) continue;
      if (px > r.x + r.w + 2) break;
      const py = y(bars[t].c);
      if (py >= r.y - 2 && py <= r.y + r.h + 2) return true;
    }
    return false;
  };

  // --- pVAH reference (dashed blue, tagged — never color-alone) ------------
  if (opts.pVah != null && pVahInDomain) {
    const vy = y(opts.pVah);
    ctx.strokeStyle = theme.volBlue;
    ctx.globalAlpha = 0.45;
    ctx.setLineDash([3, 4]);
    hline(TAPE_PAD_L, pathR, vy);
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    ctx.fillStyle = theme.volBlue;
    ctx.font = `10px ${theme.fontMono}`;
    const w = ctx.measureText('pVAH').width;
    ledger.claim({ x: pathR - 4 - w, y: vy - 13, w, h: 10 });
    ctx.fillText('pVAH', pathR - 4, vy - 5);
  } else if (opts.pVah != null) {
    // far below the tape: bottom-edge tag with the distance kept honest
    ctx.fillStyle = theme.volBlue;
    ctx.globalAlpha = 0.8;
    ctx.font = `10px ${theme.fontMono}`;
    ctx.textAlign = 'left';
    const tag = `pVAH ${opts.pVah.toFixed(2)} ↓`;
    ledger.claim({ x: TAPE_PAD_L + 2, y: padT + plotH - 13, w: ctx.measureText(tag).width, h: 10 });
    ctx.fillText(tag, TAPE_PAD_L + 2, padT + plotH - 5);
    ctx.textAlign = 'right';
    ctx.globalAlpha = 1;
  }

  // --- engine-computed levels (entry/stop/target), tagged ------------------
  for (const lv of opts.levels ?? []) {
    const ly = y(lv.price);
    const color = toneColor(theme, lv.tone);
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.6;
    ctx.setLineDash([4, 3]);
    hline(TAPE_PAD_L, pathR, ly);
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    ctx.font = `600 9px ${theme.fontMono}`;
    const text = `${lv.label} ${lv.price.toFixed(2)}`;
    const w = ctx.measureText(text).width;
    // preferred side first (mockup behavior), then the flip, then the right
    // end of the line — the ledger takes the first spot that doesn't collide
    const above = ly - 13;
    const below = ly + 3;
    const pref = lv.labelBelow ? below : above;
    const alt = lv.labelBelow ? above : below;
    const r = ledger.place(
      [
        { x: TAPE_PAD_L + 2, y: pref },
        { x: TAPE_PAD_L + 2, y: alt },
        { x: pathR - 6 - w, y: pref },
        { x: pathR - 6 - w, y: alt },
      ],
      w,
      10,
      labelBounds,
      pathBlocked,
    );
    ctx.textAlign = 'left';
    ctx.fillText(text, r.x, r.y + 9);
    ctx.textAlign = 'right';
  }

  // --- price path (per-bar closes) ------------------------------------------
  ctx.strokeStyle = theme.inkBody;
  ctx.lineWidth = 1.5;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(x(0), y(bars[0].c));
  for (let t = 1; t < bars.length; t++) ctx.lineTo(x(t), y(bars[t].c));
  ctx.stroke();
  ctx.lineWidth = 1;

  // --- developing profile at the right edge ---------------------------------
  const rowPx = (opts.rowStep / (pMax - pMin)) * plotH;
  const barH = Math.max(2, Math.min(rowPx - 1.5, rowPx * 0.8));
  for (let r = 0; r < profile.rows.length; r++) {
    const w = profile.rows[r] / maxVol;
    const bw = Math.max(6, w * (TAPE_PROF_W - 6));
    const cy = y(rowToPrice(profile, r));
    ctx.fillStyle = r === profile.poc ? theme.pocOrange : w >= 0.4 ? theme.volBlue : theme.volBlueDim;
    ctx.fillRect(gridR - bw, cy - barH / 2, bw, barH);
  }

  // --- POC axis price + tag (color always with text, mockup rule) ----------
  ctx.fillStyle = theme.pocOrange;
  ctx.font = `600 11px ${theme.fontMono}`;
  ctx.fillText(pocPrice.toFixed(2), cssW - 10, pocY + 3.5);
  ctx.font = `10px ${theme.fontMono}`;
  const pocW = Math.max(6, (profile.rows[profile.poc] / maxVol) * (TAPE_PROF_W - 6));
  ctx.fillText('POC', gridR - pocW - 6, pocY + 3.5);

  // --- pins (past calls) ------------------------------------------------------
  for (const { pin, cx, cy } of pinPos) {
    const color = toneColor(theme, pin.tone);
    ctx.beginPath();
    ctx.arc(cx, cy, 8.5, 0, Math.PI * 2);
    ctx.fillStyle = theme.surface1;
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.25;
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.fillStyle = color;
    ctx.font = `700 10px ${theme.fontMono}`;
    ctx.textAlign = 'center';
    ctx.fillText(pin.glyph, cx, cy + 3.5);
    ctx.font = `10px ${theme.fontMono}`;
    const w = ctx.measureText(pin.label).width;
    // centered above the pin when free, else below, else beside it — never
    // over a level tag and never clipped at an edge
    const r = ledger.place(
      [
        { x: cx - w / 2, y: cy - 23 },
        { x: cx - w / 2, y: cy + 13 },
        { x: cx + 12, y: cy - 5 },
        { x: cx - 12 - w, y: cy - 5 },
      ],
      w,
      10,
      labelBounds,
      pathBlocked,
    );
    ctx.textAlign = 'left';
    ctx.fillText(pin.label, r.x, r.y + 9);
  }
  ctx.textAlign = 'right';

  // --- last-price dot ---------------------------------------------------------
  const last = bars[bars.length - 1];
  ctx.fillStyle = theme.highlight;
  ctx.beginPath();
  ctx.arc(x(last.t), y(last.c), 2.5, 0, Math.PI * 2);
  ctx.fill();

  // --- caption strip ----------------------------------------------------------
  if (opts.caption) {
    ctx.fillStyle = theme.inkMuted;
    ctx.font = `11px ${theme.fontMono}`;
    ctx.textAlign = 'left';
    ctx.fillText(opts.caption, TAPE_PAD_L + 2, cssH - 11);
    ctx.textAlign = 'right';
  }

  return now() - t0;
}
