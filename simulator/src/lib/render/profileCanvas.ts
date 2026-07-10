/* ============================================================================
   @render/profileCanvas — the Canvas 2D chart layer of the drill loop.

   Geometry is a 1:1 port of the mockup SVG in
   /docs/simulator/design/mockups/screen-drill-loop.html (padT/padB 14,
   axisW 64, barX 16, per-row bars with 3.5px gaps, VA band + hairlines,
   POC tag + price + dashed leader, axis labels every 4th row skipping the
   POC callout). All colors/fonts come from tokens.css via readChartTheme
   (getComputedStyle) so chart and DOM can never diverge (tokens.css header).

   ACCURACY RULE (GDD §9): everything drawn here is a Profile/Bar computed by
   @core — this module positions pixels, it never computes POC/VA/HVN/LVN.

   The pure layout helpers (profileLayout / rowY / rowAtY / priceAxisRows) are
   exported for unit tests; only render*() touches a canvas. Rendering is
   budgeted <8ms per full profile redraw — render*() returns the measured ms
   and the drill screen logs it (also exposed to e2e via window.__auction).
   ========================================================================== */

import type { Bar, Profile } from '../types';
import { rowToPrice } from '../core/profile';

/** Colors + fonts the renderer needs, resolved from tokens.css custom props. */
export interface ChartTheme {
  surface1: string;
  hairline: string;
  gridline: string;
  inkMuted: string;
  inkGhost: string;
  volBlue: string;
  volBlueDim: string;
  pocOrange: string;
  highlight: string;
  vaBand: string;
  vaEdge: string;
  buyGreen: string;
  sellRed: string;
  fontMono: string;
}

/** Fallbacks equal to tokens.css values (used when a var is unset in tests). */
const THEME_FALLBACK: ChartTheme = {
  surface1: '#1a1a19',
  hairline: '#2c2c2a',
  gridline: '#232322',
  inkMuted: '#898781',
  inkGhost: '#55534d',
  volBlue: '#3987e5',
  volBlueDim: '#2a5a8f',
  pocOrange: '#d95926',
  highlight: '#eda100',
  vaBand: 'rgba(57, 135, 229, .06)',
  vaEdge: 'rgba(57, 135, 229, .30)',
  buyGreen: '#1baf7a',
  sellRed: '#e66767',
  fontMono: "'JetBrains Mono', monospace",
};

/** Read the chart theme from the computed styles of any mounted element. */
export function readChartTheme(el: Element): ChartTheme {
  const cs = getComputedStyle(el);
  const v = (name: string, fallback: string): string => {
    const val = cs.getPropertyValue(name).trim();
    return val === '' ? fallback : val;
  };
  return {
    surface1: v('--surface-1', THEME_FALLBACK.surface1),
    hairline: v('--hairline', THEME_FALLBACK.hairline),
    gridline: v('--gridline', THEME_FALLBACK.gridline),
    inkMuted: v('--ink-muted', THEME_FALLBACK.inkMuted),
    inkGhost: v('--ink-ghost', THEME_FALLBACK.inkGhost),
    volBlue: v('--vol-blue', THEME_FALLBACK.volBlue),
    volBlueDim: v('--vol-blue-dim', THEME_FALLBACK.volBlueDim),
    pocOrange: v('--poc-orange', THEME_FALLBACK.pocOrange),
    highlight: v('--highlight', THEME_FALLBACK.highlight),
    vaBand: v('--va-band', THEME_FALLBACK.vaBand),
    vaEdge: v('--va-edge', THEME_FALLBACK.vaEdge),
    buyGreen: v('--buy-green', THEME_FALLBACK.buyGreen),
    sellRed: v('--sell-red', THEME_FALLBACK.sellRed),
    fontMono: v('--font-mono', THEME_FALLBACK.fontMono),
  };
}

/* ----------------------------------------------------------------------------
   Pure layout (mockup geometry) — unit-tested, no DOM
   -------------------------------------------------------------------------- */

/** Mockup constants (screen-drill-loop.html chart script). */
export const PAD_T = 14;
export const PAD_B = 14;
export const AXIS_W = 64;
export const BAR_X = 16;
export const BAR_MIN_LEN = 7;
/** Snap-to-row tolerance in CSS px (GDD §2 / design system §6.3). */
export const SNAP_PX = 22;

export interface ProfileLayout {
  /** Canvas CSS width/height. */
  w: number;
  h: number;
  /** Row count. */
  n: number;
  /** Height of one row slot. */
  rowH: number;
  /** Height of one bar (rowH minus gap, min 5). */
  barH: number;
  /** Max bar length in px. */
  maxLen: number;
  /** Right edge of the VA band / gridlines (left edge of the axis gutter). */
  bandR: number;
}

/**
 * Layout for an nRows-row profile in a w×h CSS-pixel canvas. Screen rows are
 * indexed 0 = TOP (highest price); profile rows are 0 = lowest price — use
 * screenRow()/profileRow() to flip.
 */
export function profileLayout(nRows: number, w: number, h: number): ProfileLayout {
  const rowH = (h - PAD_T - PAD_B) / nRows;
  return {
    w,
    h,
    n: nRows,
    rowH,
    barH: Math.max(5, rowH - 3.5),
    maxLen: w - AXIS_W - BAR_X - 12,
    bandR: w - AXIS_W + 8,
  };
}

/** Screen index (0 = top) of a profile row (0 = lowest price). Involution. */
export function screenRow(profileRow: number, nRows: number): number {
  return nRows - 1 - profileRow;
}

/** Top y of a screen row slot. */
export function rowY(layout: ProfileLayout, screenIdx: number): number {
  return PAD_T + screenIdx * layout.rowH;
}

/** Center y of a PROFILE row. */
export function rowCenterY(layout: ProfileLayout, profileRow: number): number {
  return rowY(layout, screenRow(profileRow, layout.n)) + layout.rowH / 2;
}

/**
 * Snap a canvas-local y (CSS px) to the nearest PROFILE row center within
 * SNAP_PX; null when the tap is farther than the tolerance from every row
 * (only possible in the top/bottom padding when rows are sparse).
 */
export function rowAtY(layout: ProfileLayout, y: number): number | null {
  const i = Math.round((y - PAD_T) / layout.rowH - 0.5);
  const clamped = Math.min(layout.n - 1, Math.max(0, i));
  const center = rowY(layout, clamped) + layout.rowH / 2;
  if (Math.abs(y - center) > Math.max(SNAP_PX, layout.rowH / 2)) return null;
  return screenRow(clamped, layout.n);
}

/**
 * Screen rows that get an axis price label: every 4th, skipping rows within
 * 3 of the POC callout (mockup rule) so labels never collide with it.
 */
export function priceAxisRows(nRows: number, pocScreenRow: number | null): number[] {
  const out: number[] = [];
  for (let i = 0; i < nRows; i += 4) {
    if (pocScreenRow !== null && Math.abs(i - pocScreenRow) < 3) continue;
    out.push(i);
  }
  return out;
}

/* ----------------------------------------------------------------------------
   Profile renderer
   -------------------------------------------------------------------------- */

/** A row marker drawn at verdict time (snap drills: pick vs truth). */
export interface RowMarker {
  /** PROFILE row index (0 = lowest price). */
  row: number;
  /** Who/what the marker shows — colors + glyphs are CVD-redundant. */
  kind: 'you-wrong' | 'you-right' | 'truth';
  /** Right-side tag text (e.g. "YOU", "VAH"). */
  label: string;
}

export interface ProfileRenderOptions {
  theme: ChartTheme;
  /** Draw the POC row orange + tag + leader (default true). Drill A "tap the
      POC" items hide it — the answer must never be visible pre-commit. */
  showPoc?: boolean;
  /** Draw the VA band + VAH/VAL hairlines + tags (default true). */
  showVa?: boolean;
  /** PROFILE row to halo with --highlight (Drill B extreme). */
  highlightRow?: number | null;
  /** Verdict markers (drawn last). */
  markers?: RowMarker[];
}

function setupCanvas(
  canvas: HTMLCanvasElement,
  cssW: number,
  cssH: number,
): CanvasRenderingContext2D {
  const dpr = (globalThis as { devicePixelRatio?: number }).devicePixelRatio ?? 1;
  const pw = Math.max(1, Math.round(cssW * dpr));
  const ph = Math.max(1, Math.round(cssH * dpr));
  if (canvas.width !== pw) canvas.width = pw;
  if (canvas.height !== ph) canvas.height = ph;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('profileCanvas: no 2d context');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  return ctx;
}

const now = (): number =>
  typeof performance !== 'undefined' ? performance.now() : Date.now();

/** Crisp 1px horizontal line (device-pixel aligned via the 0.5 offset). */
function hline(ctx: CanvasRenderingContext2D, x1: number, x2: number, y: number): void {
  const yy = Math.round(y) + 0.5;
  ctx.beginPath();
  ctx.moveTo(x1, yy);
  ctx.lineTo(x2, yy);
  ctx.stroke();
}

/**
 * Render a full volume profile. Returns the measured render time in ms
 * (budget: <8ms on a 60–120-row profile — GDD §2 / task spec).
 */
export function renderProfile(
  canvas: HTMLCanvasElement,
  profile: Profile,
  cssW: number,
  cssH: number,
  opts: ProfileRenderOptions,
): number {
  const t0 = now();
  const { theme } = opts;
  const showPoc = opts.showPoc ?? true;
  const showVa = opts.showVa ?? true;
  const ctx = setupCanvas(canvas, cssW, cssH);
  const rows = profile.rows;
  const n = rows.length;
  const L = profileLayout(n, cssW, cssH);
  let maxVol = 0;
  for (const v of rows) if (v > maxVol) maxVol = v;
  if (maxVol <= 0) maxVol = 1;

  const pocS = screenRow(profile.poc, n);
  const vahS = screenRow(profile.vah, n); // smaller screen index (top)
  const valS = screenRow(profile.val, n);
  const axisRows = priceAxisRows(n, showPoc ? pocS : null);

  // --- gridlines at labeled ticks -------------------------------------------
  ctx.strokeStyle = theme.hairline;
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.7;
  for (const i of axisRows) hline(ctx, 0, L.bandR, rowY(L, i) + L.rowH / 2);
  ctx.globalAlpha = 1;

  // --- VA band + edges (behind bars) ----------------------------------------
  if (showVa) {
    ctx.fillStyle = theme.vaBand;
    ctx.fillRect(0, rowY(L, vahS), L.bandR, rowY(L, valS + 1) - rowY(L, vahS));
    ctx.strokeStyle = theme.vaEdge;
    hline(ctx, 0, L.bandR, rowY(L, vahS));
    hline(ctx, 0, L.bandR, rowY(L, valS + 1));
  }

  // --- highlight halo (Drill B extreme) — glyph-paired via the ▸ tag --------
  if (opts.highlightRow != null) {
    const hs = screenRow(opts.highlightRow, n);
    ctx.fillStyle = theme.highlight;
    ctx.globalAlpha = 0.14;
    ctx.fillRect(0, rowY(L, hs), L.bandR, L.rowH);
    ctx.globalAlpha = 1;
  }

  // --- bars -------------------------------------------------------------------
  for (let r = 0; r < n; r++) {
    const s = screenRow(r, n);
    const inVa = r >= profile.val && r <= profile.vah;
    ctx.fillStyle =
      showPoc && r === profile.poc
        ? theme.pocOrange
        : inVa && showVa
          ? theme.volBlue
          : showVa
            ? theme.volBlueDim
            : theme.volBlue;
    const len = Math.max(BAR_MIN_LEN, (rows[r] / maxVol) * L.maxLen);
    ctx.fillRect(BAR_X, rowY(L, s) + 1.75, len, L.barH);
  }

  // --- axis price labels -------------------------------------------------------
  ctx.font = `11px ${theme.fontMono}`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = theme.inkMuted;
  for (const i of axisRows) {
    const price = rowToPrice(profile, screenRow(i, n));
    ctx.fillText(price.toFixed(2), cssW - 10, rowY(L, i) + L.rowH / 2 + 3.5);
  }

  // --- POC callout: tag + price + dashed leader (never color-alone) ----------
  if (showPoc) {
    const pocY = rowY(L, pocS) + L.rowH / 2;
    ctx.fillStyle = theme.pocOrange;
    ctx.font = `600 9px ${theme.fontMono}`;
    ctx.fillText('POC', cssW - 10, pocY - 8);
    ctx.font = `600 11px ${theme.fontMono}`;
    ctx.fillText(rowToPrice(profile, profile.poc).toFixed(2), cssW - 10, pocY + 4.5);
    ctx.strokeStyle = theme.pocOrange;
    ctx.globalAlpha = 0.45;
    ctx.setLineDash([2, 3]);
    const pocLen = Math.max(BAR_MIN_LEN, (rows[profile.poc] / maxVol) * L.maxLen);
    hline(ctx, BAR_X + pocLen + 4, L.bandR, pocY);
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  // --- VAH/VAL tags -------------------------------------------------------------
  if (showVa) {
    ctx.fillStyle = theme.volBlue;
    ctx.font = `10px ${theme.fontMono}`;
    ctx.fillText('VAH', L.bandR - 4, rowY(L, vahS) - 5);
    ctx.fillText('VAL', L.bandR - 4, rowY(L, valS + 1) + 12);
  }

  // --- highlight tag (after bars so it reads) ---------------------------------
  if (opts.highlightRow != null) {
    const hs = screenRow(opts.highlightRow, n);
    ctx.fillStyle = theme.highlight;
    ctx.font = `600 10px ${theme.fontMono}`;
    ctx.fillText('▸ HERE', L.bandR - 4, rowY(L, hs) + L.rowH / 2 + 3.5);
  }

  // --- verdict markers ----------------------------------------------------------
  let lastLabelY = -Infinity;
  for (const m of opts.markers ?? []) {
    const y = rowCenterY(L, m.row);
    const color =
      m.kind === 'truth' ? theme.buyGreen : m.kind === 'you-right' ? theme.buyGreen : theme.sellRed;
    ctx.strokeStyle = color;
    ctx.globalAlpha = m.kind === 'truth' ? 0.9 : 0.75;
    hline(ctx, 0, L.bandR, y);
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    ctx.font = `700 10px ${theme.fontMono}`;
    const glyph = m.kind === 'you-wrong' ? '✗ ' : '✓ ';
    // adjacent markers: drop the second label below its line so they never collide
    const labelY = Math.abs(y - 4 - lastLabelY) < 12 ? y + 11 : y - 4;
    ctx.fillText(`${glyph}${m.label}`, L.bandR - 4, labelY);
    lastLabelY = labelY;
  }

  return now() - t0;
}

/* ----------------------------------------------------------------------------
   Opening-line renderer (Drill F — first 90 minutes as a price line)
   -------------------------------------------------------------------------- */

export interface LineRenderOptions {
  theme: ChartTheme;
  /** Price of the session open — drawn as a labeled reference hairline. */
  openPrice: number;
  /** Minutes per vertical divider (default 30 = one TPO bracket). */
  bracketMinutes?: number;
}

export interface LineLayout {
  padL: number;
  padR: number;
  padT: number;
  padB: number;
  plotW: number;
  plotH: number;
  pMin: number;
  pMax: number;
}

/** Pure layout for the opening line chart (exported for tests). */
export function lineLayout(bars: Bar[], w: number, h: number): LineLayout {
  let pMin = Infinity;
  let pMax = -Infinity;
  for (const b of bars) {
    if (b.l < pMin) pMin = b.l;
    if (b.h > pMax) pMax = b.h;
  }
  const span = Math.max(1e-9, pMax - pMin);
  pMin -= span * 0.06;
  pMax += span * 0.06;
  const padL = 8;
  const padR = AXIS_W;
  const padT = 14;
  const padB = 18;
  return { padL, padR, padT, padB, plotW: w - padL - padR, plotH: h - padT - padB, pMin, pMax };
}

export function lineX(L: LineLayout, t: number, nBars: number): number {
  return L.padL + (t / Math.max(1, nBars - 1)) * L.plotW;
}

export function lineY(L: LineLayout, price: number): number {
  return L.padT + (1 - (price - L.pMin) / (L.pMax - L.pMin)) * L.plotH;
}

/**
 * Render the first-N-minutes tape as a close line with bracket dividers and
 * the open reference. Returns measured ms.
 */
export function renderOpeningLine(
  canvas: HTMLCanvasElement,
  bars: Bar[],
  cssW: number,
  cssH: number,
  opts: LineRenderOptions,
): number {
  const t0 = now();
  const { theme } = opts;
  const ctx = setupCanvas(canvas, cssW, cssH);
  const n = bars.length;
  if (n < 2) return now() - t0;
  const L = lineLayout(bars, cssW, cssH);
  const bracketMin = opts.bracketMinutes ?? 30;

  // --- horizontal gridlines + right price labels (4 ticks) --------------------
  // (tick labels within 16px of the OPEN callout are skipped — no collisions)
  const openYRef = lineY(L, opts.openPrice);
  ctx.font = `11px ${theme.fontMono}`;
  ctx.textAlign = 'right';
  ctx.strokeStyle = theme.hairline;
  ctx.lineWidth = 1;
  for (let k = 0; k <= 3; k++) {
    const price = L.pMin + ((L.pMax - L.pMin) * k) / 3;
    const y = lineY(L, price);
    ctx.globalAlpha = 0.7;
    hline(ctx, 0, cssW - AXIS_W + 8, y);
    ctx.globalAlpha = 1;
    if (Math.abs(y - openYRef) < 16) continue;
    ctx.fillStyle = theme.inkMuted;
    ctx.fillText(price.toFixed(2), cssW - 10, y + 3.5);
  }

  // --- bracket dividers + letters (A, B, C…) ----------------------------------
  ctx.textAlign = 'center';
  ctx.font = `10px ${theme.fontMono}`;
  for (let m = bracketMin; m < n; m += bracketMin) {
    const x = Math.round(lineX(L, m, n)) + 0.5;
    ctx.strokeStyle = theme.gridline;
    ctx.beginPath();
    ctx.moveTo(x, L.padT);
    ctx.lineTo(x, L.padT + L.plotH);
    ctx.stroke();
  }
  ctx.fillStyle = theme.inkGhost;
  for (let br = 0; br * bracketMin < n; br++) {
    const x0 = lineX(L, br * bracketMin, n);
    const x1 = lineX(L, Math.min(n - 1, (br + 1) * bracketMin), n);
    ctx.fillText(String.fromCharCode(65 + br), (x0 + x1) / 2, cssH - 5);
  }

  // --- open reference hairline (ghost, dashed, tagged) ------------------------
  const openY = lineY(L, opts.openPrice);
  ctx.strokeStyle = theme.inkGhost;
  ctx.setLineDash([3, 3]);
  hline(ctx, 0, cssW - AXIS_W + 8, openY);
  ctx.setLineDash([]);
  ctx.textAlign = 'right';
  ctx.fillStyle = theme.inkMuted;
  ctx.font = `600 9px ${theme.fontMono}`;
  ctx.fillText('OPEN', cssW - 10, openY - 3);
  ctx.font = `11px ${theme.fontMono}`;
  ctx.fillText(opts.openPrice.toFixed(2), cssW - 10, openY + 11);

  // --- per-minute range whiskers (h–l) under the close line -------------------
  ctx.strokeStyle = theme.volBlueDim;
  ctx.globalAlpha = 0.55;
  ctx.beginPath();
  for (let t = 0; t < n; t++) {
    const x = lineX(L, t, n);
    ctx.moveTo(x, lineY(L, bars[t].h));
    ctx.lineTo(x, lineY(L, bars[t].l));
  }
  ctx.stroke();
  ctx.globalAlpha = 1;

  // --- close line ---------------------------------------------------------------
  ctx.strokeStyle = theme.volBlue;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(lineX(L, 0, n), lineY(L, bars[0].c));
  for (let t = 1; t < n; t++) ctx.lineTo(lineX(L, t, n), lineY(L, bars[t].c));
  ctx.stroke();
  ctx.lineWidth = 1;

  // --- last-price dot ------------------------------------------------------------
  const lx = lineX(L, n - 1, n);
  const ly = lineY(L, bars[n - 1].c);
  ctx.fillStyle = theme.volBlue;
  ctx.beginPath();
  ctx.arc(lx, ly, 2.5, 0, Math.PI * 2);
  ctx.fill();

  return now() - t0;
}
