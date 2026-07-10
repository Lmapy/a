/* ============================================================================
   @render/statsCanvas — Canvas 2D renderers for the Stats dashboard:
   · the calibration reliability plot (design-system §3.8): 50–100% on x,
     y extends below 50 when an observed hit rate demands it (no honest dot
     is ever pinned to the axis floor), gridlines, ghost perfect-calibration
     diagonal, CI whiskers ∝ 1/√n, dots area-sized by n, worst-bin
     annotation, dot-size key
   · the R-multiple histogram: 0.5R bins, wins/losses by --vol-blue ramp,
     zero divider, avg marker in --highlight

   Geometry is a 1:1 port of the SVG in
   /docs/simulator/design/mockups/screen-dashboard.html. All colors resolve
   from tokens.css via readStatsTheme (getComputedStyle) — zero hex here.

   ACCURACY RULE (GDD §9): this module positions pixels; every plotted value
   (bucket means, hit rates, n, R bins, averages) arrives pre-computed by
   @stats/model from the persisted ledger.
   ========================================================================== */

import { readChartTheme } from './profileCanvas';
import type { ChartTheme } from './profileCanvas';
import type { CalibrationView, RHistogram } from '../stats/model';

/** ChartTheme plus the two ink tiers the dashboard annotations need. */
export interface StatsTheme extends ChartTheme {
  inkBody: string;
  inkHero: string;
}

/** Resolve the dashboard chart theme from tokens.css custom properties. */
export function readStatsTheme(el: Element): StatsTheme {
  const cs = getComputedStyle(el);
  const v = (name: string, fallback: string): string => {
    const val = cs.getPropertyValue(name).trim();
    return val === '' ? fallback : val;
  };
  return {
    ...readChartTheme(el),
    inkBody: v('--ink-body', '#c3c2b7'),
    inkHero: v('--ink-hero', '#ffffff'),
  };
}

function prep(canvas: HTMLCanvasElement): { ctx: CanvasRenderingContext2D; w: number; h: number } | null {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (w === 0 || h === 0) return null;
  const dpr = typeof devicePixelRatio === 'number' ? devicePixelRatio : 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

/* ----------------------------------------------------------------------------
   Calibration reliability plot
   -------------------------------------------------------------------------- */

/** Draw the reliability curve. Dots/worst arrive from calibrationView(). */
export function renderCalibration(
  canvas: HTMLCanvasElement,
  theme: StatsTheme,
  view: CalibrationView,
): void {
  const p = prep(canvas);
  if (!p) return;
  const { ctx, w, h } = p;

  const padL = 40;
  const padR = 20;
  const padT = 10;
  const padB = 32;
  const pw = w - padL - padR;
  const ph = h - padT - padB;
  // The mockup square is 50–100 on both axes, but an observed hit rate may
  // fall BELOW 50% (e.g. a guess bucket at 0% observed): the y-domain
  // extends down to the lowest plotted dot so no honest value gets pinned to
  // the axis floor and misread as 50%.
  const allDots = view.dots.filter((d) => d.saidPct >= 50);
  const yLo = Math.min(50, ...allDots.map((d) => Math.floor(d.hapPct / 10) * 10));
  const X = (v: number): number => padL + ((v - 50) / 50) * pw;
  const Y = (v: number): number => padT + ph - ((v - yLo) / (100 - yLo)) * ph;

  const mono10 = `10px ${theme.fontMono}`;
  const mono11 = `11px ${theme.fontMono}`;

  // gridlines + tick labels: x 50..100, y yLo..100
  ctx.lineWidth = 1;
  ctx.strokeStyle = theme.gridline;
  ctx.fillStyle = theme.inkMuted;
  ctx.font = mono11;
  for (let v = yLo; v <= 100; v += 10) {
    ctx.beginPath();
    ctx.moveTo(X(50), Y(v) + 0.5);
    ctx.lineTo(X(100), Y(v) + 0.5);
    ctx.stroke();
    ctx.textAlign = 'right';
    ctx.fillText(String(v), X(50) - 8, Y(v) + 4);
  }
  for (let v = 50; v <= 100; v += 10) {
    ctx.beginPath();
    ctx.moveTo(X(v) + 0.5, Y(yLo));
    ctx.lineTo(X(v) + 0.5, Y(100));
    ctx.stroke();
    ctx.textAlign = 'center';
    ctx.fillText(String(v), X(v), Y(yLo) + 18);
  }

  // axis titles
  ctx.font = mono10;
  ctx.textAlign = 'center';
  ctx.fillText('you said %', X(75), h - 4);
  ctx.save();
  ctx.translate(12, Y(75));
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('it happened %', 0, 0);
  ctx.restore();

  // perfect-calibration diagonal + rotated ghost label
  ctx.strokeStyle = theme.inkGhost;
  ctx.beginPath();
  ctx.moveTo(X(50), Y(50));
  ctx.lineTo(X(100), Y(100));
  ctx.stroke();
  ctx.save();
  ctx.translate(X(89), Y(93.5));
  ctx.rotate(-Math.atan((Y(50) - Y(100)) / (X(100) - X(50))));
  ctx.fillStyle = theme.inkGhost;
  ctx.textAlign = 'center';
  ctx.fillText('perfect calibration', 0, 0);
  ctx.restore();

  const dots = allDots;
  if (dots.length > 0) {
    // connecting line under the dots
    ctx.strokeStyle = theme.volBlue;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.globalAlpha = 0.55;
    ctx.beginPath();
    dots.forEach((d, i) => {
      const x = X(d.saidPct);
      const y = Y(d.hapPct);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.globalAlpha = 1;

    // CI whiskers — wider when n is small
    ctx.globalAlpha = 0.28;
    for (const d of dots) {
      const ci = 40 / Math.sqrt(d.n);
      ctx.beginPath();
      ctx.moveTo(X(d.saidPct), Y(Math.min(100, d.hapPct + ci)));
      ctx.lineTo(X(d.saidPct), Y(Math.max(yLo, d.hapPct - ci)));
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // dots sized by n, surface ring
    for (const d of dots) {
      const r = 3.2 + Math.sqrt(d.n) / 2.4;
      ctx.beginPath();
      ctx.arc(X(d.saidPct), Y(d.hapPct), r, 0, Math.PI * 2);
      ctx.fillStyle = theme.volBlue;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = theme.surface1;
      ctx.stroke();
    }

    // annotate the worst qualified bin, below its dot
    if (view.worst) {
      const d = view.worst;
      const gap = d.hapPct - d.saidPct;
      const x = Math.min(X(92), Math.max(X(58), X(d.saidPct)));
      const y = Math.min(Y(yLo + 2), Y(d.hapPct) + 26);
      ctx.font = mono10;
      ctx.textAlign = 'center';
      ctx.fillStyle = theme.inkBody;
      ctx.fillText(`said ${d.saidPct} · happened ${d.hapPct}`, x, y);
      ctx.fillStyle = theme.inkMuted;
      ctx.fillText(`gap ${gap > 0 ? '+' : '−'}${Math.abs(gap)} · n ${d.n}`, x, y + 14);
    }
  }

  // dot-size key, empty region top-left above the diagonal
  const kx = X(53.5);
  const ky = Y(95.5);
  ctx.strokeStyle = theme.inkGhost;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(kx, ky, 3.5, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(kx, ky, 7, 0, Math.PI * 2);
  ctx.stroke();
  ctx.font = mono10;
  ctx.textAlign = 'left';
  ctx.fillStyle = theme.inkGhost;
  ctx.fillText('dot area = n', kx + 14, ky + 4);
}

/* ----------------------------------------------------------------------------
   R-multiple histogram
   -------------------------------------------------------------------------- */

/** Draw the R histogram. Bins/avg arrive from rHistogram(). */
export function renderRHistogram(
  canvas: HTMLCanvasElement,
  theme: StatsTheme,
  hist: RHistogram,
): void {
  const p = prep(canvas);
  if (!p) return;
  const { ctx, w, h } = p;

  const { lo, step, counts } = hist;
  const padL = 8;
  const padR = 8;
  const padT = 6;
  const padB = 26;
  const pw = w - padL - padR;
  const ph = h - padT - padB;
  const bw = pw / counts.length;
  const span = counts.length * step;
  const X = (r: number): number => padL + ((r - lo) / span) * pw;

  const rawMax = Math.max(1, ...counts);
  const gridStep = rawMax <= 12 ? 5 : rawMax <= 30 ? 10 : rawMax <= 60 ? 20 : 50;
  const maxC = Math.ceil((rawMax * 1.1) / gridStep) * gridStep;
  const Y = (c: number): number => padT + ph * (1 - c / maxC);

  const mono10 = `10px ${theme.fontMono}`;
  const mono11 = `11px ${theme.fontMono}`;

  // gridlines + right-edge count labels
  ctx.lineWidth = 1;
  ctx.font = mono10;
  for (let c = gridStep; c < maxC; c += gridStep) {
    ctx.strokeStyle = theme.gridline;
    ctx.beginPath();
    ctx.moveTo(padL, Y(c) + 0.5);
    ctx.lineTo(w - padR, Y(c) + 0.5);
    ctx.stroke();
    ctx.fillStyle = theme.inkGhost;
    ctx.textAlign = 'right';
    ctx.fillText(String(c), w - padR, Y(c) - 3);
  }

  // bars — 2px surface gap, rounded data-end, squared baseline
  counts.forEach((c, idx) => {
    if (c === 0) return;
    const r0 = lo + idx * step;
    const x = X(r0) + 1;
    const bwPx = bw - 2;
    const y = Y(c);
    const hh = padT + ph - y;
    ctx.fillStyle = r0 >= 0 ? theme.volBlue : theme.volBlueDim;
    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') {
      ctx.roundRect(x, y, bwPx, hh, 3);
    } else {
      ctx.rect(x, y, bwPx, hh);
    }
    ctx.fill();
    if (hh > 6) ctx.fillRect(x, y + hh - 4, bwPx, 4);
  });

  // baseline + zero divider
  ctx.strokeStyle = theme.hairline;
  ctx.beginPath();
  ctx.moveTo(padL, padT + ph + 0.5);
  ctx.lineTo(w - padR, padT + ph + 0.5);
  ctx.stroke();
  ctx.strokeStyle = theme.inkGhost;
  ctx.beginPath();
  ctx.moveTo(X(0) + 0.5, padT - 2);
  ctx.lineTo(X(0) + 0.5, padT + ph);
  ctx.stroke();

  // avg marker
  if (hist.n > 0) {
    const ax = X(Math.min(lo + span, Math.max(lo, hist.avg)));
    ctx.strokeStyle = theme.highlight;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.9;
    ctx.beginPath();
    ctx.moveTo(ax, padT + 2);
    ctx.lineTo(ax, padT + ph);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.font = mono10;
    ctx.fillStyle = theme.highlight;
    ctx.textAlign = 'left';
    const sign = hist.avg >= 0 ? '+' : '−';
    ctx.fillText(`avg ${sign}${Math.abs(hist.avg).toFixed(2)}R`, ax + 6, padT + 12);
  }

  // x ticks at whole R
  ctx.font = mono11;
  ctx.fillStyle = theme.inkMuted;
  const hiR = lo + span;
  for (let r = Math.ceil(lo); r <= hiR; r++) {
    ctx.textAlign = r === lo ? 'left' : r === hiR ? 'right' : 'center';
    ctx.fillText(`${r > 0 ? '+' : ''}${r}R`, X(r) + (r === lo ? 2 : 0), padT + ph + 16);
  }
}
