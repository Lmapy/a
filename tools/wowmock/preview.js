// Render a wowmock scene (run.lua --dump) to a PNG with headless Chromium.
// This is an approximation for previews: game icons are replaced by emoji
// and the game font by Inter. Layout, sizes and colours come from the addon.
//
//   node tools/wowmock/preview.js scene.json out.png
const fs = require('fs');
const { chromium } = require('playwright');

const [, , input, output] = process.argv;
if (!input || !output) {
  console.error('usage: node tools/wowmock/preview.js scene.json out.png');
  process.exit(1);
}
const scene = JSON.parse(fs.readFileSync(input, 'utf8'));
const W = scene.width;
const H = scene.height;

// Stand-ins for game icons (file ids or paths) used in the test scene.
const ICONS = {
  136241: '⚒️', 136248: '⛏️', 136245: '🎣', 133971: '🍳', 134071: '💎', 136246: '🌿', 134366: '🐺',
  'Interface\\Icons\\Spell_Holy_SealOfSacrifice': '🩹', 'Interface\\Icons\\Trade_BlackSmithing': '⚒️',
  'Interface\\Icons\\Trade_Mining': '⛏️', 'Interface\\Icons\\Trade_Fishing': '🎣', 'Interface\\Icons\\INV_Misc_Food_15': '🍳',
};

const css = (c, alpha = 1) => {
  const [r, g, b, a = 1] = c || [1, 1, 1, 1];
  const to = (v) => Math.round(Math.max(0, Math.min(1, v)) * 255);
  return `rgba(${to(r)},${to(g)},${to(b)},${(a * alpha).toFixed(3)})`;
};
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// WoW colour escapes (|cffRRGGBB ... |r) to spans.
function richText(text) {
  let out = '';
  let open = false;
  const re = /\|c([0-9a-fA-F]{8})|\|r/g;
  let last = 0;
  let match;
  while ((match = re.exec(text))) {
    out += esc(text.slice(last, match.index));
    if (open) out += '</span>';
    open = false;
    if (match[1]) {
      out += `<span style="color:#${match[1].slice(2)}">`;
      open = true;
    }
    last = re.lastIndex;
  }
  out += esc(text.slice(last));
  if (open) out += '</span>';
  return out;
}

let minX = W, minY = H, maxX = 0, maxY = 0;
const grow = (l, t, r, b) => {
  minX = Math.min(minX, l); minY = Math.min(minY, t); maxX = Math.max(maxX, r); maxY = Math.max(maxY, b);
};

const parts = scene.ops.map((op, z) => {
  if (op.type === 'line') {
    const x1 = op.x1, y1 = H - op.y1, x2 = op.x2, y2 = H - op.y2;
    const len = Math.hypot(x2 - x1, y2 - y1);
    const angle = Math.atan2(y2 - y1, x2 - x1);
    grow(Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2));
    return `<div style="position:absolute;z-index:${z};left:${x1}px;top:${y1 - op.thickness / 2}px;width:${len + 0.6}px;height:${op.thickness}px;`
      + `background:${css(op.color, op.alpha)};transform-origin:0 50%;transform:rotate(${angle}rad)"></div>`;
  }
  const { l, r, t, b } = op.rect;
  const left = l, top = H - t, width = r - l, height = t - b;
  if (op.type === 'tex') {
    if (width <= 0.01 || height <= 0.01) return '';
    grow(left, top, left + width, top + height);
    const radius = op.round ? 'border-radius:50%;' : '';
    if (op.color) {
      return `<div style="position:absolute;z-index:${z};left:${left}px;top:${top}px;width:${width}px;height:${height}px;${radius}background:${css(op.color, op.alpha)}"></div>`;
    }
    const icon = ICONS[op.path];
    if (icon === undefined) return '';
    const filter = op.desaturated ? 'filter:grayscale(1);' : '';
    return `<div style="position:absolute;z-index:${z};left:${left}px;top:${top}px;width:${width}px;height:${height}px;${radius}${filter}`
      + `opacity:${op.alpha};background:radial-gradient(circle at 35% 30%,#4a4652,#17151c);display:flex;align-items:center;justify-content:center;`
      + `font-size:${Math.round(height * 0.68)}px;line-height:1;overflow:hidden">${icon}</div>`;
  }
  if (op.type === 'text') {
    const align = (op.justifyH || 'CENTER').toLowerCase();
    const base = `position:absolute;z-index:${z};font-family:Inter,'DejaVu Sans',sans-serif;font-weight:500;font-size:${op.size}px;`
      + `color:${css(op.color, op.alpha)};white-space:nowrap;text-shadow:1px 1px 0 rgba(0,0,0,0.9);line-height:${height}px;height:${height}px;`;
    grow(left, top, left + width, top + height);
    if (op.fixedWidth) {
      return `<div style="${base}left:${left}px;top:${top}px;width:${width}px;text-align:${align};overflow:hidden;text-overflow:ellipsis">${richText(op.text)}</div>`;
    }
    // Auto-sized strings grow away from the point they are anchored by.
    const anchor = (op.anchorH || 'CENTER').toLowerCase();
    const x = anchor === 'right' ? left + width : anchor === 'left' ? left : left + width / 2;
    const shift = anchor === 'right' ? '-100%' : anchor === 'left' ? '0' : '-50%';
    return `<div style="${base}left:${x}px;top:${top}px;transform:translateX(${shift})">${richText(op.text)}</div>`;
  }
  return '';
});

const margin = 26;
const clip = {
  x: Math.max(0, Math.floor(minX - margin)),
  y: Math.max(0, Math.floor(minY - margin)),
  width: Math.min(W, Math.ceil(maxX + margin)) - Math.max(0, Math.floor(minX - margin)),
  height: Math.min(H, Math.ceil(maxY + margin + 18)) - Math.max(0, Math.floor(minY - margin)),
};

const html = `<!DOCTYPE html><html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0}
#screen{position:relative;width:${W}px;height:${H}px;overflow:hidden;
background:radial-gradient(ellipse at 30% 20%,#2c4a52 0%,#1a2b33 35%,#10161c 75%),#10161c}
#screen:before{content:"";position:absolute;inset:0;background:
radial-gradient(circle at 75% 80%,rgba(120,90,50,0.25),transparent 40%),
radial-gradient(circle at 10% 90%,rgba(40,80,50,0.35),transparent 45%)}
#note{position:absolute;right:${W - (clip.x + clip.width) + 10}px;top:${clip.y + clip.height - 20}px;z-index:99999;
font:500 10px Inter,sans-serif;color:rgba(255,255,255,0.45)}
</style></head><body><div id="screen">${parts.join('')}
<div id="note">Mock render from tools/wowmock (stand-in icons and font), not an in-game screenshot</div></div></body></html>`;

(async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 2 });
  const page = await context.newPage();
  await page.setContent(html);
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: output, clip });
  await browser.close();
  console.log(`wrote ${output} (${clip.width}x${clip.height})`);
})();
