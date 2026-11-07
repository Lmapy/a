// Render a wowmock scene (run.lua --dump) to a PNG with headless Chromium.
// This is an approximation for previews: game icons are replaced by emoji
// and the game font by Inter. Layout, sizes and colours come from the addon.
//
//   node tools/wowmock/preview.js scene.json out.png          cropped close-up
//   node tools/wowmock/preview.js scene.json out.png game     full 1080p screen with a
//                                                              stand-in game scene and HUD
const fs = require('fs');
const { chromium } = require('playwright');

const [, , input, output, mode] = process.argv;
const game = mode === 'game';
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
    if (op.gradient) {
      const [dir, low, high] = op.gradient;
      const toward = dir === 'VERTICAL' ? 'to top' : 'to right';
      return `<div style="position:absolute;z-index:${z};left:${left}px;top:${top}px;width:${width}px;height:${height}px;${radius}`
        + `background:linear-gradient(${toward},${css(low, op.alpha)},${css(high, op.alpha)})"></div>`;
    }
    if (op.color && op.path && /UI-StatusBar/.test(op.path)) {
      // Approximate the game's glossy status bar texture.
      const [r, g, b, a = 1] = op.color;
      const shade = (k) => css([r * k, g * k, b * k, a], op.alpha);
      return `<div style="position:absolute;z-index:${z};left:${left}px;top:${top}px;width:${width}px;height:${height}px;`
        + `background:linear-gradient(to bottom,${shade(1.25)} 0%,${shade(1.0)} 40%,${shade(0.72)} 100%)"></div>`;
    }
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
    // Strings the mock measured as several lines wrap like the client does.
    const lines = Math.round(height / (op.size * 1.2));
    if (op.fixedWidth && op.wrap && lines > 1) {
      return `<div style="${base}left:${left}px;top:${top}px;width:${width}px;text-align:${align};white-space:normal;line-height:${height / lines}px">${richText(op.text)}</div>`;
    }
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

// ---- Stand-in game scene (game mode only). Generic shapes, no game art. ----
const HUD_CSS = `
#ui{position:absolute;inset:0;z-index:2}
.scene{position:absolute;inset:0;z-index:0}
.hud{position:absolute;z-index:1;font-family:Inter,sans-serif;color:#e8e6e1;text-shadow:1px 1px 0 rgba(0,0,0,.9)}
.panel{background:rgba(8,10,14,.55);border:1px solid rgba(255,255,255,.08)}
.bar{height:10px;background:rgba(0,0,0,.6);border:1px solid rgba(0,0,0,.8)}
.bar>i{display:block;height:100%}
.slot{width:34px;height:34px;background:linear-gradient(160deg,#2a2f3a,#12151b);border:1px solid rgba(0,0,0,.9);
box-shadow:inset 0 0 0 1px rgba(255,255,255,.07);display:inline-block;margin-right:4px;vertical-align:top;position:relative}
.slot>b{position:absolute;inset:4px;border-radius:2px;opacity:.85}
.chat div{font-size:10.5px;line-height:15px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#note{right:auto!important;left:50%!important;top:6px!important;transform:translateX(-50%);white-space:nowrap}`;

function gameScene() {
  const trees = [];
  for (let i = 0; i < 26; i++) {
    const x = (i * 97 + (i % 3) * 23) % 1366;
    const y = 548 + ((i * 37) % 30);
    const h = 26 + ((i * 13) % 22);
    trees.push(`<polygon points="${x},${y - h} ${x - h * 0.32},${y} ${x + h * 0.32},${y}" fill="#0b1511" opacity="${0.75 + (i % 4) * 0.06}"/>`);
  }
  const slots = (n, tint) => Array.from({ length: n }, (_, i) =>
    `<span class="slot">${i % 3 === 1 ? '' : `<b style="background:linear-gradient(135deg,hsl(${(i * 47 + tint) % 360} 45% 45%),hsl(${(i * 47 + tint + 40) % 360} 40% 22%))"></b>`}</span>`).join('');
  return `
<svg class="scene" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
 <defs>
  <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
   <stop offset="0" stop-color="#0d1730"/><stop offset=".5" stop-color="#34385c"/>
   <stop offset=".68" stop-color="#9a5d5c"/><stop offset=".76" stop-color="#e09a6c"/></linearGradient>
  <linearGradient id="ground" x1="0" y1="0" x2="0" y2="1">
   <stop offset="0" stop-color="#1f3229"/><stop offset="1" stop-color="#0c1411"/></linearGradient>
  <radialGradient id="sun"><stop offset="0" stop-color="#ffe2b0" stop-opacity=".9"/><stop offset="1" stop-color="#ffb070" stop-opacity="0"/></radialGradient>
 </defs>
 <rect width="${W}" height="${H}" fill="url(#sky)"/>
 <circle cx="930" cy="505" r="120" fill="url(#sun)"/>
 <polygon points="0,500 120,430 260,470 400,410 560,468 700,420 860,470 1000,425 1150,465 1290,430 1366,452 1366,560 0,560" fill="#2a2f4c"/>
 <polygon points="0,528 180,478 330,515 520,470 690,520 880,482 1060,522 1230,486 1366,510 1366,580 0,580" fill="#1c2236"/>
 <path d="M0,560 C220,530 420,548 640,540 C880,532 1080,552 1366,536 L1366,768 L0,768 Z" fill="url(#ground)"/>
 <path d="M640,768 C660,690 700,620 760,566 L800,566 C760,620 740,690 760,768 Z" fill="#3a3a2c" opacity=".35"/>
 ${trees.join('')}
 <ellipse cx="683" cy="636" rx="26" ry="6" fill="#000" opacity=".45"/>
 <path d="M672,634 L676,600 C668,596 664,580 668,566 C672,556 680,552 683,552 C686,552 694,556 698,566 C702,580 698,596 690,600 L694,634 Z" fill="#0a0e0d"/>
 <circle cx="683" cy="540" r="10" fill="#0a0e0d"/>
</svg>
<div class="hud" style="left:18px;top:18px;width:236px;height:62px">
 <div style="position:absolute;left:0;top:0;width:58px;height:58px;border-radius:50%;background:radial-gradient(circle at 40% 35%,#6b5a4a,#1e1813);border:2px solid #3b3326"></div>
 <div style="position:absolute;left:66px;top:4px;font-size:12px;font-weight:600">Tester <span style="color:#bdb9ad;font-weight:400">18</span></div>
 <div class="bar" style="position:absolute;left:66px;top:22px;width:160px"><i style="width:82%;background:linear-gradient(#3fbf4f,#1e7a29)"></i></div>
 <div class="bar" style="position:absolute;left:66px;top:36px;width:160px;height:7px"><i style="width:64%;background:linear-gradient(#4a7bff,#2343a8)"></i></div>
</div>
<div class="hud" style="right:22px;top:16px;width:150px;text-align:center;font-size:11px">Westfall
 <div style="margin:6px auto 0;width:140px;height:140px;border-radius:50%;border:3px solid #2c2a26;
  background:radial-gradient(circle at 55% 45%,#5a6b3f 0,#3e5233 35%,#2a3a2a 70%,#1b241b 100%);box-shadow:0 0 0 1px #000"></div></div>
<div class="hud" style="right:24px;top:196px;width:210px;font-size:11px;line-height:16px">
 <div style="color:#ffd100;font-weight:600">The Copper Order</div><div style="color:#cfcbc0">&nbsp;- Smelt Copper Bars: 2/6</div>
 <div style="color:#ffd100;font-weight:600;margin-top:6px">Wolves at the Gate</div><div style="color:#cfcbc0">&nbsp;- Wolf Pelt: 4/8</div></div>
<div class="hud panel chat" style="left:18px;bottom:62px;width:410px;height:132px;padding:6px 8px;box-sizing:border-box">
 <div><span style="color:#ffc0c0">[2. Trade]</span> Someone: LF blacksmith, Copper Chain Boots</div>
 <div style="color:#c8f">[Guild] Bob: anyone need Copper Ore?</div>
 <div style="color:#59f">Your skill in Blacksmithing has increased to 45.</div>
 <div><span style="color:#66ccff">Profession Bars</span>: click a bar for that profession's plan. Hover the bars for options, /pb help for commands.</div>
 <div style="color:#3c3">You receive loot: [Copper Ore]x2.</div>
 <div style="color:#9a9">Welcome to the realm!</div></div>
<div class="hud" style="left:50%;bottom:50px;transform:translateX(-50%);white-space:nowrap">${slots(12, 200)}</div>
<div class="hud" style="left:50%;bottom:10px;transform:translateX(-50%);white-space:nowrap">${slots(12, 20)}</div>
<div class="hud" style="right:18px;bottom:10px;white-space:nowrap">${slots(5, 90)}</div>`;
}

const margin = 26;
const clip = game ? { x: 0, y: 0, width: W, height: H } : {
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
${game ? HUD_CSS : ''}
</style></head><body><div id="screen">${game ? gameScene() : ''}<div id="ui">${parts.join('')}</div>
<div id="note">${game ? 'Mock-up: stand-in game scene and HUD. Profession Bars is rendered from the addon code (stand-in icons and font).' : 'Mock render from tools/wowmock (stand-in icons and font), not an in-game screenshot'}</div></div></body></html>`;

(async () => {
  const browser = await chromium.launch();
  // Game mode renders a 1080p screen: WoW's UI is 768 units tall at any resolution.
  const context = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: game ? 1080 / H : 2 });
  const page = await context.newPage();
  await page.setContent(html);
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: output, clip });
  await browser.close();
  console.log(`wrote ${output} (${clip.width}x${clip.height})`);
})();
