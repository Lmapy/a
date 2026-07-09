# The Auction — Design System ("Terminal")

**Status:** v1.0 — packaged from the winning "terminal" direction.
**Source of truth:** `tokens.css` (this directory) + the final mockups in `./mockups/`
(`direction-terminal.html`, `screen-drill-loop.html`, `screen-dashboard.html`,
`screen-skill-tree.html`, `screen-boss-intro.html`).
**Product spec:** `../game-design-document.md` (GDD). Where this document and the GDD
disagree, the GDD wins on behavior; this document wins on pixels.

---

## 1. Art direction rationale

The Auction is a serious training tool wearing the clothes of a professional trading
terminal — not a game wearing the clothes of a market. Every direction decision follows
from three product facts:

1. **The chart is the content.** A drill is a stimulus (the profile), a question, and a
   judgment. So the chart is the only place color lives. UI chrome is a grayscale ramp
   of warm near-blacks and warm grays; the blue/orange/amber of the profile and the
   green/red of verdicts read as *events on top of silence*, exactly the way structure
   reads on a real DOM/profile terminal.

2. **The loop is sub-400ms.** At 8–15 reps/minute, any decoration becomes strobing.
   Elevation is lightness (no shadows), the primary CTA is warm off-white (`--cta`,
   never pure white — a white slab flashing every rep is fatigue), the verdict layer is
   the only animated layer, and the chart never moves when a verdict lands (GDD §2:
   juice on the verdict layer only).

3. **The numbers must be trusted.** Everything measured — prices, ratings, Brier,
   latencies, R-multiples — is set in JetBrains Mono with tabular numerals, at a fixed
   position, so a number changing is visible as *the number changing*, not the layout
   reflowing. Hierarchy is carried by the ink ramp (`hero → body → muted → ghost`), not
   by size inflation: an entire drill screen uses 14px base text with one 16px verdict
   line, and it reads instantly.

The references are Linear (surface discipline), monkeytype (one stimulus, one input,
nothing else), and lichess (dense data that respects the player). The anti-reference is
every broker app that confuses excitement with urgency.

---

## 2. Token reference

Canonical values live in `tokens.css`. These tables are the human-readable index.

### 2.1 Color — surfaces & ink

| Token | Value | Role | Contrast* |
|---|---|---|---|
| `--surface-0` | `#141413` | App background, every screen | — |
| `--surface-1` | `#1a1a19` | Panels: chart frame, cards, warm-up | — |
| `--surface-2` | `#222220` | Raised: chips, rating chip, NEXT | — |
| `--hairline` | `#2c2c2a` | All borders/dividers, ring tracks | — |
| `--gridline` | `#232322` | Chart-internal gridlines only | — |
| `--ink-hero` | `#ffffff` | Hero numbers, verdict word, active nav | 17.4:1 |
| `--ink-body` | `#c3c2b7` | Body, labels, explanations | 9.7:1 |
| `--ink-muted` | `#898781` | Meta, axis, captions, kickers | 4.85:1 |
| `--ink-ghost` | `#55534d` | Locked/unmeasured/watermark, done-ticks | 2.4:1 † |

\* vs `--surface-1`. † **Ghost is a non-text tier by design**: it appears only where
the *absence* of information is the information (locked node, unmeasured cell,
completed tick, chart reference line). It is never the sole carrier of meaning and
never used for content the player must read to act.

### 2.2 Color — chart & semantic

| Token | Value | Role | Contrast* |
|---|---|---|---|
| `--vol-blue` | `#3987e5` | In-VA bars, data lines, dots, sparklines | 4.8:1 |
| `--vol-blue-dim` | `#2a5a8f` | Outside-VA bars, loss bins | 2.45:1 † |
| `--poc-orange` | `#d95926` | POC row + axis price + `POC` text tag | 4.5:1 |
| `--highlight` | `#eda100` | Rusty/armed states, boss overline, avg markers, tells | 8.0:1 |
| `--va-band` | `rgba(57,135,229,.06)` | VA shading behind bars | — |
| `--va-edge` | `rgba(57,135,229,.30)` | VAH/VAL hairlines | — |
| `--buy-green` | `#1baf7a` | Correct / long — always with ✓ or ▲ | 6.2:1 |
| `--sell-red` | `#e66767` | Incorrect / short — always with ✗ or ▼ | 5.4:1 |
| `--sell-red-wash` | `rgba(230,103,103,.08)` | Picked-wrong chip fill | — |
| `--buy-green-wash` | `rgba(27,175,122,.08)` | Picked-right chip fill (derived) | — |
| `--cta` | `#e8e8e4` | Primary CTA fill; ink on it is `--surface-0` | 15.0:1 |

† Dim blue is graphical-only and always adjacent to full `--vol-blue` in the same
histogram; the datum is encoded by bar *length* (with a 6–7px minimum so single-tick
rows still read), never by the dim/bright distinction alone.

**CVD safety.** The green/red pair was chosen at distinct lightness (green L\* ≈ 62 vs
red L\* ≈ 62 hue-separated toward teal/salmon) and is *never load-bearing*: every
verdict pairs color with a glyph (§7). Blue/orange (bars/POC) is the classic
deuteranopia-safe axis; amber (`--highlight`) is reserved and always paired with a
symbol (`⟳`, `◉`, `●`) or text tag.

### 2.3 Typography

Two families, six faces, all local woff2 (no network):

```css
@font-face{font-family:'Inter';font-weight:400;src:url('fonts/inter-latin-400-normal.woff2') format('woff2')}
/* … Inter 500 / 600 / 700; JetBrains Mono 400 / 600 likewise … */
```

| Token | px | Face | Used for |
|---|---|---|---|
| `--text-2xs` | 10 | Mono 400/600, caps, tracked | Chart tags (POC/VAH/VAL), table headers |
| `--text-xs` | 11 | Mono 400–600, caps | Kickers, meta, axis prices, block labels, nav |
| `--text-sm` | 12 | Mono/Inter 400–600 | Latency line, captions, rating chip, replay |
| `--text-body-sm` | 13 | Inter 400/500 | Skill names, in-panel question |
| `--text-body` | 14 | Inter 400/500 | Base body, explanations, CTA labels |
| `--text-body-lg` | 15 | Inter 500 | Trap line, single emphasized sentences |
| `--text-verdict` | 16 | Inter 600 | Verdict line |
| `--text-chip` | 18 | Mono 600 | Answer-chip letters |
| `--text-title` | 20 | Inter 700 | Large verdict variant / section titles |
| `--text-display` | 24 | Inter 700, −1% tracked | Boss name |
| `--text-stat` | 28 | Mono 600 | Hero stats (Bot Points, Brier) |

Rules:
- **Every number the engine computed is mono + `font-variant-numeric: tabular-nums`.**
  No exceptions — prices, ratings, deltas, latencies, percentages, R-multiples.
- Anything ≤12px is uppercase, letter-spaced (see tracking ramp in `tokens.css`), and
  at `--ink-muted` or below — small text is always *chrome*, never *content*.
- Inter never renders a price. Mono never sets a sentence (single-letter shape terms
  like `P`/`b` inside explanations are the one sanctioned mono-inline use).

### 2.4 Spacing, radii, elevation, sizes

- **Spacing:** 4/8/12/16/24/32 (`--s1…--s6`). Mobile gutter and card padding `--s4`;
  desktop page padding `--s5`. No off-scale gaps.
- **Radii:** 8 default (panels, chips, buttons) · 6 small chips · 2 micro ticks ·
  1 profile bars · 50% dots/badges.
- **Elevation:** three surface steps + hairline borders. **No box-shadows anywhere.**
- **Controls:** 44px minimum hit target; 48 secondary; 52 CTA; 56 answer chips.

---

## 3. Component inventory (as observed in the mockups)

### 3.1 Chart frame
`--surface-1` panel, 1px `--hairline` border, `--radius`. Optional in-panel header:
question left (`13px` Inter 500 body ink) + provenance meta right (`11px` mono muted,
e.g. `RTH · 0.5 pt rows`). The chart itself:
- Horizontal volume bars, `--radius-bar`, min 6–7px length; in-VA `--vol-blue`,
  outside `--vol-blue-dim`, POC row `--poc-orange`.
- VA band `--va-band` behind bars, VAH/VAL hairlines `--va-edge` with `VAH`/`VAL`
  10px mono tags set clear of the bars.
- Price axis right-aligned, 11px mono `--ink-muted`, labels every ~4 rows with
  `--gridline` rules at labeled ticks; POC price + `POC` tag emphasized in orange with
  a dashed leader from bar to axis (never color-alone).
- In production this frame is the Canvas 2D chart layer; the SVG in mockups is
  geometry-identical.

### 3.2 Verdict banner
The moment of judgment, below the chart:
- **Line 1:** glyph + result — `✗ You said P — answer b`. Glyph and the player's pick
  in `--sell-red` (or ✓ + `--buy-green`); the true answer letter always in
  `--buy-green`; connective tissue muted. Right-aligned latency: `1.8s · par 3.0`
  (12px mono, actual value body-ink).
- **Line 2:** one-line auction-logic explanation, 14px Inter 400 body ink, max-width
  60ch, domain terms (`P`, `b`, `POC`) as 13px mono 600 hero-ink inline.
- **Line 3:** replay chip — 32px hairline-outline button, `⟲` + a *named* replay
  ("Replay the tail forming"), muted.

### 3.3 Answer chips
Equal-width row, 56px tall, `--surface-2` + hairline, mono 600 18px letters in muted
ink. States:
- **neutral** (pre-answer) · **dimmed** (post-verdict, opacity .45)
- **picked-wrong:** `--sell-red` border + `--sell-red-wash` fill + red letter +
  corner badge `✗` (16px circle, red fill, dark glyph, 2px surface ring) + `YOU` tag
- **truth:** `--buy-green` border + green letter + corner badge `✓`.
The correct answer is never painted red, even when missed.
A confidence ride-along row sits beneath: pre-answer three live 44px segments
(sure/lean/guess, default lean); post-verdict it collapses to a 30px echo with
`logged 75%` in mono.

### 3.4 Rating chip
Mono 600 12px on `--surface-2`, hairline border, `--radius-sm`, `2px 8px` padding:
`1512`, optional muted delta `↓18`. Dim variant (borderless, muted, `—`) for locked
nodes. Deltas are muted mono arrows — never green/red, never animated: Glicko movement
is information, not a reward.

### 3.5 Streak pill / warm-up card
Streak lives *only* in the Daily Warm-Up card header: `STREAK 12 · 2 freezes`, 11px
mono, count in body ink — a meta-metric, never celebration. The warm-up card is a
`--surface-1` panel: mono kicker title, one 13px description line (`3 min · 2 misses
due · …`), and the `--cta` START button (44px). In Rated drill chrome a zeroed streak
drops to muted/400 so it can never compete with the rating.

### 3.6 Navigation
- **Mobile bottom nav:** 56px, top hairline, full-bleed; 3 items (TREE / DRILL /
  STATS), 11px mono caps tracked `--track-nav`; active = hero ink + 32×2px body-ink
  tick overlapping the hairline. No icons.
- **Desktop top bar:** 56px, bottom hairline; wordmark (12px mono 600, tracked, with
  `▮` orange tick) + text nav (13px Inter, active hero/600); right side is a scope
  line, not controls (`last 90 days · all modes · 214 scored judgments`).

### 3.7 Table (expectancy ledger)
Fixed layout; mono 10px caps tracked headers under a hairline; 38px rows separated by
`--surface-2` rules (quieter than hairline). Setup column left-aligned Inter 500 13px;
data cells 12px mono right-aligned: hero-ink expectancy + muted `· win% · n` sub.
Unmeasured cells: centered 10px mono `--ink-ghost` "unmeasured = folklore" watermark.
Folklore column renders quoted strings in muted mono. Lede sentence above the table;
methodology caption below (12px muted).

### 3.8 Calibration chart
50–100% square reliability plot: `--gridline` grid, ghost diagonal labeled "perfect
calibration", blue connecting line (55% opacity) under CI whiskers (opacity ∝ 1/√n)
and blue dots area-sized by n with `--surface-1` rings. Axis titles 10px tracked
("you said %" / "it happened %"). The worst bin is annotated in plain text near its
dot; a dot-size key sits in the empty region. Always led by a sentence, never by the
chart: *"When you say 85%, it happens 70% — overconfident above 75."*

### 3.9 Supporting components
- **Block ticks:** 18×3px rounded ticks; upcoming `--surface-2`, done `--ink-ghost`,
  current `--ink-hero`; `REP 7/10` mono label right.
- **Par ring:** 16px SVG ring, hairline track, muted sweep; informational in Rated,
  a deadline only in Rush.
- **Stat row:** caps key (11px) → 28px mono hero value → muted mono delta, with a
  130×44 sparkline right (blue line, ghost target rule, end-dot + end-label).
- **Skill-tree node:** subway rail (1px hairline) + 19px state dot — mastered (body-ink
  fill, dark ✓), rusty (`--highlight` ring + ⟳), armed (`--highlight` ring + dot),
  in-progress (progress ring + core), locked (hairline ring) — plus name, optional
  status sub-line, rating chip. Active row gets a `--surface-1` card + chevron.
  Future tiers render as ghost silhouette boxes.
- **Boss framing:** amber mono overline (`◆ BOSS SESSION … GATES T2 → T3`), 24px
  Inter 700 name, honest expectation line (`Expect ~40% success` — amber mono number),
  "the trap" tell list (amber ● + mono stats), rules-of-engagement panel with
  hairline-ruled rows, best-attempt mono line, `--cta` enter button.
- **Buttons:** primary = `--cta` fill, dark ink, 44–52px, caps tracked 600; secondary =
  `--surface-2` + hairline (NEXT in-loop, so the loop's own CTA never flashes bright);
  tertiary = hairline outline, transparent (replay).

---

## 4. Layout grids

| Breakpoint | Grid | Chrome |
|---|---|---|
| **Mobile portrait ≤479px** (drills, tree, boss intro; design canvas 390×844) | Single column; page padding `--s4` (16px); vertical stack rhythm on `--s2/--s3/--s4` | Top bar 44–52px; bottom nav 56px full-bleed with top hairline |
| **Desktop ≥1024px** (stats, boss/sim; design canvas 1440×900) | 12-column CSS grid, `--s4` (16px) gap, page padding `--s5` (24px); observed spans: 6/3/3 top row, 6/3/(3 rolling) second row; fixed hero row (380px) + fluid rows | Top bar 56px with bottom hairline; no bottom nav |

**Drill screen vertical contract (mobile):** top chrome (52) → block ticks (~26) →
chart panel (`flex:1`, ≈70% viewport pre-answer, compressing when the verdict block
enters) → verdict block → answer chips (56) → confidence row → footer CTA (52). The
chart absorbs all flexible space; controls never move — the chip row is at the same
y-position on every item so the thumb never travels.

Tablet/intermediate widths center the mobile column (max-width ~480px) until the
desktop grid engages; there is no bespoke tablet layout in v1.

---

## 5. Motion spec — the sub-400ms verdict moment

Budgets from GDD §2; every deadline is a **latest-allowed**, not a target to fill.
All animation on the verdict layer only, compositor properties only
(`transform`/`opacity`); the chart layer never repaints, shakes, or emits anything.

| t | Event | Motion |
|---|---|---|
| **0ms** | `pointerdown` on chip/level (commit on pointerdown, `touch-action: manipulation`) | Chip pressed state: instant, no transition |
| **≤100ms** | **Verdict paint** (locally computed — no network) | Tapped level hit-flash (opacity pulse, `--dur-1`, `--ease-out`); verdict badge squash-stretches in (`scale .6→1`, `--dur-1`, `--ease-squash` — the system's only overshoot); chip borders/wash switch state; correct/incorrect earcon fires (Web Audio, pre-decoded) |
| **≤250ms** | **Explanation** | One-liner slides in under the chart (`translateY(6px)→0` + fade, `--dur-2`, `--ease-out`); non-picked chips dim to .45; latency line appears with no animation |
| **250–400ms** | **NEXT live** | Score/streak count-up (`--dur-3`, `--ease-linear`, tabular-nums so no reflow); NEXT enabled; replay chip fades in (`--dur-2`) |
| **~400ms** | Next item ready | Pre-generated + pre-rendered during the answer window; item swap is a `--dur-2` cross-fade of the chart layer, never a slide |

Constraints:
- `prefers-reduced-motion`: every entrance becomes an opacity fade at the same
  duration; squash-stretch and count-ups are removed (values appear settled).
- ≤3 luminance flashes/second even in Rush; the hit-flash never repeats.
- Nothing animates on an interval; nothing loops; nothing moves while the player is
  reading a profile. Checkpoint mode disables the verdict layer entirely (juice-free
  by spec).
- `--dur-4` (320ms) exists for screen-to-screen transitions only and is banned inside
  the rep loop.

---

## 6. Accessibility rules

1. **CVD redundancy (hard rule):** color never carries meaning alone. Every verdict
   pairs color with ✓/✗; direction with ▲/▼; POC with the `POC` text tag + axis price;
   rusty/armed with ⟳/◉ glyphs; Rush strikes are filled/hollow dots *plus* a count.
   The colorblind pass in the GDD test plan (§10-D) is the acceptance test: every
   verdict readable by icon alone.
2. **Contrast:** all content text ≥4.5:1 (`--ink-muted` at 4.85:1 is the floor);
   verdict/semantic colors ≥3:1 on any surface they appear on (measured: 4.5–8:1);
   `--ink-ghost` (2.4:1) and `--vol-blue-dim` (2.45:1) are restricted to the
   non-information roles defined in §2 and never gate a decision.
3. **Touch targets:** ≥44px everywhere; answer chips 56px; snap-to-row on chart taps
   (nearest row center within 22px) so pixel precision is never required.
4. **Motion:** `prefers-reduced-motion` honored (§5); ≤3 flashes/sec; replays pausable
   with step controls.
5. **Sound:** earcons are redundant to visuals, never sole feedback; one-tap mute.
6. **Text:** explanations ≤160 chars to first period, task-referenced only; base 14px,
   sub-12px reserved for chrome; `lang`/semantic markup on tables and nav; hero
   numbers exposed to AT with their caps keys as labels.

---

## 7. Restraint charter — what this UI will never do

1. **No shadows, no gradients, no glows, no blur.** Elevation is surface lightness +
   hairline. If a panel needs to feel higher, it gets one step lighter, not softer.
2. **No color in chrome.** Grayscale UI forever; color belongs to the chart (blue/
   orange/amber) and to verdicts (green/red, glyph-paired, at the moment of judgment
   only). No colored buttons, no brand-tinted surfaces.
3. **No naked verdicts, no naked charts.** Every ✓/✗ carries its one-line auction
   logic; every chart is led by a sentence.
4. **No celebration.** No confetti, particles, badges, XP, levels, coins, chests,
   fire emoji, or streak dramatization. A milestone is a sentence and a number. The
   chart never shakes. Rating deltas are muted arrows in mono.
5. **No urgency theater.** No pulsing CTAs, no countdown reds (the par ring is
   hairline-gray even in Rush), no "3 others are drilling now", no notification dots.
6. **No dark patterns.** No appointment mechanics, no decaying rewards, no near-miss
   dramatization, no guilt copy on a broken streak (it just goes muted), no
   monetization surface anywhere in the loop.
7. **No decoration on data.** Nothing is drawn on a chart that the engine didn't
   compute (`@auction/core` single-library rule); no smoothing, no fake ticks, no
   illustrative charts. P&L is shown always, scored never — and styled accordingly.
8. **No layout motion.** Numbers change in place (tabular-nums); controls never
   relocate mid-loop; nothing auto-scrolls, auto-plays, or animates unprompted.
9. **No new values.** No hex outside `tokens.css` (linted); no off-scale spacing,
   sizes, or durations. A new token requires removing or justifying an old one.
10. **No light mode in v1, no theming.** One dark surface, tuned once, everywhere —
    traders live here; the trainer should feel like where they already work.
