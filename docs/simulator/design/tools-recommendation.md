# The Auction — Frontend Tooling Recommendation (v1)

**Status:** Decision doc, v1.0.
**Scope:** rendering, framework, animation, audio, persistence, rating math, build/test,
PWA, and design-token plumbing for the v1 web client described in the GDD
(`../game-design-document.md`) and the terminal design system (`design-system.md`,
`tokens.css`).
**Decision drivers, in order:** (1) the sub-400ms tap→verdict→explanation loop with
every latency-critical operation local and imperative; (2) pixel-accurate,
`@auction/core`-derived charts with crisp monospace price text; (3) zero heavyweight
dependencies in the hot path; (4) offline-first PWA with no backend.

---

## Recommended stack

| Area | Choice | Runner-up | Why (one line) |
|---|---|---|---|
| Chart rendering | **Hand-rolled Canvas 2D** (~1–2k LoC) | PixiJS/WebGL | Workload is 100× below Canvas limits; WebGL costs crisp mono text we can't lose |
| Framework | **Svelte 5 (runes) + SvelteKit static adapter** | React 19 + Compiler | Hot path bypasses any framework; Svelte wins the margins (~2–5kB runtime, top benchmark scores) |
| Animation | **CSS transitions/keyframes + WAAPI** | GSAP (if debrief scrubber outgrows WAAPI) | Verdict juice is compositor-thread micro-interaction; a library adds bytes, not capability |
| Audio | **Raw Web Audio API** | Howler | Three earcons need one AudioContext and pre-decoded buffers, ~10–30ms start latency |
| Persistence | **Dexie 4 on IndexedDB** (+ `navigator.storage.persist()`) | raw IDB | Indexed ledger queries, async off the main thread; localStorage for trivial prefs only |
| Rating math | **Vendored Glicko-2 (~200 lines)** in `@auction/schedule` | `glicko2.ts` npm dep | Golden-tested against the npm package; consistent with zero-dep pure-TS core culture |
| Build/test | **Vite + Vitest 4 + Playwright** | — | Node-mode golden/property tests; Browser Mode + Playwright screenshots make pixels a tested invariant |
| PWA | **vite-plugin-pwa** (generateSW, autoUpdate) | hand-rolled SW | Zero runtime network requests → full offline is nearly free; precache <~1MB |
| Tokens | **Vanilla CSS custom properties** (`tokens.css`) | Tailwind v4 `@theme` | Canvas reads the same variables via `getComputedStyle`; Stylelint bans hex elsewhere |

---

## Rationale by area

### Rendering: hand-rolled Canvas 2D

The workload — ~200 bars plus a histogram per item — is two orders of magnitude below
where Canvas 2D breaks (tens of thousands of elements at 60fps). WebGL/PixiJS buys
headroom we don't need and costs the thing we can't lose: crisp monospace price text.
TradingView lightweight-charts is time-axis-bound; volume profiles (price-axis
histograms) exist only as plugin workarounds — and every displayed quantity must come
from `@auction/core` anyway, so a charting library's math is dead weight at best and a
divergence risk at worst.

Own the ~1–2k-line renderer:

- **DPR-scaled backing store** (`ctx.scale(devicePixelRatio)`) for crisp text and
  true 1px hairlines.
- **Two stacked canvases:** the chart layer repaints only on item change; the
  verdict/juice layer never touches chart pixels — satisfying GDD §9 ("juice never
  repaints data") literally, in the compositor.
- **Item generation in a Web Worker**, with optional OffscreenCanvas pre-render of the
  next item during the answer window, so the ~400ms item swap never stutters.
- **Commit answers on `pointerdown`** with `touch-action: manipulation` — worth
  50–100ms of the 400ms budget on mobile versus waiting for click.

### Framework: Svelte 5 (runes) + SvelteKit static adapter

Honest caveat first: framework choice barely matters here, because the verdict hot
path is imperative — `pointerdown` → grade against local labels → paint canvas → CSS
transition — and bypasses any framework. Svelte 5 wins on the margins that remain:

- ~2–5kB runtime vs React's ~45kB, which is real on a PWA cold start on mid-tier
  phones;
- top-tier js-framework-benchmark scores for the tickers, film strip, and stats views
  that *do* go through the framework;
- runes work in plain `.svelte.ts` files, so the drill state machine
  (`ITEM → ARMED → ANSWERED → VERDICT → NEXT`) and the scheduler stay
  framework-agnostic TypeScript beside the pure-TS core packages.

React 19 + Compiler is an acceptable substitute if the team is React-native; SolidJS
is the marginally-faster, smaller-ecosystem runner-up.

### Animation: CSS transitions/keyframes + WAAPI, no library

Verdict squash-stretch, explanation slide-in, and count-ups are compositor-thread
micro-interactions covered by the motion tokens in `tokens.css` (`--dur-1…4`,
`--ease-squash` etc.); `prefers-reduced-motion` is one media query. GSAP (now fully
free) is approved *only* if the boss debrief timeline scrubber outgrows WAAPI —
nothing in the rep loop ever justifies it.

### Audio: raw Web Audio API

One `AudioContext({ latencyHint: 'interactive' })` unlocked on first gesture; all
three earcons pre-decoded to `AudioBuffer`s at boot; a fresh `AudioBufferSourceNode`
per play. That yields ~10–30ms start latency — comfortably inside the 100ms verdict
budget. Howler is unnecessary for four sounds.

### Persistence: Dexie 4 on IndexedDB

The decision ledger `(decisionId, nodeId, seed, paramsVersion, answer, latency,
verdict, brier)` and rep history need indexed queries by node/date/schedule, hundreds
of MB of headroom, and async access with no main-thread jank — that's IndexedDB, and
Dexie 4 is the thinnest ergonomic layer over it. Add `navigator.storage.persist()` so
the browser can't evict a player's history. localStorage only for trivial prefs
(mute, reduced-motion override). **Never await a write in the verdict path** — queue
and flush on idle.

### Glicko-2: vendor ~200 lines

Vendor the implementation into `@auction/schedule`, golden-tested against the
`glicko2.ts` npm package, consistent with the zero-dependency pure-TS culture of
`core`/`gen`/`schedule`. Rating math is product logic, not a dependency.

### Build/test: Vite + Vitest 4 + Playwright

- **Node-mode Vitest** for golden-file parity with the Python reference
  (`/scripts/figures/`), property tests, and ensemble invariants (GDD §10-A).
- **Vitest 4 Browser Mode** (now stable) with `toMatchScreenshot` for pixel-level
  component regression of the profile renderer.
- **Playwright `toHaveScreenshot`** for screen-level baselines — captured inside the
  Playwright Docker image so CI and dev machines agree on font rendering, with
  `animations: 'disabled'`, dynamic regions masked, baselines in git.
- A **throttled-CPU Playwright perf test** asserts the 100/250/400ms budgets (GDD
  §10-C) on every build.

### PWA: vite-plugin-pwa (generateSW, autoUpdate)

Precache the shell + woff2 fonts + earcons via `globPatterns`. With zero runtime
network requests, full offline is nearly free and the precache stays under ~1MB.

### Tokens: vanilla CSS custom properties

One `tokens.css` (already authored: seed palette, spacing scale, Inter/JetBrains Mono
stacks, motion). The canvas renderer reads the same variables via `getComputedStyle`
at init, so chart and DOM can never diverge; Stylelint bans raw hex outside the token
file. Tailwind v4's `@theme` is the acceptable alternative if the team wants utility
classes; style-dictionary is deferred until a second platform exists.

---

## What we are explicitly not using (and why)

| Rejected | Reason |
|---|---|
| PixiJS / WebGL / three.js | Headroom we don't need; blurry or atlas-managed text; shader pipeline to maintain |
| TradingView lightweight-charts / ECharts / d3-render | Time-axis assumptions, plugin-hack volume profiles, math we must not trust anyway |
| React (default choice) | Acceptable substitute, but 45kB runtime buys nothing the hot path uses |
| GSAP / Motion One / Lottie | Micro-interactions don't need a timeline engine (GSAP conditionally approved for the debrief scrubber only) |
| Howler / Tone.js | Four sounds |
| Redux / Zustand / XState | The drill state machine is ~5 states of plain TS; runes cover reactive glue |
| localStorage as primary store | Sync, string-only, ~5MB, main-thread |
| glicko2 npm at runtime | Vendored + golden-tested instead; used only as test oracle |
| style-dictionary / Tailwind | Deferred; one platform, one token file |
| Any backend/CDN/analytics SDK | v1 is local-first, offline, zero external requests by design |

## Risks & mitigations

- **Hand-rolled renderer correctness** → mitigated by the single-library rule (all
  geometry from `@auction/core`), Browser-Mode screenshot tests, and golden parity
  with the Python reference.
- **Svelte team familiarity** → runes-in-`.ts` keeps 80% of the logic
  framework-agnostic; React 19 substitution is pre-approved with no architecture
  change.
- **Screenshot-test flake** → Dockerized Playwright image, disabled animations,
  masked dynamic regions.
- **IndexedDB eviction** → `navigator.storage.persist()` + exportable ledger (GDD §9
  analytics are local and exportable).

**Bottom line:** zero heavyweight dependencies in the hot path, every
latency-critical operation local and imperative, and pixel accuracy a tested
invariant rather than a hope.
