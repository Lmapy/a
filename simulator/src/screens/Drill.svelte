<script lang="ts">
  /* Drill screen — the hero loop (GDD §2/§8 screen 2), pixel-matched to
     design/mockups/screen-drill-loop.html. Layout contract (design system
     §4): top chrome → block ticks → chart panel (flex:1) → verdict → answer
     chips → confidence → footer CTA. The chart is a Canvas 2D layer; the
     verdict layer is DOM — juice never repaints data (GDD §2).

     Dev/e2e query params (after the hash route, e.g.
     #/drill?drill=shape-alphabet&seed=42&state=verdict&pick=P):
       drill = one of the five wired drill ids   (default shape-alphabet)
       seed  = deterministic block seed; implies noqueue (reproducible)
       mode  = 'checkpoint' → the delayed gate exam (GDD §3 mode 6):
               juice-free (no per-item verdict, auto-advance, distinct
               frame), 40/60 interleave with mastered siblings, full
               debrief + pass/fail at the end. With `seed` the block is
               deterministic; sibling mix comes from the real ledger.
       state = 'verdict' → auto-commits an answer at a fixed 1.8s latency
       pick  = the auto answer (chip label, or a row index for snap drills);
               default = first wrong chip / truth+3 rows
       conf  = sure|lean|guess for the auto answer (default lean)          */
  import { onMount } from 'svelte';
  import { DrillBlock } from '../lib/drills/block.svelte';
  import {
    DRILL_TITLE,
    LOOP_DRILLS,
    barClock,
    renderFlags,
    type LoopDrillId,
  } from '../lib/drills/items';
  import {
    readChartTheme,
    renderProfile,
    renderOpeningLine,
    profileLayout,
    rowAtY,
    type ChartTheme,
    type RowMarker,
  } from '../lib/render/profileCanvas';
  import { devHooks } from '../lib/render/devhooks';
  import { buildProfile, rowToPrice } from '../lib/core/profile';
  import { player } from '../lib/stores/player.svelte';
  import { INITIAL_RD } from '../lib/schedule/glicko';
  import { confidenceProb } from '../lib/schedule/persist';
  import type { Confidence } from '../lib/types';
  import type { RepRecord } from '../lib/drills/summary';

  /* ---------------- params + block lifecycle ---------------- */

  function queryParams(): URLSearchParams {
    const h = window.location.hash;
    const q = h.indexOf('?');
    return new URLSearchParams(q >= 0 ? h.slice(q + 1) : '');
  }

  let block = $state<DrillBlock | null>(null);
  let startRating = $state(1500);
  let startRd = $state(350);
  let paramSig = '';

  function isLoopDrill(x: string | null): x is LoopDrillId {
    return x !== null && (LOOP_DRILLS as string[]).includes(x);
  }

  async function startBlock(): Promise<void> {
    const p = queryParams();
    const drillId: LoopDrillId = isLoopDrill(p.get('drill')) ? (p.get('drill') as LoopDrillId) : 'shape-alphabet';
    const seed = p.get('seed') ?? undefined;
    const b = new DrillBlock({
      drillId,
      seed,
      noQueue: seed !== undefined,
      checkpoint: p.get('mode') === 'checkpoint',
    });
    filmIdx = null;
    thumbs = [];
    block = b;
    startRating = Math.round(player.rating(drillId).rating);
    startRd = Math.round(player.rating(drillId).rd);
    await b.init();
    if (p.get('state') === 'verdict') autoAnswer(b, p);
  }

  /** Deterministic auto-answer for screenshots/e2e (?state=verdict). */
  function autoAnswer(b: DrillBlock, p: URLSearchParams): void {
    const li = b.current;
    if (!li || b.phase !== 'armed') return;
    const conf = p.get('conf');
    if (conf === 'sure' || conf === 'lean' || conf === 'guess') b.confidence = conf;
    const pickParam = p.get('pick');
    if (li.item.choices.length > 0) {
      const pick =
        pickParam && li.item.choices.includes(pickParam)
          ? pickParam
          : (li.item.choices.find((c) => c !== li.item.groundTruth) ?? li.item.choices[0]);
      commit(pick, 1800);
    } else {
      const truth = li.item.groundTruth as number;
      const nRows = li.item.stimulus.profile.rows.length;
      const parsed = pickParam !== null ? Number(pickParam) : Number.NaN;
      const row = Number.isFinite(parsed) ? parsed : Math.min(nRows - 1, truth + 3);
      commit(row, 1800);
    }
  }

  function paramsSignature(): string {
    const p = queryParams();
    return `${p.get('drill')}|${p.get('seed')}|${p.get('state')}|${p.get('pick')}|${p.get('mode')}`;
  }

  onMount(() => {
    paramSig = paramsSignature();
    void startBlock();
    const onHash = (): void => {
      const sig = paramsSignature();
      if (sig !== paramSig) {
        paramSig = sig;
        void startBlock();
      }
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  });

  /* ---------------- chart layer ---------------- */

  let bodyEl = $state<HTMLDivElement | null>(null);
  let canvasEl = $state<HTMLCanvasElement | null>(null);
  let theme: ChartTheme | null = null;
  let lastRenderMs = $state(0);

  function markers(): RowMarker[] {
    const b = block;
    // checkpoint = juice-free: truth markers only in the debrief film strip
    if (!b?.current || b.phase !== 'verdict' || b.isCheckpoint) return [];
    if (b.current.item.drillId !== 'poc-va-snap') return [];
    const truth = b.current.item.groundTruth as number;
    const pick = typeof b.lastAnswer?.choice === 'number' ? b.lastAnswer.choice : truth;
    const target = b.current.item.question === 'Tap the POC' ? 'POC' : b.current.item.question.slice(-3);
    const out: RowMarker[] = [];
    if (pick === truth) {
      out.push({ row: truth, kind: 'you-right', label: `YOU · ${target}` });
    } else {
      out.push({ row: pick, kind: b.verdict?.correct ? 'you-right' : 'you-wrong', label: 'YOU' });
      out.push({ row: truth, kind: 'truth', label: target });
    }
    return out;
  }

  function draw(barsUpTo?: number): void {
    const b = block;
    if (!b?.current || !canvasEl || !bodyEl) return;
    if (!theme) theme = readChartTheme(bodyEl);
    const w = bodyEl.clientWidth;
    const h = bodyEl.clientHeight;
    if (w < 10 || h < 10) return;
    const item = b.current.item;
    let ms: number;
    if (item.drillId === 'open-type-ladder') {
      const bars = barsUpTo ? item.stimulus.bars.slice(0, barsUpTo) : item.stimulus.bars;
      ms = renderOpeningLine(canvasEl, bars, w, h, { theme, openPrice: item.stimulus.bars[0].o });
    } else {
      const profile = barsUpTo
        ? buildProfile(item.stimulus.bars.slice(0, barsUpTo), item.stimulus.profile.rowStep)
        : item.stimulus.profile;
      ms = renderProfile(canvasEl, profile, w, h, {
        theme,
        ...renderFlags(item),
        highlightRow: barsUpTo ? null : (item.stimulus.highlightRow ?? null),
        markers: barsUpTo ? [] : markers(),
      });
    }
    lastRenderMs = ms;
    devHooks().lastRenderMs = ms;
    if (import.meta.env.DEV) console.debug(`[auction] profile redraw ${ms.toFixed(2)}ms`);
  }

  // redraw on item / phase changes
  $effect(() => {
    void block?.current;
    void block?.phase;
    void canvasEl;
    if (!replaying) draw();
  });

  // redraw on panel resize
  $effect(() => {
    if (!bodyEl) return;
    const ro = new ResizeObserver(() => {
      if (!replaying) draw();
    });
    ro.observe(bodyEl);
    return () => ro.disconnect();
  });

  /* ---------------- answering ---------------- */

  function commit(choice: string | number, latencyOverride?: number): void {
    const b = block;
    if (!b || b.phase !== 'armed') return;
    const t0 = performance.now();
    const v = b.answer(choice, latencyOverride);
    if (v) {
      // freeze the par ring at the answer's latency (informational, GDD §2)
      if (b.current && b.lastAnswer) {
        parProgress = Math.min(1, b.lastAnswer.latencyMs / b.current.item.parMs);
      }
      requestAnimationFrame(() => {
        devHooks().verdictPaintMs = performance.now() - t0;
      });
      // checkpoint mode is juice-free (GDD §3 mode 6): no per-item verdict —
      // hold a short "recorded" beat, then serve the next item automatically
      if (b.isCheckpoint) {
        setTimeout(() => {
          if (b === block && b.phase === 'verdict') b.advanceRep();
        }, 350);
      }
    }
  }

  function onCanvasDown(ev: PointerEvent): void {
    const b = block;
    if (!b?.current || b.phase !== 'armed' || b.current.item.drillId !== 'poc-va-snap' || !canvasEl) return;
    const rect = canvasEl.getBoundingClientRect();
    const L = profileLayout(b.current.item.stimulus.profile.rows.length, rect.width, rect.height);
    const row = rowAtY(L, ev.clientY - rect.top);
    if (row !== null) commit(row);
  }

  /* ---------------- par ring ---------------- */

  let parProgress = $state(0);
  const reducedMotion =
    typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  $effect(() => {
    const b = block;
    if (!b || b.phase !== 'armed' || reducedMotion) {
      // keep the ring frozen through the verdict; reset otherwise
      if (!b || (b.phase !== 'verdict' && b.phase !== 'armed')) parProgress = 0;
      return;
    }
    parProgress = 0;
    let raf = 0;
    const tick = (): void => {
      if (b.phase !== 'armed' || !b.current) return;
      parProgress = Math.min(1, (performance.now() - b.armedAt) / b.current.item.parMs);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  });

  function arcPath(p: number): string {
    const t = Math.max(0.02, Math.min(0.998, p));
    const a = -Math.PI / 2 + t * 2 * Math.PI;
    const x = 8 + 6.5 * Math.cos(a);
    const y = 8 + 6.5 * Math.sin(a);
    return `M 8 1.5 A 6.5 6.5 0 ${t > 0.5 ? 1 : 0} 1 ${x.toFixed(2)} ${y.toFixed(2)}`;
  }

  /* ---------------- micro-replay (player-invoked, GDD §2) ---------------- */

  let replaying = $state(false);
  let replayRaf = 0;

  function replayLabel(id: LoopDrillId): string {
    switch (id) {
      case 'excess-or-poor':
        return 'Replay the extreme forming';
      case 'open-type-ladder':
        return 'Replay the open';
      case 'regime-gate':
        return 'Replay the tape';
      default:
        return 'Replay the profile forming';
    }
  }

  function toggleReplay(): void {
    const b = block;
    if (!b?.current) return;
    if (replaying) {
      cancelAnimationFrame(replayRaf);
      replaying = false;
      draw();
      return;
    }
    replaying = true;
    const total = b.current.item.stimulus.bars.length;
    const from = Math.max(10, Math.floor(total * 0.25));
    const durMs = 1400;
    const t0 = performance.now();
    const step = (): void => {
      const f = Math.min(1, (performance.now() - t0) / durMs);
      draw(Math.max(2, Math.floor(from + (total - from) * f)));
      if (f < 1 && replaying) {
        replayRaf = requestAnimationFrame(step);
      } else {
        replaying = false;
        draw();
      }
    };
    replayRaf = requestAnimationFrame(step);
  }

  /* ------------- film strip (GDD §2 Permanence / §8 screen 3) ------------- */

  let thumbs = $state<(HTMLCanvasElement | null)[]>([]);
  let filmIdx = $state<number | null>(null);
  let filmCanvas = $state<HTMLCanvasElement | null>(null);

  /** Paint one thumbnail: the rep's served profile rows, engine-computed. */
  function paintThumb(cv: HTMLCanvasElement, rep: RepRecord): void {
    const w = cv.clientWidth;
    const h = cv.clientHeight;
    if (w < 2 || h < 2) return;
    if (!theme) theme = readChartTheme(cv);
    const dpr = typeof devicePixelRatio === 'number' ? devicePixelRatio : 1;
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
    const g = cv.getContext('2d');
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const rows = rep.li.item.stimulus.profile.rows;
    const max = Math.max(...rows, 1e-9);
    const rh = h / rows.length;
    g.fillStyle = rep.verdict.correct ? theme.volBlue : theme.volBlueDim;
    for (let r = 0; r < rows.length; r++) {
      const bw = (rows[r] / max) * (w - 2);
      if (bw <= 0) continue;
      // row 0 = lowest price = bottom of the thumb
      g.fillRect(1, h - (r + 1) * rh, bw, Math.max(0.75, rh - 0.5));
    }
  }

  $effect(() => {
    const b = block;
    if (!b || b.phase !== 'summary') return;
    void thumbs.length;
    for (let i = 0; i < b.reps.length; i++) {
      const cv = thumbs[i];
      if (cv) paintThumb(cv, b.reps[i]);
    }
  });

  /** Verdict markers for a replayed snap rep (same rules as the live layer). */
  function filmMarkers(rep: RepRecord): RowMarker[] {
    if (rep.li.item.drillId !== 'poc-va-snap') return [];
    const truth = rep.li.item.groundTruth as number;
    const pick = typeof rep.answer.choice === 'number' ? rep.answer.choice : truth;
    const target = rep.li.item.question === 'Tap the POC' ? 'POC' : rep.li.item.question.slice(-3);
    if (pick === truth) return [{ row: truth, kind: 'you-right', label: `YOU · ${target}` }];
    return [
      { row: pick, kind: rep.verdict.correct ? 'you-right' : 'you-wrong', label: 'YOU' },
      { row: truth, kind: 'truth', label: target },
    ];
  }

  /** One frame of a film replay — the same engine renderers as the live chart. */
  function drawFilmFrame(rep: RepRecord, barsUpTo?: number): void {
    const cv = filmCanvas;
    const host = cv?.parentElement;
    if (!cv || !host) return;
    if (!theme) theme = readChartTheme(host);
    const w = host.clientWidth;
    const h = host.clientHeight;
    if (w < 10 || h < 10) return;
    const item = rep.li.item;
    if (item.drillId === 'open-type-ladder') {
      const bars = barsUpTo ? item.stimulus.bars.slice(0, barsUpTo) : item.stimulus.bars;
      renderOpeningLine(cv, bars, w, h, { theme, openPrice: item.stimulus.bars[0].o });
    } else {
      const profile = barsUpTo
        ? buildProfile(item.stimulus.bars.slice(0, barsUpTo), item.stimulus.profile.rowStep)
        : item.stimulus.profile;
      renderProfile(cv, profile, w, h, {
        theme,
        ...renderFlags(item),
        highlightRow: barsUpTo ? null : (item.stimulus.highlightRow ?? null),
        markers: barsUpTo ? [] : filmMarkers(rep),
      });
    }
  }

  // replay the selected thumbnail: developing profile rebuilt per frame
  $effect(() => {
    const i = filmIdx;
    const cv = filmCanvas;
    const b = block;
    if (i === null || !cv || !b || b.phase !== 'summary') return;
    const rep = b.reps[i];
    if (!rep) return;
    if (reducedMotion) {
      drawFilmFrame(rep);
      return;
    }
    const total = rep.li.item.stimulus.bars.length;
    const from = Math.max(10, Math.floor(total * 0.25));
    const durMs = 1400;
    const t0 = performance.now();
    let raf = 0;
    const step = (): void => {
      const f = Math.min(1, (performance.now() - t0) / durMs);
      drawFilmFrame(rep, f < 1 ? Math.max(2, Math.floor(from + (total - from) * f)) : undefined);
      if (f < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  });

  function tapThumb(i: number): void {
    filmIdx = filmIdx === i ? null : i;
  }

  /* ---------------- derived display ---------------- */

  const CONF: Confidence[] = ['sure', 'lean', 'guess'];

  const item = $derived(block?.current?.item ?? null);
  const isSnap = $derived(item?.drillId === 'poc-va-snap');
  const isCp = $derived(block?.isCheckpoint ?? false);
  const rating = $derived(block ? Math.round(player.rating(block.drillId).rating) : 1500);
  /* Provisional-RD framing: while rating deviation is still high, a single
     miss legitimately swings the display rating hard (Glicko-2). Keyed off
     the BLOCK-START deviation (a fresh node's INITIAL_RD collapses within a
     rep or two) so the whole first block — and long-idle nodes whose RD has
     decayed back up — carry the framing, and a new player's early swings
     don't read as punishment. */
  const ratingRd = $derived(block ? Math.round(player.rating(block.drillId).rd) : INITIAL_RD);
  const provisional = $derived(startRd >= 0.85 * INITIAL_RD);

  function metaLine(): string {
    const b = block;
    if (!b?.current) return '';
    const it = b.current.item;
    if (it.drillId === 'open-type-ladder') {
      return `first 90 min · open ${it.stimulus.bars[0].o.toFixed(2)}`;
    }
    if (it.drillId === 'regime-gate') {
      const loc =
        it.labels.openLocation === 'in-value'
          ? 'in value'
          : it.labels.openLocation === 'out-of-range'
            ? 'out of range'
            : 'out of value';
      return `thru ${barClock(it.stimulus.revealBar)} · open ${loc}`;
    }
    return `RTH · ${it.stimulus.profile.rowStep} pt rows`;
  }

  function pickText(): string {
    const b = block;
    if (!b?.lastAnswer || !b.current) return '';
    if (typeof b.lastAnswer.choice === 'number') {
      return rowToPrice(b.current.item.stimulus.profile, b.lastAnswer.choice).toFixed(2);
    }
    return String(b.lastAnswer.choice);
  }

  function truthText(): string {
    const b = block;
    if (!b?.current) return '';
    const gt = b.current.item.groundTruth;
    if (typeof gt === 'number') return rowToPrice(b.current.item.stimulus.profile, gt).toFixed(2);
    return String(gt);
  }

  function chipState(c: string): 'picked' | 'truth' | 'dimmed' | 'neutral' | 'committed' {
    const b = block;
    if (!b || b.phase !== 'verdict') return 'neutral';
    const picked = String(b.lastAnswer?.choice) === c;
    // checkpoint: register the tap without revealing right/wrong or the truth
    if (b.isCheckpoint) return picked ? 'committed' : 'dimmed';
    const truth = String(b.current?.item.groundTruth) === c;
    if (picked) return 'picked';
    if (truth) return 'truth';
    return 'dimmed';
  }

  function letterClass(c: string): string {
    if (c.length <= 1) return 'letter';
    if (c.length <= 9) return 'letter small';
    return 'letter tiny';
  }

  const fmtS = (ms: number): string => (ms / 1000).toFixed(1);

  /* Explanation presentation: the verdict line already carries the glyph, so
     the leading ✗/✓ is stripped for display (the canonical template string is
     preserved verbatim in Verdict.explanation for the ledger/audit); domain
     terms render inline in mono per design-system §3.2 line 2. */
  interface ExplainSeg {
    t: string;
    term: boolean;
  }
  const TERM_RE = /(TREND|nPOC|POC|VAH|VAL|HVN|LVN|IB|P|b|D|B)/g;
  function explainSegs(explanation: string): ExplainSeg[] {
    const text = explanation.replace(/^[✗✓]+ ?/u, '');
    const isBoundary = (c: string | undefined): boolean => c === undefined || !/[A-Za-z0-9-]/.test(c);
    const out: ExplainSeg[] = [];
    let last = 0;
    for (const m of text.matchAll(TERM_RE)) {
      const i = m.index ?? 0;
      if (!isBoundary(text[i - 1]) || !isBoundary(text[i + m[0].length])) continue;
      if (i > last) out.push({ t: text.slice(last, i), term: false });
      out.push({ t: m[0], term: true });
      last = i + m[0].length;
    }
    if (last < text.length) out.push({ t: text.slice(last), term: false });
    return out;
  }
</script>

<section class="page">
  {#if block && block.phase === 'summary' && block.summary}
    {@const s = block.summary}
    {@const cp = block.checkpointResult}
    <!-- result screen (GDD §8 screen 3): sentence first, numbers second -->
    <div class="topbar">
      <div class="mode num"><b>{cp ? 'Checkpoint' : 'Rated'}</b> · {DRILL_TITLE[block.drillId]}</div>
      <div class="top-stats">
        <div class="stat">
          <span class="k">Rating</span>
          <span class="v hero num">{rating.toLocaleString('en-US')}</span>
          <span class="delta num">{rating >= startRating ? `↑${rating - startRating}` : `↓${startRating - rating}`}</span>
        </div>
      </div>
    </div>
    <div class="blockrow">
      <div class="ticks">
        {#each Array(block.size) as _, i (i)}
          <div class="tick done"></div>
        {/each}
      </div>
      <div class="blocklabel num">{cp ? 'CHECKPOINT COMPLETE' : 'BLOCK COMPLETE'}</div>
    </div>
    <div class="summary">
      {#if cp}
        <!-- checkpoint debrief (GDD §3): pass grants mastery; failure UX is
             no-punishment copy — retry date + automatic spaced repetition -->
        {#if cp.passed}
          <p class="lede">
            Checkpoint passed — <b>{DRILL_TITLE[block.drillId]}</b> mastered. {cp.nCorrect}/{block.size} at gate difficulty.
          </p>
        {:else}
          <p class="lede">
            {cp.nCorrect}/{block.size}{cp.cardinals > 0 ? ` with ${cp.cardinals} cardinal` : ''} — this gate needs 8/{block.size}{cp.cardinals > 0 ? ' and zero cardinal errors' : ''}.
          </p>
          <p class="lede sub-lede">
            Checkpoint retry available {cp.retryAt !== null ? new Date(cp.retryAt).toLocaleDateString('en-US', { weekday: 'long' }) : 'in 2 days'} — today's misses are already queued for review.
          </p>
        {/if}
      {:else if s.calibrationLine}
        <!-- sentence first (GDD §8 screen 3): round calibration, then base rate -->
        <p class="lede">{s.calibrationLine}</p>
        <p class="lede sub-lede">{s.baseRateLine}</p>
      {:else}
        <p class="lede">{s.baseRateLine}</p>
      {/if}
      <dl class="sumgrid">
        <dt class="num">ACCURACY</dt>
        <dd class="num">{s.correct}/{s.reps}</dd>
        <dt class="num">LATENCY</dt>
        <dd class="num">{fmtS(s.medianLatencyMs)}s <span class="sub">med · par {fmtS(s.parMs)}</span></dd>
        {#if !cp}
          <dt class="num">BOT POINTS</dt>
          <dd class="num">
            {s.botPoints === null ? '—' : (s.botPoints >= 0 ? '+' : '') + s.botPoints.toFixed(1)}
            {#if s.botPoints !== null}<span class="sub">this block of {s.reps} — the dashboard scores per 25</span>{/if}
          </dd>
        {/if}
        <dt class="num">RATING</dt>
        <dd class="num">
          {startRating.toLocaleString('en-US')} → {rating.toLocaleString('en-US')}
          {#if provisional}<span class="sub">provisional (±{startRd} → ±{ratingRd}) — early swings run large until the deviation settles</span>{/if}
        </dd>
      </dl>

      <!-- film strip of every profile judged, tap to replay (GDD §2/§8) -->
      <div class="striplabel num">FILM STRIP · TAP TO REPLAY</div>
      <div class="strip">
        {#each block.reps as rep, i (rep.answer.itemId)}
          <button
            class="thumb"
            class:active={filmIdx === i}
            onclick={() => tapThumb(i)}
            aria-label={`Replay rep ${i + 1}`}
          >
            <canvas bind:this={thumbs[i]}></canvas>
            <span class="mark num" class:ok={rep.verdict.correct}>{rep.verdict.correct ? '✓' : '✗'}</span>
          </button>
        {/each}
      </div>
      {#if filmIdx !== null && block.reps[filmIdx]}
        {@const rep = block.reps[filmIdx]}
        <div class="film">
          <div class="film-canvas"><canvas bind:this={filmCanvas}></canvas></div>
          <div class="film-line">
            <span class="glyph num" class:ok={rep.verdict.correct}>{rep.verdict.correct ? '✓' : '✗'}</span>
            <span class="txt"
              >{typeof rep.answer.choice === 'number'
                ? rowToPrice(rep.li.item.stimulus.profile, rep.answer.choice).toFixed(2)
                : rep.answer.choice}
              — {typeof rep.li.item.groundTruth === 'number'
                ? rowToPrice(rep.li.item.stimulus.profile, rep.li.item.groundTruth).toFixed(2)
                : rep.li.item.groundTruth}</span
            >
            <span class="rep num">REP {filmIdx + 1}</span>
          </div>
        </div>
      {/if}

      {#if s.misses > 0}
        <p class="queued num">{s.misses} {s.misses === 1 ? 'MISS' : 'MISSES'} QUEUED → TOMORROW</p>
      {:else}
        <p class="queued num">CLEAN BLOCK — NOTHING QUEUED</p>
      {/if}
      {#if s.cardinals > 0}
        <p class="queued cardinal num">{s.cardinals} CARDINAL {s.cardinals === 1 ? 'ERROR' : 'ERRORS'} · 3× RATING WEIGHT</p>
      {/if}
    </div>
    <div class="footer summary-actions">
      {#if cp}
        <!-- no instant retry: the checkpoint re-arms per GDD §3 timing -->
        <a class="next" href="#/">BACK TO TREE<span class="arrow">▸</span></a>
      {:else}
        <button
          class="next"
          onclick={() => {
            window.location.hash = `#/drill?drill=${block?.drillId}`;
            void startBlock();
          }}>AGAIN<span class="arrow">▸</span></button
        >
        <a class="next ghost" href="#/">HOME</a>
      {/if}
    </div>
  {:else}
    <!-- top chrome -->
    <div class="topbar">
      <div class="mode num">
        <b>{isCp ? 'Checkpoint' : block?.mode === 'woodpecker' ? 'Woodpecker' : 'Rated'}</b> · {block ? DRILL_TITLE[block.drillId] : '…'}
      </div>
      <div class="top-stats">
        <div class="stat">
          <span class="k">Rating</span>
          <span class="v hero num">{rating.toLocaleString('en-US')}</span>
          {#if !isCp && block?.ratingDelta !== null && block?.ratingDelta !== undefined}
            <span class="delta num">{block.ratingDelta >= 0 ? `↑${block.ratingDelta}` : `↓${-block.ratingDelta}`}</span>
          {/if}
          {#if provisional}
            <span class="delta num prov" title="Rating deviation is still high — early hits and misses swing the number hard until it settles">±{ratingRd} prov</span>
          {/if}
        </div>
        <svg
          class="parring"
          viewBox="0 0 16 16"
          role="img"
          aria-label={item ? `Par time ${fmtS(item.parMs)}s` : 'Par time'}
        >
          <circle cx="8" cy="8" r="6.5" fill="none" stroke="var(--hairline)" stroke-width="2" />
          {#if parProgress > 0.005}
            <path d={arcPath(parProgress)} fill="none" stroke="var(--ink-muted)" stroke-width="2" stroke-linecap="round" />
          {/if}
        </svg>
      </div>
    </div>

    <!-- block progress -->
    <div class="blockrow">
      <div class="ticks">
        {#each Array(block?.size ?? 10) as _, i (i)}
          <div class="tick" class:done={block ? i < block.rep - 1 : false} class:now={block ? i === block.rep - 1 : false}></div>
        {/each}
      </div>
      <div class="blocklabel num">
        {isCp ? `GATE ${block?.rep ?? 1}/${block?.size ?? 10}` : `REP ${block?.rep ?? 1}/${block?.size ?? 10}`}
      </div>
    </div>

    <!-- chart panel (checkpoint = visually distinct slate frame, GDD §3) -->
    <div class="chart" class:cpframe={isCp}>
      <div class="chart-head">
        <span class="q">{item?.question ?? 'Generating…'}</span>
        <span class="meta num">{metaLine()}</span>
      </div>
      <div class="chart-body" bind:this={bodyEl}>
        <canvas
          bind:this={canvasEl}
          class:tappable={isSnap && block?.phase === 'armed'}
          onpointerdown={onCanvasDown}
        ></canvas>
      </div>
    </div>

    <!-- verdict layer (suppressed during checkpoints — juice-free, GDD §3) -->
    {#if block?.phase === 'verdict' && block.verdict && !isCp}
      {@const v = block.verdict}
      <div class="verdict">
        <div class="verdict-head">
          <div class="verdict-line">
            <span class="glyph" class:ok={v.correct}>{v.correct ? '✓' : '✗'}</span>
            {#if v.correct}
              <span class="you ok"
                >{isSnap ? 'You: ' : 'You said '}<span class="shape num">{pickText()}</span></span
              >
              <span class="dash">—</span>
              <span class="ans">{v.score === 100 ? 'exact' : v.score === 60 ? 'within a row' : 'judgment call'}</span>
            {:else}
              <span class="you">{isSnap ? 'You: ' : 'You said '}<span class="shape num">{pickText()}</span></span>
              <span class="dash">—</span>
              <span class="ans">answer <span class="shape truth num">{truthText()}</span></span>
            {/if}
          </div>
          <div class="latency num">
            <b>{fmtS(block.lastAnswer?.latencyMs ?? 0)}s</b> · par {fmtS(item?.parMs ?? 0)}
          </div>
        </div>
        <p class="explain">
          {#each explainSegs(v.explanation) as seg, i (i)}{#if seg.term}<span class="term">{seg.t}</span
            >{:else}{seg.t}{/if}{/each}
        </p>
        <button class="replay" onclick={toggleReplay}
          ><span class="g">⟲</span>{replaying ? 'Pause replay' : replayLabel(block.drillId)}</button
        >
      </div>
    {/if}

    <!-- answers -->
    {#if item && item.choices.length > 0}
      <div class="answers">
        {#each item.choices as c (c)}
          {@const st = chipState(c)}
          <button
            class="chip"
            class:dimmed={st === 'dimmed'}
            class:picked={st === 'picked'}
            class:pickedok={st === 'picked' && block?.verdict?.correct}
            class:truth={st === 'truth'}
            class:committed={st === 'committed'}
            onpointerdown={() => commit(c)}
            onclick={() => commit(c)}
            disabled={block?.phase !== 'armed' && st === 'dimmed'}
          >
            <span class={letterClass(c)}>{c}</span>
            {#if st === 'picked'}
              <span class="badge">{block?.verdict?.correct ? '✓' : '✗'}</span>
              <span class="who">YOU</span>
            {:else if st === 'truth'}
              <span class="badge">✓</span>
            {:else if st === 'committed'}
              <span class="who neutral">LOGGED</span>
            {/if}
          </button>
        {/each}
      </div>
    {:else if item}
      <div class="tapline num">TAP THE CHART · SNAP TO ROW</div>
    {/if}

    <!-- confidence ride-along (chip drills; GDD §2/§5) -->
    {#if item && item.choices.length > 0 && block}
      <div class="confrow">
        <span class="k">Conf</span>
        {#each CONF as c (c)}
          <button
            class="seg"
            class:on={block.phase === 'armed' ? block.confidence === c : block.lastAnswer?.confidence === c}
            class:live={block.phase === 'armed'}
            onpointerdown={() => {
              if (block && block.phase === 'armed') block.confidence = c;
            }}
          >
            {c}
          </button>
        {/each}
        {#if block.phase === 'verdict' && block.lastAnswer?.confidence && !isCp}
          <span class="logged num">logged {Math.round(confidenceProb(block.lastAnswer.confidence) * 100)}%</span>
        {/if}
      </div>
    {/if}

    <!-- footer CTA (checkpoints auto-advance — no button until the debrief) -->
    <div class="footer">
      {#if isCp}
        <p class="cpnote num">GATE EXAM — NO PER-ITEM FEEDBACK · FULL DEBRIEF AT THE END</p>
      {:else if block?.phase === 'verdict'}
        <button class="next" onclick={() => block?.advanceRep()}
          >{block.rep >= block.size ? 'RESULTS' : 'NEXT'}<span class="arrow">▸</span></button
        >
      {/if}
    </div>
  {/if}
</section>

<style>
  .page {
    flex: 1;
    min-height: 0;
    height: calc(100dvh - var(--control-xl));
    width: 100%;
    max-width: 480px;
    margin: 0 auto;
    padding: 0 var(--s4);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ---------- top chrome (mockup .topbar) ---------- */
  .topbar {
    height: 52px;
    flex: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-top: var(--s3);
  }
  .mode {
    font-size: var(--text-xs);
    font-weight: var(--weight-regular);
    letter-spacing: var(--track-label);
    text-transform: uppercase;
    color: var(--ink-muted);
  }
  .mode b {
    color: var(--ink-body);
    font-weight: var(--weight-semibold);
  }
  .top-stats {
    display: flex;
    align-items: center;
    gap: var(--s4);
  }
  .stat {
    display: flex;
    align-items: baseline;
    gap: 6px;
  }
  .stat .k {
    font-size: var(--text-xs);
    letter-spacing: var(--track-label);
    text-transform: uppercase;
    color: var(--ink-muted);
    font-weight: var(--weight-medium);
  }
  .stat .v {
    font-size: var(--text-body);
    font-weight: var(--weight-semibold);
    color: var(--ink-body);
  }
  .stat .v.hero {
    color: var(--ink-hero);
  }
  .stat .delta {
    font-size: var(--text-xs);
    font-weight: var(--weight-regular);
    color: var(--ink-muted);
  }
  .stat .delta.prov {
    white-space: nowrap;
    font-size: var(--text-2xs);
    letter-spacing: var(--track-caption);
  }
  .parring {
    width: 16px;
    height: 16px;
    flex: none;
    display: block;
  }

  /* ---------- block ticks ---------- */
  .blockrow {
    flex: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--s2) 0 var(--s3);
  }
  .ticks {
    display: flex;
    gap: 5px;
  }
  .tick {
    width: 18px;
    height: 3px;
    border-radius: var(--radius-xs);
    background: var(--surface-2);
  }
  .tick.done {
    background: var(--ink-ghost);
  }
  .tick.now {
    background: var(--ink-hero);
  }
  .blocklabel {
    font-size: var(--text-xs);
    color: var(--ink-muted);
    letter-spacing: var(--track-caption);
  }

  /* ---------- chart panel ---------- */
  .chart {
    flex: 1;
    min-height: 0;
    background: var(--surface-1);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  /* checkpoint gate exam: visually distinct slate frame (GDD §3 mode 6) */
  .chart.cpframe {
    border: 2px solid var(--ink-ghost);
    background: var(--surface-0);
  }
  .chart-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--s3);
    padding: var(--s3) var(--s4) 0;
  }
  .chart-head .q {
    font-size: var(--text-body-sm);
    font-weight: var(--weight-medium);
    color: var(--ink-body);
    letter-spacing: var(--track-body);
  }
  .chart-head .meta {
    font-size: var(--text-xs);
    color: var(--ink-muted);
    letter-spacing: var(--track-meta);
    white-space: nowrap;
  }
  .chart-body {
    flex: 1;
    min-height: 0;
    position: relative;
  }
  .chart-body canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    touch-action: manipulation;
  }
  .chart-body canvas.tappable {
    cursor: crosshair;
  }

  /* ---------- verdict layer ---------- */
  .verdict {
    flex: none;
    padding: var(--s4) 0 0;
  }
  .verdict-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
  }
  .verdict-line {
    font-size: var(--text-verdict);
    font-weight: var(--weight-semibold);
    display: flex;
    align-items: baseline;
    gap: 8px;
    animation: pop var(--dur-1) var(--ease-squash);
  }
  @keyframes pop {
    from {
      transform: scale(0.9);
      opacity: 0.4;
    }
    to {
      transform: scale(1);
      opacity: 1;
    }
  }
  .verdict-line .glyph {
    color: var(--sell-red);
    font-weight: var(--weight-bold);
    font-size: var(--text-body-lg);
  }
  .verdict-line .glyph.ok {
    color: var(--buy-green);
  }
  .verdict-line .you {
    color: var(--sell-red);
  }
  .verdict-line .you.ok {
    color: var(--buy-green);
  }
  .verdict-line .you .shape {
    font-weight: var(--weight-semibold);
  }
  .verdict-line .dash {
    color: var(--ink-muted);
    font-weight: var(--weight-regular);
  }
  .verdict-line .ans {
    color: var(--ink-muted);
    font-weight: var(--weight-regular);
    font-size: var(--text-body);
  }
  .verdict-line .ans .shape.truth {
    font-weight: var(--weight-semibold);
    font-size: var(--text-verdict);
    color: var(--buy-green);
  }
  .latency {
    font-size: var(--text-sm);
    color: var(--ink-muted);
    white-space: nowrap;
  }
  .latency b {
    font-weight: var(--weight-semibold);
    color: var(--ink-body);
  }
  .explain {
    margin-top: var(--s2);
    font-size: var(--text-body);
    font-weight: var(--weight-regular);
    color: var(--ink-body);
    max-width: 60ch;
    text-wrap: balance;
    animation: slidein var(--dur-2) var(--ease-out);
  }
  .explain .term {
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-weight: var(--weight-semibold);
    color: var(--ink-hero);
    font-size: var(--text-body-sm);
  }
  @keyframes slidein {
    from {
      transform: translateY(6px);
      opacity: 0;
    }
    to {
      transform: translateY(0);
      opacity: 1;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .verdict-line,
    .explain {
      animation: none;
    }
  }
  .replay {
    margin-top: 10px;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    height: 32px;
    padding: 0 var(--s3);
    background: transparent;
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
    color: var(--ink-muted);
    font-family: var(--font-ui);
    font-size: var(--text-sm);
    font-weight: var(--weight-medium);
    letter-spacing: var(--track-body);
  }
  .replay .g {
    font-size: var(--text-body);
    line-height: 1;
  }

  /* ---------- answer chips ---------- */
  .answers {
    flex: none;
    display: flex;
    gap: var(--s2);
    padding: var(--s4) 0 0;
  }
  .chip {
    flex: 1;
    height: var(--control-xl);
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--surface-2);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
    font-family: var(--font-mono);
    padding: 0 2px;
  }
  .chip .letter {
    font-size: var(--text-chip);
    font-weight: var(--weight-semibold);
    color: var(--ink-muted);
    line-height: 1.1;
  }
  .chip .letter.small {
    font-size: var(--text-sm);
    letter-spacing: var(--track-caption);
  }
  .chip .letter.tiny {
    font-size: 9px;
    letter-spacing: 0.04em;
    white-space: normal;
    text-align: center;
  }
  .chip.dimmed {
    opacity: 0.45;
  }
  .chip .badge {
    position: absolute;
    top: -7px;
    right: -6px;
    width: 16px;
    height: 16px;
    border-radius: var(--radius-full);
    font-family: var(--font-ui);
    font-size: var(--text-2xs);
    font-weight: var(--weight-bold);
    line-height: 16px;
    text-align: center;
    border: 2px solid var(--surface-0);
    box-sizing: content-box;
  }
  .chip.picked {
    border-color: var(--sell-red);
    background: var(--sell-red-wash);
  }
  .chip.picked .letter {
    color: var(--sell-red);
    transform: translateY(-5px);
  }
  .chip.picked .badge {
    background: var(--sell-red);
    color: var(--surface-0);
  }
  .chip.picked .who {
    position: absolute;
    bottom: 5px;
    left: 0;
    right: 0;
    text-align: center;
    font-family: var(--font-ui);
    font-size: var(--text-xs);
    font-weight: var(--weight-semibold);
    letter-spacing: var(--track-kicker);
    color: var(--sell-red);
    line-height: 1;
  }
  .chip.picked.pickedok {
    border-color: var(--buy-green);
    background: var(--buy-green-wash);
  }
  .chip.picked.pickedok .letter,
  .chip.picked.pickedok .who {
    color: var(--buy-green);
  }
  .chip.picked.pickedok .badge {
    background: var(--buy-green);
  }
  /* checkpoint: the tap registers without revealing right/wrong (GDD §3) */
  .chip.committed {
    border-color: var(--ink-muted);
    background: var(--surface-1);
  }
  .chip.committed .letter {
    color: var(--ink-body);
    transform: translateY(-5px);
  }
  .chip.committed .who.neutral {
    position: absolute;
    bottom: 5px;
    left: 0;
    right: 0;
    text-align: center;
    font-family: var(--font-ui);
    font-size: var(--text-2xs);
    font-weight: var(--weight-semibold);
    letter-spacing: var(--track-kicker);
    color: var(--ink-muted);
    line-height: 1;
  }
  .chip.truth {
    border-color: var(--buy-green);
  }
  .chip.truth .letter {
    color: var(--buy-green);
  }
  .chip.truth .badge {
    background: var(--buy-green);
    color: var(--surface-0);
  }
  .tapline {
    flex: none;
    height: var(--control-xl);
    margin-top: var(--s4);
    display: flex;
    align-items: center;
    justify-content: center;
    border: var(--hairline-w) dashed var(--hairline);
    border-radius: var(--radius);
    color: var(--ink-muted);
    font-size: var(--text-xs);
    letter-spacing: var(--track-label);
  }

  /* ---------- confidence ride-along ---------- */
  .confrow {
    flex: none;
    display: flex;
    align-items: center;
    gap: var(--s2);
    padding: var(--s3) 0 0;
  }
  .confrow .k {
    font-size: var(--text-xs);
    letter-spacing: var(--track-label);
    text-transform: uppercase;
    color: var(--ink-muted);
    font-weight: var(--weight-medium);
    margin-right: 2px;
  }
  .seg {
    height: 30px;
    padding: 0 14px;
    display: inline-flex;
    align-items: center;
    border: var(--hairline-w) solid transparent;
    border-radius: var(--radius);
    font-size: var(--text-sm);
    font-weight: var(--weight-medium);
    color: var(--ink-muted);
    opacity: 0.5;
  }
  .seg.live {
    height: var(--hit-min);
  }
  .seg.on {
    border-color: var(--hairline);
    background: var(--surface-2);
    color: var(--ink-body);
    opacity: 1;
  }
  .confrow .logged {
    margin-left: auto;
    font-size: var(--text-xs);
    color: var(--ink-muted);
  }

  /* ---------- footer ---------- */
  .footer {
    flex: none;
    padding: var(--s3) 0 var(--s4);
    min-height: calc(var(--control-lg) + var(--s3) + var(--s4));
  }
  .next {
    width: 100%;
    height: var(--control-lg);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    background: var(--surface-2);
    color: var(--ink-hero);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
    font-size: var(--text-body);
    font-weight: var(--weight-semibold);
    letter-spacing: var(--track-kicker);
  }
  .next .arrow {
    font-size: var(--text-sm);
    color: var(--ink-muted);
  }
  /* checkpoint note in the (otherwise empty) footer — juice-free copy */
  .cpnote {
    height: var(--control-lg);
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    font-size: var(--text-2xs);
    letter-spacing: var(--track-label);
    color: var(--ink-muted);
  }

  /* ---------- summary ---------- */
  .summary {
    flex: 1;
    min-height: 0;
    background: var(--surface-1);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
    padding: var(--s5) var(--s4);
    display: flex;
    flex-direction: column;
    gap: var(--s4);
    overflow-y: auto;
  }
  .lede {
    font-size: var(--text-body-lg);
    font-weight: var(--weight-medium);
    color: var(--ink-body);
    max-width: 40ch;
    text-wrap: balance;
  }
  .lede.sub-lede {
    font-size: var(--text-body);
    font-weight: var(--weight-regular);
    color: var(--ink-muted);
    margin-top: calc(-1 * var(--s3));
  }

  /* ---------- film strip (GDD §2 Permanence / §8 screen 3) ---------- */
  .striplabel {
    font-size: var(--text-2xs);
    letter-spacing: var(--track-label);
    color: var(--ink-muted);
    margin-bottom: calc(-1 * var(--s3));
  }
  .strip {
    display: flex;
    gap: var(--s2);
    flex: none;
  }
  .thumb {
    flex: 1;
    min-width: 0;
    height: 44px;
    position: relative;
    padding: 2px 2px 12px;
    background: var(--surface-2);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius-xs);
  }
  .thumb.active {
    border-color: var(--ink-muted);
  }
  .thumb canvas {
    display: block;
    width: 100%;
    height: 100%;
  }
  .thumb .mark {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    text-align: center;
    font-size: var(--text-2xs);
    font-weight: var(--weight-bold);
    line-height: 12px;
    color: var(--sell-red);
  }
  .thumb .mark.ok {
    color: var(--buy-green);
  }
  .film {
    flex: none;
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
    overflow: hidden;
  }
  .film-canvas {
    position: relative;
    height: 148px;
  }
  .film-canvas canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }
  .film-line {
    display: flex;
    align-items: baseline;
    gap: var(--s2);
    padding: var(--s2) var(--s3);
    border-top: var(--hairline-w) solid var(--hairline);
    font-size: var(--text-sm);
    color: var(--ink-body);
  }
  .film-line .glyph {
    color: var(--sell-red);
    font-weight: var(--weight-bold);
  }
  .film-line .glyph.ok {
    color: var(--buy-green);
  }
  .film-line .rep {
    margin-left: auto;
    color: var(--ink-muted);
    font-size: var(--text-xs);
    letter-spacing: var(--track-caption);
  }
  .sumgrid {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: var(--s3) var(--s4);
    align-items: baseline;
  }
  .sumgrid dt {
    font-size: var(--text-2xs);
    letter-spacing: var(--track-label);
    color: var(--ink-muted);
  }
  .sumgrid dd {
    font-size: var(--text-verdict);
    font-weight: var(--weight-semibold);
    color: var(--ink-hero);
  }
  .sumgrid dd .sub {
    font-size: var(--text-sm);
    font-weight: var(--weight-regular);
    color: var(--ink-muted);
  }
  .queued {
    font-size: var(--text-xs);
    letter-spacing: var(--track-caption);
    color: var(--ink-muted);
  }
  .queued.cardinal {
    color: var(--highlight);
  }
  .summary-actions {
    display: flex;
    gap: var(--s2);
  }
  .summary-actions .next {
    flex: 1;
  }
  .next.ghost {
    background: transparent;
    color: var(--ink-muted);
  }
</style>
