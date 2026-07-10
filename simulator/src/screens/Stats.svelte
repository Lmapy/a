<script lang="ts">
  /* Stats dashboard (GDD §5 surfaces, §8 screen 8), pixel-matched to
     design/mockups/screen-dashboard.html: sentence-led calibration
     reliability plot (Canvas 2D), Bot Points + Brier trend sparklines,
     per-node Glicko list with deltas, the expectancy ledger (cells gray
     until n ≥ 30 — "unmeasured = folklore"), R-multiple histogram from
     boss/sim trade fills. Every number is computed by @stats/model +
     @schedule from the persisted ledger; this file only binds and draws.

     Review flag: #/stats?demo=1 seeds the deterministic demo ledger. */
  import { onMount } from 'svelte';
  import MobileTopbar from '../lib/ui/MobileTopbar.svelte';
  import BottomNav from '../lib/ui/BottomNav.svelte';
  import Spark from '../lib/ui/Spark.svelte';
  import { loadAppData } from '../lib/stores/appdata';
  import type { AppData } from '../lib/stores/appdata';
  import {
    BRIER_BAR,
    SCOPE_DAYS,
    accuracyBars,
    beatingTheBot,
    bestMeasuredEdge,
    botBlockSeries,
    brierBlockSeries,
    calibrationLedger,
    calibrationView,
    expectancyTable,
    rHistogram,
    scopeDecisions,
    tradesOf,
  } from '../lib/stats/model';
  import { readStatsTheme, renderCalibration, renderRHistogram } from '../lib/render/statsCanvas';
  import { TREE_TIERS, deltaOverLastActiveDay } from '../lib/schedule/tree';
  import { player } from '../lib/stores/player.svelte';
  import type { DecisionRecord } from '../lib/types';

  let data = $state<AppData | null>(null);

  async function reload(): Promise<void> {
    data = await loadAppData();
  }

  onMount(() => {
    void reload();
    const onHash = (): void => void reload();
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  });

  /* ---------------- derivations (all @stats/model + @schedule) ------------ */

  const scoped = $derived(data ? scopeDecisions(data.decisions, data.now) : []);
  const entries = $derived(calibrationLedger(scoped));
  const cal = $derived(calibrationView(entries));
  const botBlocks = $derived(botBlockSeries(scoped));
  const brierSeries = $derived(brierBlockSeries(entries));
  const beat = $derived(beatingTheBot(botBlocks));
  const trades = $derived(tradesOf(scoped));
  const expRows = $derived(expectancyTable(trades));
  const best = $derived(bestMeasuredEdge(expRows));
  const hist = $derived(rHistogram(trades));

  const byNode = $derived.by(() => {
    const m = new Map<string, DecisionRecord[]>();
    for (const d of scoped) {
      const arr = m.get(d.nodeId);
      if (arr) arr.push(d);
      else m.set(d.nodeId, [d]);
    }
    return m;
  });

  const lastBot = $derived(botBlocks.length > 0 ? botBlocks[botBlocks.length - 1] : null);
  const botDelta = $derived(botBlocks.length > 1 ? botBlocks[botBlocks.length - 1] - botBlocks[botBlocks.length - 2] : null);
  const lastBrier = $derived(brierSeries.length > 0 ? brierSeries[brierSeries.length - 1] : null);
  const brierDelta = $derived(
    brierSeries.length > 1 ? brierSeries[brierSeries.length - 1] - brierSeries[brierSeries.length - 2] : null,
  );

  /* ---------------- formatting ---------------- */

  const fmtR = (x: number): string => `${x < 0 ? '−' : '+'}${Math.abs(x).toFixed(2)}`;
  const pct = (x: number): number => Math.round(x * 100);
  const noZero = (x: number): string => x.toFixed(3).replace(/^0/, '');
  const arrow = (x: number): string => (x >= 0 ? '↑' : '↓');

  function sparkRange(xs: number[], pad = 0.15): { lo: number; hi: number } {
    if (xs.length === 0) return { lo: 0, hi: 1 };
    const lo = Math.min(...xs);
    const hi = Math.max(...xs);
    const span = Math.max(hi - lo, 1e-9);
    return { lo: lo - span * pad, hi: hi + span * pad };
  }
  const botRange = $derived(sparkRange([...botBlocks, 0]));
  const brierRange = $derived(sparkRange([...brierSeries, BRIER_BAR]));

  /* ---------------- canvases ---------------- */

  let calCanvas = $state<HTMLCanvasElement | null>(null);
  let histCanvas = $state<HTMLCanvasElement | null>(null);

  $effect(() => {
    const c = calCanvas;
    const view = cal;
    if (!c || !data) return;
    const draw = (): void => renderCalibration(c, readStatsTheme(c), view);
    draw();
    void document.fonts?.ready.then(draw);
    const ro = new ResizeObserver(draw);
    ro.observe(c.parentElement ?? c);
    return () => ro.disconnect();
  });

  $effect(() => {
    const c = histCanvas;
    const hh = hist;
    if (!c || !data || hh.n === 0) return;
    const draw = (): void => renderRHistogram(c, readStatsTheme(c), hh);
    draw();
    void document.fonts?.ready.then(draw);
    const ro = new ResizeObserver(draw);
    ro.observe(c.parentElement ?? c);
    return () => ro.disconnect();
  });
</script>

<section class="page">
  <MobileTopbar />

  <div class="scope num">
    last {SCOPE_DAYS} days · all modes · <b>{scoped.length}</b> scored judgments
  </div>

  <div class="grid">
    <!-- ============ calibration curve ============ -->
    <div class="card" id="calibration">
      <div class="kicker num">Calibration · reliability, blocks of 25</div>
      <p class="lede">
        {#if cal.headline}
          When you say <b class="num">{cal.headline.said}%</b>, it happens
          <b class="num">{cal.headline.happened}%</b> — {cal.headline.clause}.
        {:else}
          {cal.emptyLine}
        {/if}
      </p>
      <div class="chartbox tall"><canvas bind:this={calCanvas}></canvas></div>
    </div>

    <!-- ============ brier / bot points trend ============ -->
    <div class="card" id="brier">
      <div class="kicker num">Score vs the base-rate bot</div>
      <p class="lede">
        {#if beat.of > 0}
          Beating the bot <b class="num">{beat.won}</b> of the last <span class="num">{beat.of}</span> blocks.
        {:else}
          Every drill block scores Bot Points — this trend charts them per
          <span class="num">25</span> tagged judgments once your first 25 land.
        {/if}
      </p>
      <div class="statrow">
        <div class="meta">
          <span class="k">Bot Points / block</span>
          <span class="v num">{lastBot === null ? '—' : `${lastBot >= 0 ? '+' : '−'}${Math.abs(lastBot)}`}</span>
          <span class="delta num">
            {#if botDelta !== null}{arrow(botDelta)}{Math.abs(botDelta)} vs prior block{:else}first 25-judgment block pending{/if}
          </span>
        </div>
        <Spark data={botBlocks.slice(-12)} lo={botRange.lo} hi={botRange.hi} bar={0} end={lastBot === null ? '' : String(lastBot)} />
      </div>
      <div class="statrow">
        <div class="meta">
          <span class="k">Brier · 50 items</span>
          <span class="v num">{lastBrier === null ? '—' : lastBrier.toFixed(3)}</span>
          <span class="delta num">
            {#if brierDelta !== null}{arrow(brierDelta)}{noZero(Math.abs(brierDelta))} · bar {noZero(BRIER_BAR)}{:else}bar {noZero(BRIER_BAR)}{/if}
          </span>
        </div>
        <Spark data={brierSeries.slice(-12)} lo={brierRange.lo} hi={brierRange.hi} bar={BRIER_BAR} end={lastBrier === null ? '' : noZero(lastBrier)} />
      </div>
      <p class="caption">Most players normalize within 100–200 judgments — you're at {entries.length}.</p>
    </div>

    <!-- ============ per-skill ratings ============ -->
    <div class="card" id="skills">
      <div class="kicker num">Skill ratings · Glicko-2 per node</div>
      <div class="skilllist">
        {#if data}
          {#each TREE_TIERS as tier (tier.key)}
            <div class="tierlabel num">{tier.key} · {tier.title.toLowerCase()}</div>
            {#each tier.nodes as nd (nd.id)}
              {@const p = data.tree.nodes[nd.id]}
              {@const bars = accuracyBars(byNode.get(nd.id) ?? [])}
              {@const delta = deltaOverLastActiveDay(data.replay.history[nd.id] ?? [])}
              <div class="skill" class:locked={p.state === 'locked'}>
                <span class="name">{nd.name}</span>
                {#if p.state === 'rusty'}<span class="tag num">RUSTY</span>{/if}
                {#if p.state === 'checkpoint-armed'}<span class="tag num">ARMED</span>{/if}
                {#if bars}
                  {@const mx = Math.max(...bars)}
                  <span class="bars">
                    {#each bars as b, bi (bi)}
                      <i class:hi={b === mx} style:height={`${Math.max(2, Math.round(b * 12))}px`}></i>
                    {/each}
                  </span>
                {/if}
                <span class="rating num">{p.state === 'locked' ? '—' : Math.round(player.rating(nd.id).rating)}</span>
                <span class="delta num">{delta !== 0 ? `${arrow(delta)}${Math.abs(delta)}` : ''}</span>
              </div>
            {/each}
          {/each}
        {/if}
      </div>
      <div class="divergence">
        <div class="kicker num" style:margin-bottom="4px">Drill rating vs boss Read Score</div>
        <p class="ghostnote num">unmeasured = folklore — divergence plots once boss Read Scores land</p>
      </div>
      <p class="caption">Rated drills only move these. Rush, Streak and Warm-Up never touch rating.</p>
    </div>

    <!-- ============ expectancy table ============ -->
    <div class="card" id="expectancy">
      <div class="kicker num">Expectancy ledger · setup × regime</div>
      <p class="lede wide">
        {#if best}
          <b>{best.setup}</b> in {best.regime} is your best measured edge:
          <b class="num">{fmtR(best.cell.expectancy)}R</b> over {best.cell.n} trades.
        {:else}
          No measured edge yet — cells unlock at n ≥ 30 logged sim trades.
        {/if}
      </p>
      <div class="tablebox">
        <table>
          <colgroup><col class="c-setup" /><col class="c-reg" /><col class="c-reg" /><col class="c-folk" /></colgroup>
          <thead>
            <tr><th>Setup</th><th>Balance</th><th>Imbalance</th><th>Folklore win%</th></tr>
          </thead>
          <tbody>
            {#each expRows as row (row.setup)}
              <tr>
                <td class="setup">{row.setup}</td>
                {#each [row.balance, row.imbalance] as cell, ci (ci)}
                  {#if cell && cell.measured}
                    <td class="cell num">
                      <span class="e" class:neg={cell.expectancy < 0}>{fmtR(cell.expectancy)}R</span>
                      <span class="sub">· {pct(cell.winRate)}% · n {cell.n}</span>
                    </td>
                  {:else}
                    <td class="unmeasured num">unmeasured = folklore</td>
                  {/if}
                {/each}
                <td class="folk num">{row.folklore ?? '—'}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
      <p class="caption">
        Cells stay gray until n ≥ 30. In-sim expectancy is certifiable; real-market expectancy is not.
        Folklore quotes are the guide's recorded claims — the 80% rule measures ~60–67% in community tests.
      </p>
    </div>

    <!-- ============ r-multiple histogram ============ -->
    <div class="card" id="rmult">
      <div class="kicker num">R-multiples · {hist.n} sim trades</div>
      {#if hist.n > 0}
        <p class="lede">Avg <b class="num">{fmtR(hist.avg)}R</b> per trade. P&amp;L is shown always, scored never.</p>
        <div class="legend num">
          <span class="key"><span class="sw win"></span>wins</span>
          <span class="key"><span class="sw loss"></span>losses</span>
        </div>
        <div class="chartbox"><canvas bind:this={histCanvas}></canvas></div>
        <p class="caption">Outcome ≠ decision quality — only Read Scores are graded.</p>
      {:else}
        <div class="emptybox">
          <span class="ghostnote num">unmeasured = folklore — R-multiples land with boss/sim trades</span>
        </div>
      {/if}
    </div>
  </div>

  <BottomNav />
</section>

<style>
  .page {
    flex: 1;
    width: 100%;
    max-width: 1440px;
    margin: 0 auto;
    padding: 0 var(--s4) var(--s5);
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  @media (max-width: 479px) {
    .page {
      padding-bottom: calc(var(--control-xl) + var(--s4));
    }
  }
  @media (min-width: 1024px) {
    .page {
      padding: 0 var(--s5) var(--s5);
    }
  }

  .scope {
    font-size: var(--text-xs);
    letter-spacing: var(--track-caption);
    color: var(--ink-muted);
    text-align: right;
    padding: var(--s2) 0;
  }
  .scope b {
    font-weight: var(--weight-semibold);
    color: var(--ink-body);
  }

  /* ---------- grid ---------- */
  .grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: var(--s4);
    min-width: 0;
  }
  @media (min-width: 1024px) {
    .grid {
      grid-template-columns: repeat(12, 1fr);
      grid-template-rows: 380px minmax(0, 1fr);
    }
    #calibration {
      grid-column: 1 / 7;
      grid-row: 1;
    }
    #brier {
      grid-column: 7 / 10;
      grid-row: 1;
    }
    #skills {
      grid-column: 10 / 13;
      grid-row: 1 / 3;
    }
    #expectancy {
      grid-column: 1 / 7;
      grid-row: 2;
    }
    #rmult {
      grid-column: 7 / 10;
      grid-row: 2;
    }
  }

  .card {
    background: var(--elev-panel);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
    padding: var(--s3) var(--s5);
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
    overflow: hidden;
  }
  .kicker {
    font-size: var(--text-xs);
    letter-spacing: var(--track-kicker);
    text-transform: uppercase;
    color: var(--ink-muted);
    margin-bottom: var(--s2);
  }
  .lede {
    font-size: var(--text-body);
    color: var(--ink-body);
    max-width: 60ch;
    margin-bottom: var(--s2);
  }
  .lede.wide {
    max-width: none;
  }
  .lede b {
    font-weight: var(--weight-semibold);
    color: var(--ink-hero);
  }
  .lede b.num {
    font-size: var(--text-body-sm);
  }
  .caption {
    font-size: var(--text-sm);
    color: var(--ink-muted);
    margin-top: auto;
    padding-top: var(--s2);
  }

  .chartbox {
    flex: 1;
    min-height: 160px;
    position: relative;
  }
  .chartbox.tall {
    min-height: 260px;
  }
  .chartbox canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    display: block;
  }
  .emptybox {
    flex: 1;
    min-height: 120px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .ghostnote {
    font-size: var(--text-2xs);
    letter-spacing: var(--track-caption);
    color: var(--ink-ghost);
  }

  /* ---------- brier / bot points ---------- */
  .statrow {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--s4) 0;
  }
  .statrow + .statrow {
    border-top: var(--hairline-w) solid var(--hairline);
  }
  .statrow .meta {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }
  .statrow .k {
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    letter-spacing: var(--track-label);
    text-transform: uppercase;
    color: var(--ink-muted);
  }
  .statrow .v {
    font-size: var(--text-stat);
    font-weight: var(--weight-semibold);
    color: var(--ink-hero);
    line-height: var(--leading-tight);
  }
  .statrow .delta {
    font-size: var(--text-sm);
    color: var(--ink-muted);
    white-space: nowrap;
  }

  /* ---------- skill list ---------- */
  .skilllist {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
  }
  .tierlabel {
    font-size: var(--text-2xs);
    letter-spacing: var(--track-overline);
    color: var(--ink-ghost);
    text-transform: uppercase;
    padding: var(--s4) 0 var(--s1);
  }
  .tierlabel:first-of-type {
    padding-top: var(--s2);
  }
  .skill {
    display: flex;
    align-items: center;
    gap: 10px;
    height: 34px;
  }
  .skill .name {
    flex: 1;
    font-size: var(--text-body-sm);
    color: var(--ink-body);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .skill.locked .name {
    color: var(--ink-ghost);
  }
  .skill .tag {
    font-size: var(--text-2xs);
    letter-spacing: var(--track-label);
    color: var(--highlight);
  }
  .skill .bars {
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 12px;
    width: 22px;
  }
  .skill .bars i {
    width: 4px;
    background: var(--vol-blue-dim);
    border-radius: var(--radius-bar);
  }
  .skill .bars i.hi {
    background: var(--vol-blue);
  }
  .skill .rating {
    font-size: var(--text-body-sm);
    font-weight: var(--weight-semibold);
    color: var(--ink-body);
    width: 40px;
    text-align: right;
  }
  .skill.locked .rating {
    color: var(--ink-ghost);
    font-weight: var(--weight-regular);
  }
  .skill .delta {
    font-size: var(--text-xs);
    color: var(--ink-muted);
    width: 28px;
    text-align: right;
  }
  .divergence {
    border-top: var(--hairline-w) solid var(--hairline);
    margin-top: var(--s4);
    padding-top: var(--s3);
  }

  /* ---------- expectancy table ---------- */
  .tablebox {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    table-layout: fixed;
  }
  th,
  td {
    padding: 0 var(--s3);
    text-align: right;
    vertical-align: middle;
  }
  thead th {
    font-size: var(--text-2xs);
    font-weight: var(--weight-regular);
    font-family: var(--font-mono);
    letter-spacing: var(--track-overline);
    text-transform: uppercase;
    color: var(--ink-muted);
    padding-bottom: var(--s2);
    border-bottom: var(--hairline-w) solid var(--hairline);
  }
  thead th:first-child {
    text-align: left;
  }
  tbody td {
    height: 30px;
    border-bottom: var(--hairline-w) solid var(--surface-2);
  }
  tbody tr:last-child td {
    border-bottom: none;
  }
  td.setup {
    text-align: left;
    font-size: var(--text-body-sm);
    font-weight: var(--weight-medium);
    color: var(--ink-body);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  td.cell {
    font-size: var(--text-sm);
    color: var(--ink-body);
    white-space: nowrap;
  }
  td.cell .e {
    font-weight: var(--weight-semibold);
    color: var(--ink-hero);
  }
  td.cell .e.neg {
    color: var(--ink-body);
  }
  td.cell .sub {
    color: var(--ink-muted);
  }
  td.unmeasured {
    font-size: var(--text-2xs);
    letter-spacing: 0.05em;
    color: var(--ink-ghost);
    text-align: center;
  }
  td.folk {
    font-size: var(--text-xs);
    color: var(--ink-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  col.c-setup {
    width: 25%;
  }
  col.c-reg {
    width: 27%;
  }
  col.c-folk {
    width: 21%;
  }

  /* ---------- r-multiple ---------- */
  .legend {
    display: flex;
    gap: var(--s4);
    margin-bottom: var(--s2);
    font-size: var(--text-xs);
    color: var(--ink-muted);
    letter-spacing: var(--track-meta);
  }
  .legend .key {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .legend .sw {
    width: 10px;
    height: 10px;
    border-radius: var(--radius-xs);
  }
  .legend .sw.win {
    background: var(--vol-blue);
  }
  .legend .sw.loss {
    background: var(--vol-blue-dim);
  }
</style>
