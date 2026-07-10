<script lang="ts">
  /* Home = skill tree (GDD §3/§8 screen 1+4), pixel-matched to
     design/mockups/screen-skill-tree.html: Daily Warm-Up card with streak
     pill, T0–T3 subway tiers with live node states derived from the ledger
     (schedule/tree), per-node rating chips, T4/T5 ghost silhouettes, mobile
     bottom nav. Every state/number on screen is a schedule-lib derivation —
     nothing hand-placed. Tapping an unlocked, wired node opens its drill.

     Review flag: #/?demo=1 seeds the deterministic demo ledger (in-memory
     sandbox, never the player's real data). */
  import { onMount } from 'svelte';
  import MobileTopbar from '../lib/ui/MobileTopbar.svelte';
  import BottomNav from '../lib/ui/BottomNav.svelte';
  import { loadAppData, demoFlag } from '../lib/stores/appdata';
  import type { AppData } from '../lib/stores/appdata';
  import {
    SILHOUETTE_TIERS,
    TREE_TIERS,
    deltaOverLastActiveDay,
    gateAccuracyOf,
    tierMastery,
  } from '../lib/schedule/tree';
  import type { NodeProgress } from '../lib/schedule/tree';
  import { assembleWarmUp, startOfDay } from '../lib/schedule/queue';
  import type { WarmUpPlan } from '../lib/schedule/queue';
  import { LOOP_DRILLS } from '../lib/drills/items';
  import { player } from '../lib/stores/player.svelte';
  import type { DrillId } from '../lib/types';

  let data = $state<AppData | null>(null);

  async function reload(): Promise<void> {
    data = await loadAppData();
  }

  onMount(() => {
    void reload();
    // ?demo=1 can be toggled without changing the route — reload on hash change
    const onHash = (): void => void reload();
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  });

  const NODE_NAMES: Record<string, string> = Object.fromEntries(
    TREE_TIERS.flatMap((t) => t.nodes.map((n) => [n.id, n.name])),
  );

  /** Static sub-lines for locked gate nodes (GDD §3) + v1.1 placeholders. */
  const LOCKED_SUB: Partial<Record<DrillId, string>> = {
    'regime-gate': 'Gated by the calibration intro module',
    'setup-picker': 'Gated by Boss 1 (T2 → T3)',
    // v1.1 nodes: visible on the map, never gate progression (schedule/tree)
    'hvn-lvn-marker': 'Ships v1.1 — folded into POC/VA Snap feedback for now',
    'one-timeframing-buzzer': 'Ships v1.1 — does not gate the line',
    'acceptance-clock': 'Ships v1.1 — does not gate the line',
    'calibration-range': 'Ships v1.1 — does not gate the line',
  };

  const wired = new Set<string>(LOOP_DRILLS);

  /** Where a node tap goes: the armed-and-ready gate exam, else its drill. */
  function nodeHref(p: NodeProgress): string {
    const demo = demoFlag() ? '&demo=1' : '';
    const ready =
      p.state === 'checkpoint-armed' &&
      p.checkpointAvailableAt !== null &&
      data !== null &&
      p.checkpointAvailableAt <= data.now;
    return ready ? `#/drill?drill=${p.id}&mode=checkpoint${demo}` : `#/drill?drill=${p.id}${demo}`;
  }

  /* ---------------- warm-up assembly (schedule lib) ---------------- */

  const plan = $derived.by((): WarmUpPlan | null => {
    if (!data) return null;
    const learning = Object.values(data.tree.nodes)
      .filter((n) => n.state === 'learning')
      .map((n) => n.id);
    const mastered = Object.values(data.tree.nodes)
      .filter((n) => n.state === 'mastered' || n.state === 'rusty')
      .map((n) => n.id);
    return assembleWarmUp(data.queue, learning, mastered, data.now);
  });

  /** ['3 min', <emphasized part>, <trailing part?>] — joined with ' · '. */
  const warmupParts = $derived.by((): string[] => {
    if (!plan) return ['3 min'];
    const parts: string[] = ['3 min'];
    if (plan.woodpecker.length > 0) {
      const first = plan.woodpecker[0];
      parts.push(`${plan.woodpecker.length} miss${plan.woodpecker.length === 1 ? '' : 'es'} due`);
      parts.push(`${NODE_NAMES[first.nodeId] ?? first.nodeId} review`);
    } else if (plan.newSkillNodeId) {
      parts.push(`new skill: ${NODE_NAMES[plan.newSkillNodeId] ?? plan.newSkillNodeId}`);
      if (plan.mixedNodeIds.length > 0) parts.push('mixed block');
    } else {
      parts.push('mixed block');
    }
    return parts;
  });

  const startHref = $derived.by((): string => {
    const target =
      plan?.woodpecker[0]?.drillId ??
      (data?.tree.activeNodeId && wired.has(data.tree.activeNodeId)
        ? data.tree.activeNodeId
        : 'shape-alphabet');
    const demo = demoFlag() ? '&demo=1' : '';
    return `#/drill?drill=${target}${demo}`;
  });

  /* ---------------- per-node display helpers ---------------- */

  function progress(id: string): NodeProgress | null {
    return data?.tree.nodes[id] ?? null;
  }

  function ratingOf(id: string): number {
    return Math.round(player.rating(id).rating);
  }

  /** Today's rating movement (0 → hidden), from the Glicko replay. */
  function deltaToday(id: string): number {
    if (!data) return 0;
    const hist = data.replay.history[id];
    if (!hist || hist.length === 0) return 0;
    if (startOfDay(hist[hist.length - 1].at) !== startOfDay(data.now)) return 0;
    return deltaOverLastActiveDay(hist);
  }

  function subLine(p: NodeProgress): { amber: string; rest: string } | null {
    if (p.state === 'rusty') {
      return { amber: '⟳ Rusty', rest: ' — review queued in Warm-Up' };
    }
    if (p.state === 'checkpoint-armed') {
      const ready = p.checkpointAvailableAt !== null && data !== null && p.checkpointAvailableAt <= data.now;
      const retry = p.failedCheckpointAt !== null;
      const amber = retry ? '◉ Checkpoint retry' : '◉ Checkpoint armed';
      if (ready) return { amber, rest: ' — ready today · tap to take it' };
      if (!retry) return { amber, rest: ' — unlocks tomorrow' };
      const days =
        p.checkpointAvailableAt !== null && data !== null
          ? Math.max(1, Math.ceil((p.checkpointAvailableAt - data.now) / 86_400_000))
          : 2;
      return { amber, rest: ` — available in ${days} day${days === 1 ? '' : 's'}` };
    }
    if (p.state === 'learning' && p.accuracy !== null) {
      const acc = Math.round(p.accuracy * 100);
      const gate = Math.round(gateAccuracyOf(p.id) * 100);
      return { amber: '', rest: `In progress · ${acc}% — gate arms at ${gate}%` };
    }
    if (p.state === 'locked' && LOCKED_SUB[p.id]) {
      return { amber: '', rest: LOCKED_SUB[p.id] as string };
    }
    return null;
  }

  /** Progress-ring dasharray for the active node (circumference 2π·7 ≈ 44). */
  function ringDash(p: NodeProgress): string {
    const frac = Math.max(0, Math.min(1, p.accuracy ?? 0));
    return `${(frac * 43.98).toFixed(1)} 44`;
  }
</script>

<section class="page">
  <MobileTopbar />

  <!-- daily warm-up (design-system §3.5) -->
  <div class="warmup">
    <div class="warmup-head">
      <div class="warmup-title num">DAILY WARM-UP</div>
      <div class="streak num">STREAK <b>{player.streak.days}</b> · {player.streak.freezes} freezes</div>
    </div>
    <p class="warmup-desc">
      {#each warmupParts as part, i (i)}{#if i > 0}{' · '}{/if}{#if i === 1}<b>{part}</b>{:else}{part}{/if}{/each}
    </p>
    <a class="start" href={startHref}>START<span class="arrow">▸</span></a>
  </div>

  <!-- tree -->
  <div class="tree">
    {#if data}
      {#each TREE_TIERS as tier (tier.key)}
        {@const m = tierMastery(tier, data.tree.nodes)}
        <div class="tier">
          <div class="tier-head">
            <div class="tier-name num"><b>{tier.key}</b> · {tier.title}</div>
            <div class="tier-meta num">{m.mastered}/{m.total}{m.mastered > 0 ? ' MASTERED' : ''}</div>
          </div>
          {#each tier.nodes as nd, i (nd.id)}
            {@const p = progress(nd.id)}
            {#if p}
              {@const active = data.tree.activeNodeId === nd.id}
              {@const sub = subLine(p)}
              {@const delta = deltaToday(nd.id)}
              {@const clickable = p.state !== 'locked' && wired.has(nd.id)}
              <a
                class="node"
                class:first={i === 0}
                class:last={i === tier.nodes.length - 1}
                class:active
                class:locked-row={p.state === 'locked'}
                href={clickable ? nodeHref(p) : undefined}
                aria-disabled={clickable ? undefined : true}
              >
                <span class="rail">
                  <span
                    class="dot"
                    class:mastered={p.state === 'mastered'}
                    class:rusty={p.state === 'rusty'}
                    class:armed={p.state === 'checkpoint-armed'}
                    class:locked={p.state === 'locked'}
                  >
                    {#if p.state === 'learning'}
                      <svg width="19" height="19" viewBox="0 0 19 19" aria-hidden="true">
                        <circle class="ring-track" cx="9.5" cy="9.5" r="7" />
                        <circle
                          class="ring-fill"
                          class:hero={active}
                          cx="9.5"
                          cy="9.5"
                          r="7"
                          stroke-dasharray={ringDash(p)}
                          transform="rotate(-90 9.5 9.5)"
                        />
                        <circle class="ring-core" class:hero={active} cx="9.5" cy="9.5" r="2.5" />
                      </svg>
                    {/if}
                  </span>
                </span>
                <span class="node-main">
                  <span class="node-name">{nd.name}</span>
                  {#if sub}
                    <span class="node-sub">
                      {#if sub.amber}<span class="amber">{sub.amber}</span>{/if}{sub.rest}
                    </span>
                  {/if}
                </span>
                <span class="node-right">
                  {#if p.state === 'locked'}
                    <span class="rating num dim">—</span>
                  {:else}
                    <span class="rating num">
                      {ratingOf(nd.id)}{#if delta !== 0}<span class="delta">{delta > 0 ? '↑' : '↓'}{Math.abs(delta)}</span>{/if}
                    </span>
                  {/if}
                  {#if active}<span class="chev">▸</span>{/if}
                </span>
              </a>
            {/if}
          {/each}
        </div>
      {/each}

      <!-- future tiers: ghost silhouettes (GDD §3 T4/T5) -->
      {#each SILHOUETTE_TIERS as sil (sil.key)}
        <div class="silhouette num">
          <span>{sil.key} · {sil.title}</span>
          <span class="boxes">
            {#each Array(sil.boxes) as _unused, bi (bi)}<span class="bx"></span>{/each}
          </span>
        </div>
      {/each}
    {/if}
  </div>

  <BottomNav />
</section>

<style>
  .page {
    flex: 1;
    width: 100%;
    max-width: 480px;
    margin: 0 auto;
    padding: 0 var(--s4);
    display: flex;
    flex-direction: column;
  }
  @media (max-width: 479px) {
    .page {
      padding-bottom: calc(var(--control-xl) + var(--s4));
    }
  }
  @media (min-width: 480px) {
    .page {
      padding-top: var(--s4);
    }
  }

  /* ---------- warm-up card ---------- */
  .warmup {
    flex: none;
    background: var(--elev-panel);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
    padding: var(--s4);
    margin-top: var(--s3);
  }
  .warmup-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
  }
  .warmup-title {
    font-size: var(--text-xs);
    font-weight: var(--weight-semibold);
    letter-spacing: var(--track-kicker);
    color: var(--ink-body);
  }
  .streak {
    font-size: var(--text-xs);
    color: var(--ink-muted);
    letter-spacing: var(--track-meta);
  }
  .streak b {
    font-weight: var(--weight-semibold);
    color: var(--ink-body);
  }
  .warmup-desc {
    margin-top: var(--s2);
    font-size: var(--text-body-sm);
    color: var(--ink-muted);
  }
  .warmup-desc b {
    font-weight: var(--weight-medium);
    color: var(--ink-body);
  }
  .start {
    margin-top: var(--s3);
    height: var(--hit-min);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--s2);
    background: var(--cta);
    color: var(--cta-ink);
    border-radius: var(--radius);
    font-size: var(--text-body);
    font-weight: var(--weight-semibold);
    letter-spacing: var(--track-kicker);
  }
  .start .arrow {
    font-size: var(--text-sm);
  }

  /* ---------- tree ---------- */
  .tree {
    flex: 1;
    min-height: 0;
    margin-top: var(--s2);
  }
  .tier {
    margin-top: var(--s3);
  }
  .tier-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    padding: 0 var(--s3) var(--s1);
  }
  .tier-name {
    font-size: var(--text-xs);
    font-weight: var(--weight-semibold);
    letter-spacing: var(--track-kicker);
    color: var(--ink-muted);
  }
  .tier-name b {
    color: var(--ink-body);
  }
  .tier-meta {
    font-size: var(--text-xs);
    color: var(--ink-muted);
    letter-spacing: var(--track-meta);
  }

  .node {
    display: grid;
    grid-template-columns: 28px 1fr auto;
    align-items: center;
    min-height: 38px;
    padding: 2px var(--s3);
    position: relative;
  }
  /* subway rail */
  .rail {
    position: relative;
    width: 28px;
    height: 100%;
    min-height: 34px;
    align-self: stretch;
  }
  .rail::before {
    content: '';
    position: absolute;
    left: 9px;
    top: -3px;
    bottom: -3px;
    width: 1px;
    background: var(--hairline);
  }
  .node.first .rail::before {
    top: 50%;
  }
  .node.last .rail::before {
    bottom: 50%;
  }
  .dot {
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    width: 19px;
    height: 19px;
    border-radius: var(--radius-full);
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--surface-0);
  }
  /* node states (design-system §3.9 skill-tree node) */
  .dot.mastered {
    background: var(--ink-body);
  }
  .dot.mastered::after {
    content: '✓';
    font-size: var(--text-xs);
    font-weight: var(--weight-bold);
    color: var(--surface-0);
  }
  .dot.rusty {
    border: 1.5px solid var(--highlight);
  }
  .dot.rusty::after {
    content: '⟳';
    font-size: var(--text-xs);
    color: var(--highlight);
  }
  .dot.armed {
    border: 1.5px solid var(--highlight);
  }
  .dot.armed::after {
    content: '';
    width: 7px;
    height: 7px;
    border-radius: var(--radius-full);
    background: var(--highlight);
  }
  .dot.locked {
    border: var(--hairline-w) solid var(--hairline);
  }
  .dot svg {
    display: block;
  }
  .ring-track {
    fill: none;
    stroke: var(--ink-ghost);
    stroke-width: 1.5;
  }
  .ring-fill {
    fill: none;
    stroke: var(--ink-muted);
    stroke-width: 1.5;
    stroke-linecap: round;
  }
  .ring-fill.hero {
    stroke: var(--ink-hero);
  }
  .ring-core {
    fill: var(--ink-muted);
  }
  .ring-core.hero {
    fill: var(--ink-hero);
  }

  .node-main {
    display: flex;
    flex-direction: column;
  }
  .node-name {
    font-size: var(--text-body);
    font-weight: var(--weight-medium);
    color: var(--ink-body);
    line-height: 1.25;
  }
  .node-sub {
    font-size: var(--text-xs);
    color: var(--ink-muted);
    line-height: 1.35;
    margin-top: 1px;
  }
  .node-sub .amber {
    color: var(--highlight);
  }
  .node.locked-row .node-name {
    color: var(--ink-muted);
    font-weight: var(--weight-regular);
  }

  /* rating chip (design-system §3.4) */
  .rating {
    font-size: var(--text-sm);
    font-weight: var(--weight-semibold);
    color: var(--ink-body);
    background: var(--elev-raised);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius-sm);
    padding: 2px var(--s2);
    letter-spacing: 0.02em;
  }
  .rating .delta {
    font-weight: var(--weight-regular);
    color: var(--ink-muted);
    margin-left: var(--s1);
  }
  .rating.dim {
    background: transparent;
    border-color: transparent;
    color: var(--ink-muted);
    font-weight: var(--weight-regular);
  }

  /* active (tap-next) row */
  .node.active {
    background: var(--elev-panel);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
    margin: 2px 0;
  }
  .node.active .rail::before {
    display: none;
  }
  .node.active .node-name {
    color: var(--ink-hero);
    font-weight: var(--weight-semibold);
  }
  .node-right {
    display: flex;
    align-items: center;
    justify-self: end;
  }
  .chev {
    color: var(--ink-body);
    font-size: var(--text-sm);
    margin-left: var(--s3);
  }

  /* T4/T5 silhouettes */
  .silhouette {
    display: flex;
    align-items: center;
    gap: var(--s2);
    padding: var(--s2) var(--s3) 0;
    font-size: var(--text-xs);
    letter-spacing: var(--track-kicker);
    color: var(--ink-muted);
  }
  .silhouette .boxes {
    display: flex;
    gap: var(--s1);
  }
  .silhouette .bx {
    width: 14px;
    height: 8px;
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius-xs);
  }
</style>
