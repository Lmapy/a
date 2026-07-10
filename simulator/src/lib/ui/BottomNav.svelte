<script lang="ts">
  /* Mobile bottom nav (design-system §3.6): 56px, top hairline, full-bleed,
     4 items TREE / DRILL / BOSS / STATS, 11px mono caps tracked --track-nav,
     active = hero ink + 32×2px body-ink tick overlapping the hairline. No
     icons. Rendered by Home/Stats at ≤479px only (the desktop top bar takes
     over above). BOSS is here because the gate exam is a first-class GDD
     destination and this nav is the only chrome on the mobile home screen.
     The ?demo=1 review flag is preserved across taps. */
  import { router } from '../router.svelte';
  import { demoFlag } from '../stores/appdata';

  const items = [
    { id: 'home', base: '#/', label: 'TREE' },
    { id: 'drill', base: '#/drill', label: 'DRILL' },
    { id: 'boss', base: '#/boss', label: 'BOSS' },
    { id: 'stats', base: '#/stats', label: 'STATS' },
  ] as const;

  // router.route in the deps → hrefs recompute after every hash change.
  const hrefs = $derived.by(() => {
    void router.route;
    const demo = demoFlag();
    return items.map((it) => (demo ? `${it.base}?demo=1` : it.base));
  });
</script>

<nav class="bottomnav" aria-label="Primary">
  {#each items as item, i (item.id)}
    <a
      href={hrefs[i]}
      class="item num"
      class:on={router.route === item.id}
      aria-current={router.route === item.id ? 'page' : undefined}
    >
      {item.label}
    </a>
  {/each}
</nav>

<style>
  .bottomnav {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    height: var(--control-xl);
    display: flex;
    align-items: stretch;
    border-top: var(--hairline-w) solid var(--hairline);
    background: var(--surface-0);
    z-index: 10;
  }
  @media (min-width: 480px) {
    .bottomnav {
      display: none;
    }
  }
  .item {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    letter-spacing: var(--track-nav);
    color: var(--ink-muted);
  }
  .item.on {
    color: var(--ink-hero);
    font-weight: var(--weight-semibold);
  }
  .item.on::before {
    content: '';
    position: absolute;
    top: -1px;
    left: 50%;
    transform: translateX(-50%);
    width: 32px;
    height: 2px;
    background: var(--ink-body);
  }
</style>
