<script lang="ts">
  /* Top bar per design-system.md §3.6 (desktop top bar pattern): 56px,
     bottom hairline, wordmark (12px mono 600, tracked, orange ▮ tick) +
     text nav; active item = hero ink / 600. No icons, no color in chrome. */
  import { router, type RouteId } from '../router.svelte';

  const items: { id: RouteId; href: string; label: string }[] = [
    { id: 'home', href: '#/', label: 'Tree' },
    { id: 'drill', href: '#/drill', label: 'Drill' },
    { id: 'stats', href: '#/stats', label: 'Stats' },
    { id: 'boss', href: '#/boss', label: 'Boss' },
  ];
</script>

<header class="topbar">
  <a class="wordmark" href="#/"><span class="tick">▮</span>THE AUCTION</a>
  <nav aria-label="Primary">
    {#each items as item (item.id)}
      <a
        href={item.href}
        class="nav-item"
        class:active={router.route === item.id}
        aria-current={router.route === item.id ? 'page' : undefined}
      >
        {item.label}
      </a>
    {/each}
  </nav>
</header>

<style>
  .topbar {
    height: var(--control-xl);
    display: flex;
    align-items: center;
    gap: var(--s6);
    padding: 0 var(--s5);
    border-bottom: var(--hairline-w) solid var(--hairline);
    background: var(--surface-0);
  }
  .wordmark {
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    font-weight: var(--weight-semibold);
    letter-spacing: var(--track-overline);
    color: var(--ink-body);
    white-space: nowrap;
  }
  .tick {
    color: var(--poc-orange);
    margin-right: var(--s2);
  }
  nav {
    display: flex;
    gap: var(--s5);
  }
  .nav-item {
    font-family: var(--font-ui);
    font-size: var(--text-body-sm);
    color: var(--ink-muted);
    line-height: var(--control-xl);
  }
  .nav-item.active {
    color: var(--ink-hero);
    font-weight: var(--weight-semibold);
  }
</style>
