<script lang="ts">
  /* Mobile top chrome (skill-tree mockup): 44px, wordmark with orange ▮ tick,
     ⚙ meta glyph right. Shown ≤479px only — the desktop top bar (TopNav)
     takes over above. */
</script>

<div class="topbar">
  <a class="wordmark num" href="#/"><span class="tick">▮</span> THE AUCTION</a>
  <span class="gear" aria-hidden="true">⚙</span>
</div>

<style>
  .topbar {
    height: 44px;
    flex: none;
    display: none;
    align-items: center;
    justify-content: space-between;
    padding-top: var(--s2);
  }
  @media (max-width: 479px) {
    .topbar {
      display: flex;
    }
  }
  .wordmark {
    font-size: var(--text-xs);
    font-weight: var(--weight-semibold);
    letter-spacing: var(--track-overline);
    color: var(--ink-body);
  }
  .tick {
    color: var(--poc-orange);
  }
  .gear {
    color: var(--ink-muted);
    font-size: var(--text-body-lg);
  }
</style>
