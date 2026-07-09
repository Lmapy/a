<script lang="ts">
  /* Boss intro placeholder (GDD §6, §8 screen 6; design-system §3.9 boss
     framing: amber overline, honest expectation line). Owned by the boss team. */
</script>

<section class="page">
  <div class="overline num">◆ BOSS SESSION · GATES T2 → T3</div>
  <h1 class="name">Trend-Day Fade Gauntlet</h1>
  <p class="expect">Expect <span class="num amber">~40%</span> success. Errors are the content.</p>
  <div class="rules">
    <div class="rule"><span>Desk limits</span><span class="num">day −$500 · trailing drawdown</span></div>
    <div class="rule"><span>Grading</span><span>Read Score vs planted truth — P&amp;L shown, never scored</span></div>
  </div>
  <button class="cta" type="button" disabled>ENTER — LOCKED IN SCAFFOLD</button>
</section>

<style>
  .page {
    flex: 1;
    width: 100%;
    max-width: 480px;
    margin: 0 auto;
    padding: var(--s5) var(--s4);
    display: flex;
    flex-direction: column;
    gap: var(--s4);
  }
  .overline {
    font-size: var(--text-xs);
    letter-spacing: var(--track-overline);
    color: var(--highlight);
  }
  .name {
    font-size: var(--text-display);
    font-weight: var(--weight-bold);
    letter-spacing: var(--track-display);
    line-height: var(--leading-tight);
    color: var(--ink-hero);
  }
  .expect {
    font-size: var(--text-body-lg);
    font-weight: var(--weight-medium);
    color: var(--ink-body);
  }
  .amber {
    color: var(--highlight);
    font-weight: var(--weight-semibold);
  }
  .rules {
    background: var(--elev-panel);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
  }
  .rule {
    display: flex;
    justify-content: space-between;
    gap: var(--s3);
    padding: var(--s3) var(--s4);
    font-size: var(--text-body-sm);
    color: var(--ink-body);
  }
  .rule + .rule {
    border-top: var(--hairline-w) solid var(--hairline);
  }
  .rule span:first-child {
    color: var(--ink-muted);
  }
  .cta {
    height: var(--control-lg);
    background: var(--cta);
    color: var(--cta-ink);
    border-radius: var(--radius);
    font-size: var(--text-sm);
    font-weight: var(--weight-semibold);
    letter-spacing: var(--track-kicker);
    text-transform: uppercase;
  }
  .cta:disabled {
    background: var(--surface-2);
    color: var(--ink-muted);
    cursor: not-allowed;
  }
</style>
