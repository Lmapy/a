<script lang="ts">
  /* Stats placeholder (GDD §8 screen 8): sentence-led calibration, confusion
     matrices, discipline counters, expectancy ledger. Owned by the stats team. */
</script>

<section class="page">
  <p class="lede">Stats land here — sentence first, numbers second.</p>
  <div class="card">
    <div class="key num">SCORED JUDGMENTS</div>
    <div class="value num">0</div>
    <p class="caption">Most players normalize within 100–200 scored judgments.</p>
  </div>
  <div class="card unmeasured">
    <span class="watermark num">unmeasured = folklore</span>
  </div>
</section>

<style>
  .page {
    flex: 1;
    width: 100%;
    max-width: 960px;
    margin: 0 auto;
    padding: var(--s5);
    display: flex;
    flex-direction: column;
    gap: var(--s4);
  }
  .lede {
    font-size: var(--text-body-lg);
    font-weight: var(--weight-medium);
    color: var(--ink-body);
    max-width: 60ch;
  }
  .card {
    background: var(--elev-panel);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
    padding: var(--s4) var(--s5);
    display: flex;
    flex-direction: column;
    gap: var(--s2);
  }
  .key {
    font-size: var(--text-xs);
    letter-spacing: var(--track-label);
    color: var(--ink-muted);
    text-transform: uppercase;
  }
  .value {
    font-size: var(--text-stat);
    font-weight: var(--weight-semibold);
    color: var(--ink-hero);
    line-height: var(--leading-tight);
  }
  .caption {
    font-size: var(--text-sm);
    color: var(--ink-muted);
  }
  .unmeasured {
    align-items: center;
    justify-content: center;
    min-height: 120px;
  }
  .watermark {
    font-size: var(--text-2xs);
    letter-spacing: var(--track-caption);
    color: var(--ink-ghost);
  }
</style>
