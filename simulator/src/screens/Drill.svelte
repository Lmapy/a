<script lang="ts">
  /* Drill screen placeholder (GDD §8 screen 2). The real loop (canvas chart,
     answer chips, verdict layer, state machine ITEM→ARMED→ANSWERED→VERDICT→
     NEXT) is owned by the drill-loop team. This placeholder proves the route,
     the vertical contract, and that stimuli come from the REAL engine libs:
     the profile stats shown below are computed by gen + core, never typed in. */
  import { Prng } from '../lib/gen/prng';
  import { compileScript } from '../lib/gen/scripts';
  import { generateSession } from '../lib/gen/generator';
  import { buildProfile, rowToPrice } from '../lib/core/profile';

  const script = compileScript(new Prng('20260709'), 'normal', 'open-auction', 'in-value');
  const session = generateSession(script);
  const profile = buildProfile(session.bars, script.rowStep);
</script>

<section class="page">
  <div class="mode num">RATED · SCAFFOLD</div>
  <div class="chart-frame">
    <div class="chart-head">
      <span class="question">Drill loop lands here</span>
      <span class="meta num">SEED {script.seed} · {script.rowStep} PT ROWS</span>
    </div>
    <div class="chart-body">
      <p class="placeholder">
        Canvas chart layer (chart + verdict compositor) — engine-computed from
        seed {script.seed}:
      </p>
      <dl class="stats">
        <dt class="num">POC</dt>
        <dd class="num">{rowToPrice(profile, profile.poc).toFixed(2)}</dd>
        <dt class="num">VAH</dt>
        <dd class="num">{rowToPrice(profile, profile.vah).toFixed(2)}</dd>
        <dt class="num">VAL</dt>
        <dd class="num">{rowToPrice(profile, profile.val).toFixed(2)}</dd>
        <dt class="num">ROWS</dt>
        <dd class="num">{profile.rows.length}</dd>
      </dl>
    </div>
  </div>
</section>

<style>
  .page {
    flex: 1;
    width: 100%;
    max-width: 480px;
    margin: 0 auto;
    padding: var(--s4);
    display: flex;
    flex-direction: column;
    gap: var(--s3);
  }
  .mode {
    font-size: var(--text-xs);
    letter-spacing: var(--track-label);
    color: var(--ink-muted);
  }
  .chart-frame {
    flex: 1;
    background: var(--elev-panel);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
    display: flex;
    flex-direction: column;
  }
  .chart-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--s3);
    padding: var(--s3) var(--s4);
    border-bottom: var(--hairline-w) solid var(--hairline);
  }
  .question {
    font-size: var(--text-body-sm);
    font-weight: var(--weight-medium);
    color: var(--ink-body);
  }
  .meta {
    font-size: var(--text-xs);
    letter-spacing: var(--track-meta);
    color: var(--ink-muted);
  }
  .chart-body {
    flex: 1;
    padding: var(--s4);
    display: flex;
    flex-direction: column;
    gap: var(--s4);
  }
  .placeholder {
    font-size: var(--text-body);
    color: var(--ink-muted);
    max-width: 60ch;
  }
  .stats {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: var(--s2) var(--s4);
  }
  dt {
    font-size: var(--text-2xs);
    letter-spacing: var(--track-label);
    color: var(--ink-muted);
    align-self: center;
  }
  dd {
    font-size: var(--text-verdict);
    font-weight: var(--weight-semibold);
    color: var(--ink-hero);
  }
</style>
