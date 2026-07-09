<script lang="ts">
  /* Home = skill tree root (GDD §8 screen 1/4). Placeholder: warm-up card +
     skill list from the player store. Tree rendering owned by the UI team. */
  import { player, V1_NODE_IDS } from '../lib/stores/player.svelte';

  const names: Record<string, string> = {
    'poc-va-snap': 'POC / VA',
    'excess-or-poor': 'Excess / Poor',
    'hvn-lvn-marker': 'HVN / LVN',
    'shape-alphabet': 'Shapes',
    'open-type-ladder': 'Open Type',
    'one-timeframing-buzzer': 'One-Timeframing',
    'acceptance-clock': 'Acceptance',
    'regime-gate': 'Regime Gate',
    'calibration-range': 'Calibration',
    'setup-picker': 'Setup Picker',
  };
</script>

<section class="page">
  <div class="card">
    <div class="kicker num">DAILY WARM-UP · 3 MIN</div>
    <p class="desc">Scaffold placeholder — warm-up assembly lands with the schedule lib.</p>
    <a class="cta" href="#/drill">START</a>
  </div>

  <h2 class="section-label num">SKILLS</h2>
  <ul class="skills">
    {#each V1_NODE_IDS as id (id)}
      <li class="skill">
        <span class="name">{names[id] ?? id}</span>
        <span class="state num">{player.nodeStates[id]}</span>
        <span class="rating num" class:dim={player.nodeStates[id] === 'locked'}>
          {player.nodeStates[id] === 'locked' ? '—' : Math.round(player.rating(id).rating)}
        </span>
      </li>
    {/each}
  </ul>
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
    gap: var(--s4);
  }
  .card {
    background: var(--elev-panel);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius);
    padding: var(--s4);
    display: flex;
    flex-direction: column;
    gap: var(--s3);
  }
  .kicker {
    font-size: var(--text-xs);
    letter-spacing: var(--track-kicker);
    color: var(--ink-muted);
    text-transform: uppercase;
  }
  .desc {
    font-size: var(--text-body-sm);
    color: var(--ink-body);
  }
  .cta {
    display: flex;
    align-items: center;
    justify-content: center;
    height: var(--hit-min);
    background: var(--cta);
    color: var(--cta-ink);
    border-radius: var(--radius);
    font-size: var(--text-sm);
    font-weight: var(--weight-semibold);
    letter-spacing: var(--track-kicker);
    text-transform: uppercase;
  }
  .section-label {
    font-size: var(--text-xs);
    letter-spacing: var(--track-label);
    color: var(--ink-muted);
    text-transform: uppercase;
    font-weight: var(--weight-regular);
  }
  .skills {
    list-style: none;
    display: flex;
    flex-direction: column;
  }
  .skill {
    display: flex;
    align-items: center;
    gap: var(--s3);
    min-height: var(--hit-min);
    border-bottom: var(--hairline-w) solid var(--surface-2);
  }
  .name {
    flex: 1;
    font-size: var(--text-body-sm);
    font-weight: var(--weight-medium);
    color: var(--ink-body);
  }
  .state {
    font-size: var(--text-2xs);
    letter-spacing: var(--track-caption);
    text-transform: uppercase;
    color: var(--ink-ghost);
  }
  .rating {
    font-size: var(--text-sm);
    font-weight: var(--weight-semibold);
    color: var(--ink-body);
    background: var(--surface-2);
    border: var(--hairline-w) solid var(--hairline);
    border-radius: var(--radius-sm);
    padding: 2px var(--s2);
  }
  .rating.dim {
    border-color: transparent;
    color: var(--ink-muted);
    background: none;
  }
</style>
