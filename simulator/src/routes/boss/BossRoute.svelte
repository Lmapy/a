<script lang="ts">
  /* Boss route switcher (#/boss): intro ↔ run, driven by hash query params.
       #/boss                        → intro (mockup screen-boss-intro.html)
       #/boss?stage=run&seed=<dec>   → the gauntlet on that seed
     Dev/e2e params (run stage): &snap=decision1|verdict-fade|interrupt|debrief
     jumps deterministically to that moment (documented in BossRun.svelte). */
  import { onMount } from 'svelte';
  import BossIntro from './BossIntro.svelte';
  import BossRun from './BossRun.svelte';

  function queryParams(): URLSearchParams {
    const h = window.location.hash;
    const q = h.indexOf('?');
    return new URLSearchParams(q >= 0 ? h.slice(q + 1) : '');
  }

  let stage = $state<'intro' | 'run'>('intro');
  let seed = $state('1');
  let snap = $state<string | null>(null);
  /** Remount key: a new seed/snap must restart the run component. */
  let runKey = $state('');

  function readParams(): void {
    const p = queryParams();
    const s = p.get('seed');
    if (p.get('stage') === 'run' && s !== null) {
      seed = s;
      snap = p.get('snap');
      runKey = `${s}|${snap ?? ''}`;
      stage = 'run';
    } else {
      stage = 'intro';
    }
  }

  onMount(() => {
    readParams();
    window.addEventListener('hashchange', readParams);
    return () => window.removeEventListener('hashchange', readParams);
  });

  /** Fresh decimal-string run seed (53 random bits + time). */
  function freshSeed(): string {
    return String((BigInt(Date.now()) << 20n) ^ BigInt(Math.floor(Math.random() * 2 ** 40)));
  }

  function enter(): void {
    window.location.hash = `#/boss?stage=run&seed=${freshSeed()}`;
  }
</script>

{#if stage === 'run'}
  {#key runKey}
    <BossRun {seed} {snap} />
  {/key}
{:else}
  <BossIntro onenter={enter} />
{/if}
