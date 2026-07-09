<script lang="ts">
  import TopNav from './lib/ui/TopNav.svelte';
  import Home from './screens/Home.svelte';
  import Drill from './screens/Drill.svelte';
  import Stats from './screens/Stats.svelte';
  import Boss from './screens/Boss.svelte';
  import { router } from './lib/router.svelte';
</script>

<TopNav />

{#if router.route === 'drill'}
  <Drill />
{:else if router.route === 'stats'}
  <Stats />
{:else if router.route === 'boss'}
  <Boss />
{:else}
  <Home />
{/if}
