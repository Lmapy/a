<script lang="ts">
  /* Sparkline (design-system §3.9 stat row): blue line, ghost target rule,
     end-dot + end-label. Geometry ported from the dashboard mockup's spark().
     Pure presentation — data arrives pre-computed from the stats model. */
  interface Props {
    data: number[];
    lo: number;
    hi: number;
    /** Ghost horizontal reference (target bar); omit for none. */
    bar?: number | null;
    /** End label next to the last point (e.g. "+37", ".191"). */
    end?: string;
    width?: number;
    height?: number;
  }
  const { data, lo, hi, bar = null, end = '', width = 130, height = 44 }: Props = $props();

  const padR = 40;
  const padY = 7;
  const x = (i: number): number =>
    4 + (data.length > 1 ? (i * (width - padR - 8)) / (data.length - 1) : 0);
  const y = (v: number): number =>
    padY + (height - 2 * padY) * (1 - (v - lo) / (hi - lo || 1));

  const path = $derived(
    data.length > 0 ? 'M' + data.map((v, i) => `${x(i)},${y(v)}`).join(' L') : '',
  );
  const lastX = $derived(data.length > 0 ? x(data.length - 1) : 0);
  const lastY = $derived(data.length > 0 ? y(data[data.length - 1]) : 0);
</script>

<svg class="spark" viewBox={`0 0 ${width} ${height}`} style:width={`${width}px`} style:height={`${height}px`} aria-hidden="true">
  {#if bar !== null}
    <line class="ref" x1="0" y1={y(bar)} x2={width - padR + 6} y2={y(bar)} />
  {/if}
  {#if data.length > 0}
    <path class="line" d={path} />
    <circle class="dot" cx={lastX} cy={lastY} r="4" />
    <text class="end num" x={lastX + 9} y={lastY + 4}>{end}</text>
  {/if}
</svg>

<style>
  .spark {
    display: block;
    flex: none;
  }
  .ref {
    stroke: var(--ink-ghost);
    stroke-width: 1;
  }
  .line {
    fill: none;
    stroke: var(--vol-blue);
    stroke-width: 2;
    stroke-linejoin: round;
    stroke-linecap: round;
  }
  .dot {
    fill: var(--vol-blue);
    stroke: var(--surface-1);
    stroke-width: 2;
  }
  .end {
    font-size: var(--text-2xs);
    font-weight: var(--weight-semibold);
    fill: var(--ink-body);
  }
</style>
