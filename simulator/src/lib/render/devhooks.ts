/* Dev/e2e instrumentation surface: window.__auction. The e2e suite asserts
   the render + verdict-paint budgets against these measured numbers. */

export interface AuctionDevHooks {
  /** Last full chart redraw, ms (budget < 8ms). */
  lastRenderMs?: number;
  /** pointerdown → verdict painted (next frame), ms (budget ≤ 100ms). */
  verdictPaintMs?: number;
  /** Last item pre-generation, ms. */
  genMs?: number;
}

/** The (lazily created) global hook bag. Safe in Node (plain object). */
export function devHooks(): AuctionDevHooks {
  const g = globalThis as unknown as { __auction?: AuctionDevHooks };
  g.__auction ??= {};
  return g.__auction;
}
