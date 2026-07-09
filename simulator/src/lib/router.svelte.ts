/* ============================================================================
   Hash router (runes-in-.svelte.ts — framework-agnostic state, no SvelteKit).

   Routes:
     #/        → home (skill tree)      · RouteId 'home'
     #/drill   → drill loop             · RouteId 'drill'
     #/stats   → stats dashboard        · RouteId 'stats'
     #/boss    → boss session           · RouteId 'boss'

   Unknown hashes fall back to 'home'. Navigate by setting location.hash
   (links: href="#/drill") or via navigate().
   ========================================================================== */

export type RouteId = 'home' | 'drill' | 'stats' | 'boss';

/** Parse a location.hash value into a RouteId. Exported for unit tests. */
export function parseHash(hash: string): RouteId {
  const path = hash.replace(/^#\/?/, '').split('?')[0].replace(/\/+$/, '');
  switch (path) {
    case 'drill':
      return 'drill';
    case 'stats':
      return 'stats';
    case 'boss':
      return 'boss';
    default:
      return 'home';
  }
}

function currentHash(): string {
  return typeof window === 'undefined' ? '' : window.location.hash;
}

class Router {
  /** The active route — reactive; read it in components as router.route. */
  route: RouteId = $state(parseHash(currentHash()));

  constructor() {
    if (typeof window !== 'undefined') {
      window.addEventListener('hashchange', () => {
        this.route = parseHash(window.location.hash);
      });
    }
  }

  /** Programmatic navigation (screen-level transitions only, --dur-4). */
  navigate(to: RouteId): void {
    if (typeof window === 'undefined') return;
    window.location.hash = to === 'home' ? '#/' : `#/${to}`;
  }
}

/** App-wide router singleton. */
export const router = new Router();
