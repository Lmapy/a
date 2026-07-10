# The Auction — simulator app

Vite + Svelte 5 + TypeScript. See `/home/user/a/docs/simulator/game-design-document.md`
(behavior) and `/home/user/a/docs/simulator/design/design-system.md` + `src/styles/tokens.css`
(pixels — tokens are THE LAW; no raw hex outside tokens.css).

## Scripts

- `npm run dev` — dev server
- `npm run build` — production build (must pass before merging)
- `npm run preview` — serve the build (Playwright targets this)
- `npm run test` — Vitest, node env, engine unit tests (`src/**/*.test.ts`)
- `npm run test:watch` — Vitest watch
- `npm run e2e` — Playwright against the preview build (pre-installed chromium; never `playwright install`)
- `npm run check` — svelte-check + tsc

## Contracts

- `src/lib/types.ts` is the domain contract — extend, don't break.
- `src/lib/{core,gen,schedule}` are pure TS: zero Svelte/DOM imports; every public
  function has Vitest coverage; all randomness via `gen/prng.ts` named substreams.
- Every displayed quantity is computed by core/gen — nothing hand-placed in the UI.
