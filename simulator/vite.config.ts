/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte()],
  test: {
    // Engine libs (core/gen/schedule) are pure TS — node env is the contract.
    environment: 'node',
    include: ['src/**/*.test.ts'],
    // e2e specs belong to Playwright, never Vitest
    exclude: ['e2e/**', 'node_modules/**'],
  },
});
