// Artifact build: everything inlined into one HTML file (fonts handled by scripts/make-artifact.mjs)
import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { viteSingleFile } from 'vite-plugin-singlefile';

export default defineConfig({
  plugins: [svelte(), viteSingleFile()],
  build: { outDir: 'dist-artifact', assetsInlineLimit: 100_000_000 },
});
