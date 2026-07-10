import { defineConfig } from '@playwright/test';

/**
 * E2E config. Browsers are pre-installed at /opt/pw-browsers — never run
 * `playwright install`. Targets the production build via `npm run preview`.
 */
export default defineConfig({
  testDir: './e2e',
  timeout: 30_000,
  retries: 0,
  use: {
    baseURL: 'http://localhost:4173',
    launchOptions: {
      executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
    },
    // deterministic screenshots / reduced flake
    viewport: { width: 390, height: 844 }, // mobile-portrait design canvas
  },
  webServer: {
    command: 'npm run preview -- --port 4173 --strictPort',
    url: 'http://localhost:4173',
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
  projects: [
    {
      name: 'chromium',
      use: {},
    },
  ],
});
