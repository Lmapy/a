import { expect, test } from '@playwright/test';

test.describe('app shell', () => {
  test('home renders the warm-up card and skill tree on the dark surface', async ({ page }) => {
    await page.goto('/#/');
    await expect(page).toHaveTitle('The Auction');
    await expect(page.getByText('DAILY WARM-UP', { exact: false })).toBeVisible();
    await expect(page.getByText('Shape Alphabet')).toBeVisible();
    await expect(page.getByText('READ THE MAP')).toBeVisible();
    const bg = await page.evaluate(
      () => getComputedStyle(document.body).backgroundColor,
    );
    expect(bg).toBe('rgb(20, 20, 19)'); // --surface-0 #141413
  });

  test('hash routes render each screen', async ({ page }) => {
    await page.goto('/#/drill');
    // the real drill loop (rep counter + question bound to the stimulus)
    await expect(page.getByText('REP 1/10')).toBeVisible();
    await expect(page.getByText('What shape is this session?')).toBeVisible();

    await page.goto('/#/stats');
    await expect(page.getByText('scored judgments')).toBeVisible();

    await page.goto('/#/boss');
    await expect(page.getByText('Trend-Day Fade Gauntlet')).toBeVisible();
  });

  test('no horizontal overflow at the mobile design width', async ({ page }) => {
    for (const route of ['', 'drill', 'stats', 'boss']) {
      await page.goto(`/#/${route}`);
      const scrollWidth = await page.evaluate(
        () => document.documentElement.scrollWidth,
      );
      expect(scrollWidth, `route #/${route}`).toBeLessThanOrEqual(390);
    }
  });

  test('nav travels between routes (mobile bottom nav on home/stats)', async ({ page }) => {
    // On mobile widths Home/Stats swap the desktop top bar for the bottom nav
    await page.goto('/#/');
    await page.getByRole('link', { name: 'DRILL', exact: true }).click();
    await expect(page).toHaveURL(/#\/drill$/);
    await expect(page.getByText('REP 1/10')).toBeVisible();
    // the drill screen keeps the top bar at every width
    await page.getByRole('link', { name: 'Stats' }).click();
    await expect(page.getByText('scored judgments')).toBeVisible();
    await page.getByRole('link', { name: 'TREE', exact: true }).click();
    await expect(page.getByText('DAILY WARM-UP')).toBeVisible();
    // the boss gate exam is reachable from the mobile home (audit fix 2)
    await page.getByRole('link', { name: 'BOSS', exact: true }).click();
    await expect(page).toHaveURL(/#\/boss$/);
    await expect(page.getByText('Trend-Day Fade Gauntlet')).toBeVisible();
  });

  test('desktop keeps the top bar on home/stats (no bottom nav)', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto('/#/');
    await expect(page.getByRole('link', { name: 'Stats' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'STATS', exact: true })).toBeHidden();
    await page.getByRole('link', { name: 'Stats' }).click();
    await expect(page.getByText('scored judgments')).toBeVisible();
  });
});
