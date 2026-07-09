import { expect, test } from '@playwright/test';

test.describe('app shell', () => {
  test('home renders the warm-up card and skill list on the dark surface', async ({ page }) => {
    await page.goto('/#/');
    await expect(page).toHaveTitle('The Auction');
    await expect(page.getByText('DAILY WARM-UP', { exact: false })).toBeVisible();
    await expect(page.getByText('Shapes')).toBeVisible();
    const bg = await page.evaluate(
      () => getComputedStyle(document.body).backgroundColor,
    );
    expect(bg).toBe('rgb(20, 20, 19)'); // --surface-0 #141413
  });

  test('hash routes render each placeholder screen', async ({ page }) => {
    await page.goto('/#/drill');
    await expect(page.getByText('Drill loop lands here')).toBeVisible();
    // engine-computed numbers visible (POC label from core)
    await expect(page.getByText('POC', { exact: true })).toBeVisible();

    await page.goto('/#/stats');
    await expect(page.getByText('SCORED JUDGMENTS', { exact: true })).toBeVisible();

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

  test('top nav navigates between routes', async ({ page }) => {
    await page.goto('/#/');
    await page.getByRole('link', { name: 'Drill' }).click();
    await expect(page).toHaveURL(/#\/drill$/);
    await expect(page.getByText('Drill loop lands here')).toBeVisible();
    await page.getByRole('link', { name: 'Stats' }).click();
    await expect(page.getByText('SCORED JUDGMENTS', { exact: true })).toBeVisible();
  });
});
