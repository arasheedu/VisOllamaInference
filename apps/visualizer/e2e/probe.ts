// Ad-hoc: launch the app, run the in-app mock, take screenshots at intervals.
// Usage: npx tsx e2e/probe.ts <outDir> [detail] [model]
import { launch } from './helpers';

const out = process.argv[2] ?? '.';
const detail = process.argv[3] ?? 'layers';
const model = process.argv[4] ?? 'mock-llama:8b';

(async () => {
  const { app, page } = await launch({ settings: { source: 'mock', detail } });
  const errors: string[] = [];
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.waitForTimeout(800);
  await page.selectOption('[data-testid=model]', model);
  await page.click('[data-testid=run]');
  for (const t of [700, 1500, 3000, 6000]) {
    await page.waitForTimeout(
      t === 700 ? 700 : t - [700, 1500, 3000, 6000][[700, 1500, 3000, 6000].indexOf(t) - 1],
    );
    await page.screenshot({ path: `${out}/shot-${detail}-${t}.png` });
  }
  await page
    .waitForSelector('[data-testid=pipeline][data-phase=done]', { timeout: 120_000 })
    .catch(() => {});
  await page.screenshot({ path: `${out}/shot-${detail}-done.png` });
  console.log('phase', await page.getAttribute('[data-testid=pipeline]', 'data-phase'));
  console.log('errors', errors.slice(0, 10));
  await app.close();
})();
