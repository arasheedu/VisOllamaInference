// Perf probe: 80 layers, ~4k context, sample renderer fps while decoding.
// Usage: npx tsx e2e/probe-perf.ts <outDir> <detail>
import { launch } from './helpers';

const out = process.argv[2] ?? '.';
const detail = process.argv[3] ?? 'layers';
(async () => {
  const { app, page } = await launch({
    port: 4796,
    settings: { source: 'mock', detail, autosave: false, mockTokPerSec: 30 },
  });
  await page.waitForTimeout(500);
  await page.selectOption('[data-testid=model]', 'mock-llama:70b');
  const long =
    'The quick brown fox explains how attention works over a very long context window. '.repeat(
      260,
    );
  await page.fill('[data-testid=prompt]', long);
  await page.click('[data-testid=run]');
  await page.waitForSelector('[data-testid=pipeline][data-phase=decoding]', { timeout: 120_000 });
  const samples: number[] = [];
  for (let i = 0; i < 10; i++) {
    await page.waitForTimeout(1000);
    samples.push(await page.evaluate(() => (window as any).__viz.useUi.getState().fps));
  }
  const kv = await page.evaluate(() => (window as any).__viz.useUi.getState().vm.kv);
  const layers = await page.evaluate(() => (window as any).__viz.useUi.getState().vm.layersCount);
  await page.screenshot({ path: `${out}/perf-${detail}.png` });
  console.log(JSON.stringify({ detail, layers, kv, fps: samples }));
  await app.close();
})();
