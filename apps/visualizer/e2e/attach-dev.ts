// Attach to a running dev session started with VISUALIZER_CDP_PORT and run one
// prompt, logging playback progress, frame rate and long frames. Usage:
//   VISUALIZER_CDP_PORT=9333 pnpm dev            (in another shell)
//   npx tsx e2e/attach-dev.ts <outDir> [port] [model]
import { writeFileSync } from 'node:fs';
import { chromium } from '@playwright/test';

const out = process.argv[2] ?? '.';
const port = process.argv[3] ?? '9333';
const model = process.argv[4] ?? 'llama3.2';
const prompt =
  process.env.DEEP_PROMPT ?? 'Explain in detail, in about 300 words, why the sky is blue.';

(async () => {
  const browser = await chromium.connectOverCDP(`http://127.0.0.1:${port}`);
  const page = browser
    .contexts()[0]
    .pages()
    .find((p) => !p.url().startsWith('devtools'))!;
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  const state = () =>
    page.evaluate(() => {
      const s = (window as any).__viz.useUi.getState();
      const p = (window as any).__viz.session.active;
      return {
        phase: s.vm.phase,
        tokens: s.vm.responseTokens.length,
        buffered: p?.events.length,
        pos: p?.position,
        lag: Math.round(s.vm.clock.lagMs),
        degraded: s.vm.clock.degraded,
        paused: s.vm.clock.paused,
        step: s.stepMode,
        fps: s.fps,
        detail: s.settings.detail,
        tab: s.deepTab,
        inspector: s.inspector?.kind ?? null,
      };
    });
  console.log(
    'url',
    page.url(),
    'viewport',
    JSON.stringify(await page.evaluate(() => [innerWidth, innerHeight, devicePixelRatio])),
  );
  console.log('before', JSON.stringify(await state()));
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Profiler.enable');
  await cdp.send('Profiler.setSamplingInterval', { interval: 500 });
  await cdp.send('Profiler.start');
  await page.evaluate(() => {
    const w = window as any;
    w.__long = [];
    new PerformanceObserver((l) => {
      for (const e of l.getEntries())
        w.__long.push([Math.round(e.startTime), Math.round(e.duration)]);
    }).observe({ type: 'long-animation-frame', buffered: false });
  });
  const models: string[] = await page.evaluate(() => (window as any).__viz.useUi.getState().models);
  const pick = models.find((m) => m.startsWith(model));
  if (pick) await page.selectOption('[data-testid=model]', pick);
  if (process.env.NO_STEP) await page.uncheck('[data-testid=step-mode]');
  await page.fill('[data-testid=prompt]', prompt);
  await page.click('[data-testid=run]');
  const t0 = Date.now();
  for (let i = 0; i < 25; i++) {
    await page.waitForTimeout(2000);
    const st = await state();
    console.log(`${Math.round((Date.now() - t0) / 1000)}s`, JSON.stringify(st));
    if (st.phase === 'done' || st.phase === 'error') break;
  }
  const { profile } = await cdp.send('Profiler.stop');
  writeFileSync(`${out}/dev.cpuprofile`, JSON.stringify(profile));
  const long: number[][] = await page.evaluate(() => (window as any).__long);
  long.sort((a, b) => b[1] - a[1]);
  console.log(
    'LONG FRAMES',
    long.length,
    'total',
    long.reduce((a, x) => a + x[1], 0),
    'worst',
    JSON.stringify(long.slice(0, 10)),
  );
  await page.screenshot({ path: `${out}/dev-attach.png` });
  console.log('errors', errors.slice(0, 8));
  await browser.close().catch(() => {});
})();
