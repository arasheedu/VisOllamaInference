// Real Deep run WITHOUT step mode: does the UI keep up and reach Done? Usage:
//   npx tsx e2e/probe-real-deep-live.ts <outDir> <ollama.exe> [model]
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { launch } from './helpers';

const out = process.argv[2] ?? '.';
const exe = process.argv[3];
const model = process.argv[4] ?? 'llama3.2';
const PORT = 4799;
const HTTP = 11439;

(async () => {
  const { app, page } = await launch({
    port: PORT,
    settings: {
      source: 'ollama',
      ollamaUrl: `http://127.0.0.1:${HTTP}`,
      detail: 'deep',
      stepGranularity: 'layer',
      autosave: false,
    },
  });
  const ollama = spawn(exe, ['serve'], {
    env: {
      ...process.env,
      OLLAMA_HOST: `127.0.0.1:${HTTP}`,
      OLLAMA_TRACE_URL: `ws://127.0.0.1:${PORT}/ingest`,
      ...(process.env.CPU_ONLY ? { CUDA_VISIBLE_DEVICES: '-1' } : {}),
    },
    stdio: 'ignore',
  });
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  const state = () =>
    page.evaluate(() => {
      const s = (window as any).__viz.useUi.getState();
      const p = (window as any).__viz.session.active;
      return {
        phase: s.vm.phase,
        running: s.running,
        tokens: s.vm.responseTokens.length,
        buffered: p?.events.length,
        pos: p?.position,
        clock: {
          paused: s.vm.clock.paused,
          live: s.vm.clock.live,
          lag: Math.round(s.vm.clock.lagMs),
          speed: +s.vm.clock.speed.toFixed(2),
          degraded: s.vm.clock.degraded,
          stepping: s.vm.clock.stepping,
        },
        hub: s.hub && { in: s.hub.eventsIn },
        fps: s.fps,
        warnings: s.warnings.map((w: any) => w.text),
      };
    });
  try {
    let models: string[] = [];
    for (let i = 0; i < 30 && !models.some((m) => m.startsWith(model)); i++) {
      await page.waitForTimeout(1000);
      models = await page.evaluate(async () => {
        await (window as any).__viz.session.refreshModels();
        return (window as any).__viz.useUi.getState().models;
      });
    }
    await page.selectOption(
      '[data-testid=model]',
      models.find((m) => m.startsWith(model))!,
    );
    if (process.env.STEP) await page.check('[data-testid=step-mode]');
    await page.fill(
      '[data-testid=prompt]',
      process.env.DEEP_PROMPT ?? 'Why is the sky blue? Answer in three sentences.',
    );
    const cdp = process.env.PROFILE ? await page.context().newCDPSession(page) : null;
    if (cdp) {
      await cdp.send('Profiler.enable');
      await cdp.send('Profiler.setSamplingInterval', { interval: 500 });
      await cdp.send('Profiler.start');
    }
    await page.evaluate(() => {
      const w = window as any;
      w.__long = [];
      new PerformanceObserver((l) => {
        for (const e of l.getEntries())
          w.__long.push([Math.round(e.startTime), Math.round(e.duration)]);
      }).observe({ type: 'long-animation-frame', buffered: false });
    });
    await page.click('[data-testid=run]');
    const t0 = Date.now();
    if (cdp)
      setTimeout(async () => {
        const { profile } = await cdp.send('Profiler.stop');
        writeFileSync(out + '/deep.cpuprofile', JSON.stringify(profile));
        console.log('profile saved');
      }, 16000);
    if (process.env.STEP) {
      await page.waitForFunction(
        () => (window as any).__viz.useUi.getState().vm.phase === 'decoding',
        null,
        { timeout: 60_000 },
      );
      await page.waitForTimeout(1500);
      console.log(
        'at decode',
        JSON.stringify(await state()),
        'banner visible:',
        await page.locator('[data-testid=step-waiting]').isVisible(),
      );
      await page.screenshot({ path: `${out}/step-waiting.png` });
      for (let k = 0; k < 3; k++) {
        const before = (await state()).tokens;
        const ts = Date.now();
        await page.getByRole('button', { name: 'Step token' }).click();
        try {
          await page.waitForFunction(
            (b) => (window as any).__viz.useUi.getState().vm.responseTokens.length > b,
            before,
            { timeout: 30_000 },
          );
        } catch {
          console.log('STEP TOKEN TIMEOUT', JSON.stringify(await state()));
        }
        console.log('step token', k, 'took', Date.now() - ts, 'ms');
      }
      await page.getByRole('button', { name: 'Continue' }).click();
    }
    for (let i = 0; i < 60; i++) {
      await page.waitForTimeout(3000);
      const st = await state();
      console.log(`${Math.round((Date.now() - t0) / 1000)}s`, JSON.stringify(st));
      if (st.phase === 'done' || st.phase === 'error') break;
    }
    const long = await page.evaluate(() => (window as any).__long as number[][]);
    long.sort((a, b) => b[1] - a[1]);
    console.log(
      'LONG FRAMES',
      long.length,
      'total',
      long.reduce((a, x) => a + x[1], 0),
      'ms; worst',
      JSON.stringify(long.slice(0, 8)),
    );
    await page.screenshot({ path: `${out}/deep-live.png` });
    console.log('errors', errors.slice(0, 5));
  } finally {
    await app.close();
    ollama.kill();
  }
})();
