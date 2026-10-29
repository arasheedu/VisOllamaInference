// Real end-to-end: app + patched Ollama. Usage:
//   npx tsx e2e/probe-real.ts <outDir> <ollama.exe> [model]
import { spawn } from 'node:child_process';
import { launch } from './helpers';

const out = process.argv[2] ?? '.';
const exe = process.argv[3];
const model = process.argv[4] ?? 'llama3.2';
const PORT = 4793;
const HTTP = 11436;

(async () => {
  const { app, page } = await launch({
    port: PORT,
    settings: {
      source: 'ollama',
      ollamaUrl: `http://127.0.0.1:${HTTP}`,
      detail: 'full',
      autosave: false,
    },
  });
  const ollama = spawn(exe, ['serve'], {
    env: {
      ...process.env,
      OLLAMA_HOST: `127.0.0.1:${HTTP}`,
      OLLAMA_TRACE_URL: `ws://127.0.0.1:${PORT}/ingest`,
    },
    stdio: 'ignore',
  });
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  const state = () =>
    page.evaluate(() => {
      const s = (window as any).__viz.useUi.getState();
      return {
        phase: s.vm.phase,
        caps: s.vm.capabilities,
        prompt: s.vm.promptTokens.length,
        resp: s.vm.responseTokens.length,
        text: s.vm.responseText.slice(0, 120),
        warnings: s.warnings.map((w: any) => `${w.id}: ${w.text}`),
        merge: s.merge ? { ok: s.merge.ok, rounds: s.merge.rounds, reason: s.merge.reason } : null,
        hub: s.hub && { producers: s.hub.producers, eventsIn: s.hub.eventsIn },
        stats: s.vm.stats,
      };
    });
  try {
    await page.waitForTimeout(3000);
    await page.evaluate(() => (window as any).__viz.session.refreshModels());
    await page.waitForTimeout(1000);
    await page.selectOption('[data-testid=model]', model);

    // 1) normal run, Full detail
    await page.fill('[data-testid=prompt]', 'In two sentences, why is the sky blue?');
    await page.click('[data-testid=run]');
    await page.waitForSelector('[data-testid=pipeline][data-phase=tokenizing]', {
      timeout: 120_000,
    });
    await page.waitForTimeout(400);
    await page.screenshot({ path: `${out}/real-merge.png` });
    await page.waitForSelector('[data-testid=pipeline][data-phase=done]', { timeout: 180_000 });
    await page.waitForTimeout(800);
    await page.screenshot({ path: `${out}/real-done.png` });
    console.log('RUN1', JSON.stringify(await state()));

    // 2) step mode (token granularity), Layers detail
    await page.click('[data-testid=detail-layers]');
    await page.check('[data-testid=step-mode]');
    await page.fill('[data-testid=prompt]', 'Count from one to five.');
    await page.click('[data-testid=run]');
    await page.waitForSelector('[data-testid=pipeline][data-phase=prefill]', { timeout: 60_000 });
    await page.waitForTimeout(1500);
    const before = (await state()).resp;
    for (let i = 0; i < 3; i++) {
      await page.getByRole('button', { name: 'Step token' }).click();
      await page.waitForTimeout(1200);
    }
    const after = (await state()).resp;
    await page.screenshot({ path: `${out}/real-step.png` });
    console.log('STEP tokens before', before, 'after 3 steps', after);
    await page.getByRole('button', { name: 'Continue' }).click();
    await page.waitForSelector('[data-testid=pipeline][data-phase=done]', { timeout: 120_000 });
    console.log('RUN2', JSON.stringify(await state()));
    console.log('errors', errors);
  } finally {
    await app.close();
    ollama.kill();
  }
})();
