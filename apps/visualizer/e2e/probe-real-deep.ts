// Real Deep run: app + deep-tier Ollama, layer step mode. Usage:
//   npx tsx e2e/probe-real-deep.ts <outDir> <ollama.exe> [model]
import { spawn } from 'node:child_process';
import { launch } from './helpers';

const out = process.argv[2] ?? '.';
const exe = process.argv[3];
const model = process.argv[4] ?? 'llama3.2';
const PORT = 4798;
const HTTP = 11438;

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
    },
    stdio: 'ignore',
  });
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  const state = () =>
    page.evaluate(() => {
      const s = (window as any).__viz.useUi.getState();
      const sd = s.vm.stepDetail;
      return {
        phase: s.vm.phase,
        detail: s.vm.detail,
        caps: s.vm.capabilities,
        step: sd?.step,
        nodes: sd?.nodes.length,
        lens: sd?.lens.length,
        heads: sd?.heads.length,
        lensTop: sd?.lens.map((l: any) => `${l.layer}:${l.top[0]?.text}`).join(' '),
        resp: s.vm.responseText.slice(0, 80),
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
    console.log('models', models.join(', '));
    await page.selectOption(
      '[data-testid=model]',
      models.find((m) => m.startsWith(model))!,
    );
    await page.check('[data-testid=step-mode]');
    await page.fill('[data-testid=prompt]', 'Why is the sky blue? One sentence.');
    await page.click('[data-testid=run]');
    await page.waitForFunction(
      () => (window as any).__viz.useUi.getState().vm.stepDetail?.nodes.length > 0,
      null,
      { timeout: 120_000 },
    );
    for (let i = 0; i < 6; i++) {
      await page.getByRole('button', { name: 'Step op' }).click();
      await page.waitForTimeout(300);
    }
    await page.screenshot({ path: `${out}/real-deep-ops.png` });
    console.log('OPS', JSON.stringify(await state()));
    for (let i = 0; i < 40; i++) {
      await page.getByRole('button', { name: 'Step layer' }).click();
      await page.waitForTimeout(350);
    }
    await page.waitForTimeout(1000);
    await page.screenshot({ path: `${out}/real-deep-layers.png` });
    console.log('LAYERS', JSON.stringify(await state()));
    await page.locator('.deep-lens-row').nth(20).click();
    await page.waitForTimeout(400);
    await page.screenshot({ path: `${out}/real-deep-pinned.png` });
    await page.getByRole('button', { name: 'Continue' }).click();
    await page.waitForSelector('[data-testid=pipeline][data-phase=done]', { timeout: 180_000 });
    await page.waitForTimeout(800);
    await page.screenshot({ path: `${out}/real-deep-done.png` });
    console.log('DONE', JSON.stringify(await state()));
    for (const k of [1, 2]) {
      await page.locator('[data-testid=response] .tok').nth(k).click();
      await page.waitForTimeout(600);
      await page.screenshot({ path: `${out}/real-deep-inspector-${k}.png` });
    }
    console.log('LENS ROWS', await page.locator('.inspector-lens-row').count());
    console.log('errors', errors);
  } finally {
    await app.close();
    ollama.kill();
  }
})();
