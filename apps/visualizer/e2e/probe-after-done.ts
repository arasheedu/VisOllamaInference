// Live Deep run to Done, then click words with and without layer data. Usage:
//   npx tsx e2e/probe-after-done.ts <outDir> <ollama.exe>
import { spawn } from 'node:child_process';
import { launch } from './helpers';

const out = process.argv[2] ?? '.';
const exe = process.argv[3];
const PORT = 4799;
const HTTP = 11439;

(async () => {
  const { app, page } = await launch({
    port: PORT,
    settings: {
      source: 'ollama',
      ollamaUrl: `http://127.0.0.1:${HTTP}`,
      detail: 'deep',
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
  const lensText = () =>
    page.evaluate(
      () =>
        [...document.querySelectorAll('.deep-lens-row .deep-lens-tok')]
          .map((e) => e.textContent)
          .filter(Boolean).length,
    );
  try {
    let models: string[] = [];
    for (let i = 0; i < 30 && !models.some((m) => m.startsWith('llama3.2')); i++) {
      await page.waitForTimeout(1000);
      models = await page.evaluate(async () => {
        await (window as any).__viz.session.refreshModels();
        return (window as any).__viz.useUi.getState().models;
      });
    }
    await page.selectOption(
      '[data-testid=model]',
      models.find((m) => m.startsWith('llama3.2'))!,
    );
    await page.uncheck('[data-testid=step-mode]');
    await page.fill('[data-testid=prompt]', 'Explain in about 120 words why the sky is blue.');
    await page.click('[data-testid=run]');
    await page.waitForSelector('[data-testid=pipeline][data-phase=done]', { timeout: 120_000 });
    const marked = await page.locator('[data-testid=response] .tok.has-deep').count();
    const all = await page.locator('[data-testid=response] .tok').count();
    console.log('words', all, 'with layer data', marked);
    // a word without data
    const idx = await page.evaluate(() => {
      const els = [...document.querySelectorAll('[data-testid=response] .tok')];
      return els.findIndex((e, i) => i > 8 && !e.classList.contains('has-deep'));
    });
    await page.locator('[data-testid=response] .tok').nth(idx).click();
    await page.waitForTimeout(600);
    console.log(
      'untraced word',
      idx,
      'message:',
      await page.locator('[data-testid=no-deep-data]').isVisible(),
      'lens note:',
      await page.locator('[data-testid=lens-none]').isVisible(),
      'lens rows with text:',
      await lensText(),
    );
    await page.screenshot({ path: `${out}/after-done-untraced.png` });
    await page.locator('[data-testid=no-deep-data] .deep-nearest button').first().click();
    await page.waitForTimeout(600);
    console.log(
      'after "nearest":',
      await page.locator('[data-testid=deep-selected]').textContent(),
      'lens rows with text:',
      await lensText(),
    );
    await page.screenshot({ path: `${out}/after-done-nearest.png` });
  } finally {
    await app.close();
    ollama.kill();
  }
})();
