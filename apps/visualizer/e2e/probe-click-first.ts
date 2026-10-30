// Live Deep run: click the first response token as soon as it appears. Usage:
//   npx tsx e2e/probe-click-first.ts <outDir> <ollama.exe>
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
  const state = () =>
    page.evaluate(() => {
      const s = (window as any).__viz.useUi.getState();
      return {
        phase: s.vm.phase,
        tokens: s.vm.responseTokens.length,
        paused: s.vm.clock.paused,
        running: s.running,
        inspector: s.inspector?.kind ?? null,
        badge: !!document.querySelector('[data-testid=live-paused]'),
      };
    });
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
    await page.fill('[data-testid=prompt]', 'Explain in about 300 words why the sky is blue.');
    await page.click('[data-testid=run]');
    await page.locator('[data-testid=response] .tok').first().waitFor({ timeout: 60_000 });
    await page.locator('[data-testid=response] .tok').first().click();
    for (let i = 0; i < 6; i++) {
      await page.waitForTimeout(2000);
      console.log(`${(i + 1) * 2}s after click`, JSON.stringify(await state()));
    }
    await page.screenshot({ path: `${out}/click-first.png` });
  } finally {
    await app.close();
    ollama.kill();
  }
})();
