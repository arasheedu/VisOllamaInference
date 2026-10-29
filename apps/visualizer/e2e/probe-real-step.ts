import { spawn } from 'node:child_process';
import { launch } from './helpers';
const exe = process.argv[2];
const PORT = 4793,
  HTTP = 11436;
(async () => {
  const { app, page } = await launch({
    port: PORT,
    settings: {
      source: 'ollama',
      ollamaUrl: `http://127.0.0.1:${HTTP}`,
      detail: 'layers',
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
  const dbg = () =>
    page.evaluate(() => {
      const v = (window as any).__viz;
      const s = v.session;
      const p = s.active;
      const r = s.run;
      return {
        phase: v.useUi.getState().vm.phase,
        toks: v.useUi.getState().vm.responseTokens.length,
        acks: r?.acksSent,
        since: r?.eventsSinceAck,
        events: p?.events.length,
        pos: p?.position,
        atEnd: p?.atEnd,
        stop: !!p?.clock.stopWhen,
        paused: p?.clock.paused,
        last: p?.events.at(-1)?.type,
        traceDone: r?.traceDone,
      };
    });
  try {
    await page.waitForTimeout(3000);
    await page.evaluate(() => (window as any).__viz.session.refreshModels());
    await page.waitForTimeout(800);
    await page.selectOption('[data-testid=model]', 'llama3.2:latest');
    await page.check('[data-testid=step-mode]');
    await page.fill('[data-testid=prompt]', 'Count from one to five.');
    await page.click('[data-testid=run]');
    await page.waitForTimeout(4000);
    console.log('start', JSON.stringify(await dbg()));
    for (let i = 0; i < 4; i++) {
      await page.getByRole('button', { name: 'Step token' }).click();
      for (const t of [100, 600, 1500]) {
        await page.waitForTimeout(t === 100 ? 100 : t - 100);
        console.log(`click${i} +${t}`, JSON.stringify(await dbg()));
      }
    }
  } finally {
    await app.close();
    ollama.kill();
  }
})();
