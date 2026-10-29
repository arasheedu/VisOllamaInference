// Smoke-test the packaged app: npx tsx e2e/probe-packaged.ts <exe>
import { _electron as electron } from '@playwright/test';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

(async () => {
  const userData = mkdtempSync(join(tmpdir(), 'llmviz-pkg-'));
  writeFileSync(join(userData, 'settings.json'), JSON.stringify({ source: 'mock', detail: 'layers', autosave: false }));
  const app = await electron.launch({
    executablePath: process.argv[2],
    env: { ...process.env, TRACE_PORT: '4797', VISUALIZER_USER_DATA: userData },
  });
  const page = await app.firstWindow();
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.waitForSelector('[data-testid=run]');
  const hub = await page.evaluate(() => (window as any).__viz.useUi.getState().hub);
  await page.selectOption('[data-testid=model]', 'mock-llama:1b');
  await page.click('[data-testid=run]');
  await page.waitForSelector('[data-testid=pipeline][data-phase=done]', { timeout: 60_000 });
  console.log(JSON.stringify({ hubListening: hub?.listening, port: hub?.port, phase: 'done', errors }));
  await app.close();
})();
