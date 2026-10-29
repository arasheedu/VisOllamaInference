import { _electron as electron, type ElectronApplication, type Page } from '@playwright/test';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

export const APP_DIR = resolve(import.meta.dirname, '..');

export async function launch(
  o: { port?: number; settings?: Record<string, unknown> } = {},
): Promise<{ app: ElectronApplication; page: Page; userData: string }> {
  const userData = mkdtempSync(join(tmpdir(), 'llmviz-'));
  if (o.settings) writeFileSync(join(userData, 'settings.json'), JSON.stringify(o.settings));
  const app = await electron.launch({
    args: [join(APP_DIR, 'out/main/index.js')],
    cwd: APP_DIR,
    env: {
      ...process.env,
      TRACE_PORT: String(o.port ?? 4790),
      VISUALIZER_USER_DATA: userData,
      ELECTRON_RENDERER_URL: '',
    },
  });
  const page = await app.firstWindow();
  await page.waitForLoadState('domcontentloaded');
  await page.setViewportSize({ width: 1600, height: 900 }).catch(() => {});
  return { app, page, userData };
}
